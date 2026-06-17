import pandas as pd
import numpy as np
from rectools import Columns
import pickle
import os

def build_ranker_features(candidates_file, interactions_df, items_df, current_time, output_file):
    print(f"Building features for {candidates_file}...")
    df = pd.read_parquet(candidates_file)
    
    # 1. User Features (Activity)
    user_stats = interactions_df.groupby(Columns.User).agg({
        Columns.Item: 'count',
        Columns.Datetime: 'max'
    }).rename(columns={
        Columns.Item: 'user_activity_count',
        Columns.Datetime: 'user_last_interaction'
    })
    
    # 2. Item Features (Popularity)
    item_stats = interactions_df.groupby(Columns.Item).size().reset_index(name='item_popularity')
    
    # 3. Join with Items (for category)
    item_metadata = pd.DataFrame(columns=[Columns.Item, "category_id"])
    
    # 4. Merge all into candidates
    df = df.merge(user_stats, on=Columns.User, how='left')
    df = df.merge(item_stats, on=Columns.Item, how='left')
    df = df.merge(item_metadata, on=Columns.Item, how='left')
    
    # 5. Temporal Features (Recency)
    df['user_recency_hours'] = (current_time - df['user_last_interaction']).dt.total_seconds() / 3600
    df['user_recency_hours'] = df['user_recency_hours'].fillna(df['user_recency_hours'].max())
    
    # 6. Fill missing
    df['user_activity_count'] = df['user_activity_count'].fillna(0)
    df['item_popularity'] = df['item_popularity'].fillna(0)
    
    # 7. Drop helper columns
    df = df.drop(columns=['user_last_interaction'])
    
    print(f"Features built. Shape: {df.shape}")
    df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    # Load dataset artifacts
    with open("sasrec_dataset6.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    train_interactions = dataset.interactions.df
    items_df = pd.DataFrame() # This contains our causal categories
    print(type(dataset.item_features)) #-------------------------------------------------------------------
    print(dir(dataset.item_features))     #-------------------------------------------------------
    if 'item_id' not in items_df.columns:
        # Rectools internal format adjustment
        items_df = items_df.reset_index().rename(columns={'id': 'item_id', 'value': 'category_id'})
    
    # Build for Part 1 (Train) - Current time is Sep 3
    cutoff_mid = train_interactions[Columns.Datetime].max() + pd.Timedelta(days=15)
    build_ranker_features(
        "ranker_train_data_part1.parquet", 
        train_interactions, 
        items_df, 
        cutoff_mid, 
        "ranker_train_features_part1.parquet"
    )
