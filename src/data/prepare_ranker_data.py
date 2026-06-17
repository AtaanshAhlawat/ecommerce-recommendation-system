import pandas as pd
import numpy as np
from rectools import Columns
import os
from datetime import datetime
import pickle
from generate_candidate import generate_candidates_for_users, load_data_and_model

def prepare_training_data():
    print("=" * 70)
    print("PREPARING LIGHTGBM RANKER TRAINING DATA")
    print("=" * 70)
    
    # 1. Load SASRec artifacts and interactions
    sasrec_model, dataset, train_interactions, future_interactions = load_data_and_model()
    
    # 2. Define Part 1 Window (Aug 20 - Sep 3)
    max_date = future_interactions[Columns.Datetime].max()
    start_30d = max_date - pd.Timedelta(days=30)
    start_15d = max_date - pd.Timedelta(days=15)
    
    # Part 1: Users who interacted between start_30d and start_15d
    part1_gt = future_interactions[
        (future_interactions[Columns.Datetime] > start_30d) & 
        (future_interactions[Columns.Datetime] <= start_15d)
    ].copy()
    
    part1_users = list(part1_gt[Columns.User].unique())
    print(f"Part 1 users with future activity: {len(part1_users):,}")
    
    # 3. Generate Candidates for Part 1 users
    print("\nGenerating SASRec candidates for Ranker Training (Part 1)...")
    candidates_df = generate_candidates_for_users(
        sasrec_model, dataset, train_interactions, part1_gt,
        target_users=part1_users, k=500
    )
    
    if candidates_df.empty:
        print("Error: No candidates generated.")
        return
    
    # 4. Label candidates (Label = 1 if user actually interacted with the item in Part 1)
    print("\nLabeling candidates...")
    # Create a set of (user, item) pairs as ground truth for fast lookup
    gt_pairs = set(zip(part1_gt[Columns.User], part1_gt[Columns.Item]))
    
    def check_label(row):
        return 1 if (row[Columns.User], row[Columns.Item]) in gt_pairs else 0
    
    # Batch label check using merge is faster for millions of rows
    part1_gt_unique = part1_gt[[Columns.User, Columns.Item]].drop_duplicates()
    part1_gt_unique['target'] = 1
    
    labeled_data = pd.merge(
        candidates_df, 
        part1_gt_unique, 
        on=[Columns.User, Columns.Item], 
        how='left'
    )
    labeled_data['target'] = labeled_data['target'].fillna(0).astype(int)
    
    positives = labeled_data['target'].sum()
    print(f"Total candidates: {len(labeled_data):,}")
    print(f"Positive samples: {positives:,} ({positives/len(labeled_data):.2%})")
    
    # 5. Save the prepared data
    output_file = "ranker_train_data_part1.parquet"
    labeled_data.to_parquet(output_file, index=False)
    print(f"\nLabeled ranker training data saved to: {output_file}")
    
    # Also save the Part 2 ground truth for final eval
    part2_gt = future_interactions[future_interactions[Columns.Datetime] > start_15d].copy()
    part2_gt.to_csv("ranker_test_gt_part2.csv", index=False)
    print(f"Part 2 (Final Eval) ground truth saved to: ranker_test_gt_part2.csv")

if __name__ == "__main__":
    prepare_training_data()
