import numpy as np
import os
import pandas as pd
import torch
import typing as tp
import warnings
import threadpoolctl
import re
import plotly.express as px

from lightning_fabric import seed_everything
from pathlib import Path

from rectools import Columns
from rectools.dataset import Dataset
from rectools.metrics import (
    MAP,
    CoveredUsers,
    AvgRecPopularity,
    Intersection,
    HitRate,
    Serendipity,
)
from rectools.models import PopularModel, EASEModel, SASRecModel, BERT4RecModel
from rectools.model_selection import TimeRangeSplitter, cross_validate
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet
from rectools.visuals import MetricsApp

warnings.simplefilter("ignore")

# Enable deterministic behaviour with CUDA >= 10.2
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import pytorch_lightning as pl
_orig_trainer_init = pl.Trainer.__init__
def _cpu_trainer_init(self, *a, **kw):
    kw["accelerator"] = "cpu"
    kw.pop("devices", None)
    _orig_trainer_init(self, *a, **kw)
pl.Trainer.__init__ = _cpu_trainer_init
torch.set_default_dtype(torch.float32)
# Random seed
RANDOM_STATE=60
torch.use_deterministic_algorithms(True)
seed_everything(RANDOM_STATE, workers=True)

# Load retail rocket data
print("Loading retail rocket data...")

# Load events data
events = pd.read_csv(
    "retail-rocket/raw/events.csv",
    dtype={
        'visitorid': 'int32',
        'itemid': 'int32', 
        'transactionid': 'object'
    }
)
print(f"Events shape: {events.shape}")
print(f"Event types: {events['event'].value_counts()}")

# Load category tree
category_tree = pd.read_csv("retail-rocket/raw/category_tree.csv")
print(f"Category tree shape: {category_tree.shape}")

# Load sample of item properties (for performance)
item_props_sample = pd.read_csv(
    "retail-rocket/raw/item_properties_part1.csv",
    nrows=500000  # Sample for performance
)
print(f"Item properties sample shape: {item_props_sample.shape}")

# Process interactions
print("\nProcessing interactions...")
# Convert timestamp to datetime
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# Create interaction weights
interaction_weights = {
    'view': 1.0,
    'addtocart': 3.0, 
    'transaction': 5.0
}

events['weight'] = events['event'].map(interaction_weights)

# Aggregate user-item interactions
interactions = events.groupby(['visitorid', 'itemid']).agg({
    'weight': 'sum',
    'timestamp': 'max'
}).reset_index()

# Rename columns to match rectools expectations
interactions = interactions.rename(columns={
    'visitorid': Columns.User,
    'itemid': Columns.Item,
    'weight': Columns.Weight,
    'timestamp': Columns.Datetime
})

print(f"Aggregated interactions shape: {interactions.shape}")
print(f"Unique users: {interactions[Columns.User].nunique()}")
print(f"Unique items: {interactions[Columns.Item].nunique()}")

# Filter for active users and items
min_user_interactions = 5
min_item_interactions = 5

user_counts = interactions[Columns.User].value_counts()
item_counts = interactions[Columns.Item].value_counts()

active_users = user_counts[user_counts >= min_user_interactions].index
popular_items = item_counts[item_counts >= min_item_interactions].index

interactions = interactions[
    interactions[Columns.User].isin(active_users) &
    interactions[Columns.Item].isin(popular_items)
]

print(f"After filtering:")
print(f"Interactions shape: {interactions.shape}")
print(f"Unique users: {interactions[Columns.User].nunique()}")
print(f"Unique items: {interactions[Columns.Item].nunique()}")

# Use all interactions for best performance
# interactions = interactions.sample(n=50000, random_state=RANDOM_STATE)
print(f"\nUsing all interactions:")
print(f"Interactions shape: {interactions.shape}")
print(f"Unique users: {interactions[Columns.User].nunique()}")
print(f"Unique items: {interactions[Columns.Item].nunique()}")

# Create user features based on behavior patterns
print("\nCreating user features...")
user_stats = interactions.groupby(Columns.User).agg({
    Columns.Weight: ['sum', 'mean', 'count'],
    Columns.Datetime: ['min', 'max']
}).reset_index()

user_stats.columns = [Columns.User, 'total_weight', 'avg_weight', 'interaction_count', 'first_interaction', 'last_interaction']

# Calculate user activity level
user_stats['activity_span_days'] = (user_stats['last_interaction'] - user_stats['first_interaction']).dt.days
user_stats['activity_level'] = pd.cut(user_stats['interaction_count'], 
                                     bins=[0, 5, 20, float('inf')], 
                                     labels=['Low', 'Medium', 'High'])

# Create user features dataframe
user_features_frames = []

# Activity level feature
activity_frame = user_stats[[Columns.User, 'activity_level']].copy()
activity_frame.columns = ["id", "value"]
activity_frame["feature"] = "activity_level"
user_features_frames.append(activity_frame)

# Interaction count category feature
user_stats['interaction_category'] = pd.cut(user_stats['interaction_count'],
                                          bins=[0, 10, 50, float('inf')],
                                          labels=['Casual', 'Regular', 'Frequent'])
interaction_cat_frame = user_stats[[Columns.User, 'interaction_category']].copy()
interaction_cat_frame.columns = ["id", "value"]
interaction_cat_frame["feature"] = "interaction_category"
user_features_frames.append(interaction_cat_frame)

user_features = pd.concat(user_features_frames)
print(f"User features shape: {user_features.shape}")
print(user_features.head())

# Create item features from category tree and properties
print("\nCreating item features...")

# Get category information for items in our interactions
item_categories = item_props_sample[item_props_sample['property'] == 'categoryid'].copy()
item_categories = item_categories[['itemid', 'value']].rename(columns={'itemid': 'item_id', 'value': 'category_id'})

# Convert category_id to numeric for merging
item_categories['category_id'] = pd.to_numeric(item_categories['category_id'], errors='coerce')

# Filter to only items in our interactions
item_categories = item_categories[item_categories['item_id'].isin(interactions[Columns.Item])]

# Merge with category tree to get parent categories if available
item_categories = item_categories.merge(category_tree, left_on='category_id', right_on='categoryid', how='left')

# Create category features
item_features_frames = []

# Primary category feature
category_frame = item_categories[['item_id', 'category_id']].copy()
category_frame.columns = ["id", "value"]
category_frame["feature"] = "category"
item_features_frames.append(category_frame)

# Parent category feature (if available)
if 'parentid' in item_categories.columns:
    parent_categories = item_categories[item_categories['parentid'].notna()].copy()
    parent_frame = parent_categories[['item_id', 'parentid']].copy()
    parent_frame.columns = ["id", "value"]
    parent_frame["feature"] = "parent_category"
    item_features_frames.append(parent_frame)

# Item popularity feature
item_popularity = interactions.groupby(Columns.Item).agg({
    Columns.User: 'count',
    Columns.Weight: 'sum'
}).reset_index()
item_popularity.columns = ['item_id', 'user_count', 'total_weight']

item_popularity['popularity_level'] = pd.cut(item_popularity['user_count'],
                                            bins=[0, 10, 50, float('inf')],
                                            labels=['Unpopular', 'Popular', 'Very_Popular'])

popularity_frame = item_popularity[['item_id', 'popularity_level']].copy()
popularity_frame.columns = ["id", "value"]
popularity_frame["feature"] = "popularity_level"
item_features_frames.append(popularity_frame)

if item_features_frames:
    item_features = pd.concat(item_features_frames)
    print(f"Item features shape: {item_features.shape}")
    print(item_features.head())
else:
    item_features = pd.DataFrame(columns=["id", "value", "feature"])
    print("No item features created")

# Construct dataset
print("\nConstructing dataset...")
dataset = Dataset.construct(
    interactions_df=interactions,
    user_features_df=user_features,
    cat_user_features=["activity_level", "interaction_category"],
    item_features_df=item_features if len(item_features) > 0 else None,
    cat_item_features=["category", "parent_category", "popularity_level"] if len(item_features) > 0 else None,
)

print(f"Dataset created successfully")
print(f"User count: {dataset.user_id_map.size}")
print(f"Item count: {dataset.item_id_map.size}")

# Prepare test user
test_user = interactions[Columns.User].iloc[0]  # First user in dataset
print(f"\nTest user: {test_user}")
print(f"User interactions: {interactions[interactions[Columns.User] == test_user].shape}")
print(interactions[interactions[Columns.User] == test_user].head(2))

sasrec = SASRecModel(
    session_max_len=50,
    loss="BCE",
    n_negatives=200,
    n_factors=128,
    n_blocks=2,
    n_heads=4,
    dropout_rate=0.1,
    lr=0.001,
    batch_size=128,
    epochs=3,
    verbose=1,
    deterministic=True,
)

sasrec.fit(dataset)

# Define test item (first item in dataset)
test_item = interactions[Columns.Item].iloc[0]
print(f"\nTest item: {test_item}")

# Get item-to-item recommendations
recos = sasrec.recommend_to_items(
    target_items=[test_item],
    dataset=dataset,
    k=3,
    filter_itself=True,
    items_to_recommend=None,
)

print(f"\nItem-to-item recommendations for item {test_item}:")
print(recos)

print(f"\nTop 3 similar items to {test_item}:")
for i, row in recos.iterrows():
    print(f"  {i+1}. Item {row['item_id']} (score: {row['score']:.4f})")

