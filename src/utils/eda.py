import numpy as np
import pandas as pd
import warnings
from pathlib import Path
import os
import threadpoolctl

warnings.filterwarnings('ignore')

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.nearest_neighbours import CosineRecommender

try:
    from lightfm import LightFM
except ModuleNotFoundError:
    pass

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import (
    ImplicitALSWrapperModel,
    ImplicitBPRWrapperModel,
    LightFMWrapperModel,
    PureSVDModel,
    ImplicitItemKNNWrapperModel,
    EASEModel
)

# For vector models optimized ranking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas");

RANDOM_STATE = 60

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

# Use smaller sample for testing
# interactions = interactions.sample(n=50000, random_state=RANDOM_STATE)
print(f"\nAfter sampling:")
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

# Prepare test users
print("\nPreparing test users...")

# Hot test users - have interactions and features
test_hot_users = interactions[Columns.User].unique()[:3].tolist()
print(f"Hot test users: {test_hot_users}")
for user_id in test_hot_users:
    user_interactions = interactions[interactions[Columns.User] == user_id]
    user_feats = user_features[user_features["id"] == user_id]
    print(f"User {user_id}: {len(user_interactions)} interactions, {len(user_feats)} features")

# Warm test users - find users with features but minimal interactions
# Get all users and their interaction counts
user_interaction_counts = interactions[Columns.User].value_counts()
all_users_with_features = set(user_features["id"].unique())

# Find users who have features but very few interactions (1-2)
users_with_minimal_interactions = []
for user_id in all_users_with_features:
    count = user_interaction_counts.get(user_id, 0)
    if 1 <= count <= 2:  # Users with very few interactions
        users_with_minimal_interactions.append(user_id)

# Use real users only - no synthetic fallback
test_warm_users = users_with_minimal_interactions[:2] if len(users_with_minimal_interactions) >= 2 else users_with_minimal_interactions

print(f"Warm test users: {test_warm_users}")
for user_id in test_warm_users:
    user_interactions = interactions[interactions[Columns.User] == user_id]
    user_feats = user_features[user_features["id"] == user_id]
    print(f"User {user_id}: {len(user_interactions)} interactions, {len(user_feats)} features")

# Cold test users - no features or interactions
test_cold_users = [99999997]  # Synthetic user ID
print(f"Cold test users: {test_cold_users}")

# Create dataset with both user and item features
print("\nCreating dataset with features...")
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

