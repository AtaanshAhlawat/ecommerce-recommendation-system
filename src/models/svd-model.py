import numpy as np
import pandas as pd
import warnings
import os
import threadpoolctl

warnings.filterwarnings('ignore')

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import PureSVDModel

# For vector models optimized ranking
os.environ["OPENBLAS_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(1, "blas")

RANDOM_STATE = 60

# Load retail rocket data
print("Loading retail rocket data...")

# Load events data
events = pd.read_csv(
    "data/events.csv",
    dtype={
        'visitorid': 'int32',
        'itemid': 'int32', 
        'transactionid': 'object'
    }
)
print(f"Events shape: {events.shape}")
print(f"Event types: {events['event'].value_counts()}")

# Load category tree
category_tree = pd.read_csv("data/category_tree.csv")
print(f"Category tree shape: {category_tree.shape}")

# Load item properties (both parts)
item_props_part1 = pd.read_csv("data/item_properties_part1.csv")
item_props_part2 = pd.read_csv("data/item_properties_part2.csv")
item_props_sample = pd.concat([item_props_part1, item_props_part2], ignore_index=True)
print(f"Item properties shape: {item_props_sample.shape}")

# Process interactions
print("\nProcessing interactions...")
# Convert timestamp to datetime
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# Create interaction weights
interaction_weights = {
    'view': 1,
    'addtocart': 3, 
    'transaction': 8
}

events['weight'] = events['event'].map(interaction_weights)

# Create full interaction matrix - keep all individual interactions
interactions = events[['visitorid', 'itemid', 'weight', 'timestamp', 'event']].copy()

# Rename columns to match rectools expectations
interactions = interactions.rename(columns={
    'visitorid': Columns.User,
    'itemid': Columns.Item,
    'weight': Columns.Weight,
    'timestamp': Columns.Datetime
})

print(f"Total interactions shape: {interactions.shape}")
print(f"Unique users: {interactions[Columns.User].nunique()}")
print(f"Unique items: {interactions[Columns.Item].nunique()}")

# Top 20 users by interaction count and unique products
print("\nTop 20 Users by Interaction Count:")
user_stats = interactions.groupby(Columns.User).agg({
    Columns.Item: ['count', 'nunique'],
    Columns.Weight: 'sum'
}).reset_index()

user_stats.columns = ['user_id', 'total_interactions', 'unique_products', 'total_weight']
user_stats = user_stats.sort_values('total_interactions', ascending=False).head(20)
print(user_stats.to_string(index=False))

# Filter users with less than 20 unique products
print("\nFiltering users with less than 20 unique products...")

users_with_20plus_items = interactions.groupby(Columns.User)[Columns.Item].nunique()
eligible_users = users_with_20plus_items[users_with_20plus_items >= 20].index

print(f"Users before filtering: {interactions[Columns.User].nunique()}")
print(f"Users with 20+ unique products: {len(eligible_users)}")

original_interactions = interactions.copy()
interactions = interactions[interactions[Columns.User].isin(eligible_users)]

print(f"Interactions before filtering: {len(original_interactions)}")
print(f"Interactions after filtering: {len(interactions)}")

# Calculate density
original_users = original_interactions[Columns.User].nunique()
original_items = original_interactions[Columns.Item].nunique()
filtered_users = interactions[Columns.User].nunique()
filtered_items = interactions[Columns.Item].nunique()

original_density = (len(original_interactions) / (original_users * original_items)) * 100
filtered_density = (len(interactions) / (filtered_users * filtered_items)) * 100

print(f"\nDensity Analysis:")
print(f"Original density: {original_density:.4f}%")
print(f"Filtered density: {filtered_density:.4f}%")

# Train/Test Split - Per-user temporal split (last 30% of interactions for test)
print("\nCreating per-user temporal train/test split...")

# Sort by timestamp
interactions = interactions.sort_values([Columns.User, Columns.Datetime])

train_list = []
test_list = []

for user_id, user_data in interactions.groupby(Columns.User):
    user_data = user_data.sort_values(Columns.Datetime)
    n_interactions = len(user_data)
    
    # Split: first 70% to train, last 30% to test
    split_idx = int(n_interactions * 0.7)
    
    # Ensure at least 1 interaction in each set
    if split_idx == 0:
        split_idx = 1
    if split_idx == n_interactions:
        split_idx = n_interactions - 1
    
    train_list.append(user_data.iloc[:split_idx])
    test_list.append(user_data.iloc[split_idx:])

train_interactions = pd.concat(train_list, ignore_index=True)
test_interactions = pd.concat(test_list, ignore_index=True)

print(f"Train set:")
print(f"  Interactions: {len(train_interactions)} ({len(train_interactions)/len(interactions)*100:.1f}%)")
print(f"  Users: {train_interactions[Columns.User].nunique()}")
print(f"  Items: {train_interactions[Columns.Item].nunique()}")

print(f"Test set:")
print(f"  Interactions: {len(test_interactions)} ({len(test_interactions)/len(interactions)*100:.1f}%)")
print(f"  Users: {test_interactions[Columns.User].nunique()}")
print(f"  Items: {test_interactions[Columns.Item].nunique()}")

# Fixed: Define train_users and test_users
train_users = train_interactions[Columns.User].unique()
test_users = test_interactions[Columns.User].unique()

train_users_set = set(train_users)
test_users_set = set(test_users)
common_users = train_users_set.intersection(test_users_set)

print(f"\nUser overlap analysis:")
print(f"  Users in both train and test: {len(common_users)} ({len(common_users)/len(train_users_set)*100:.1f}%)")

# Event type distribution
train_event_dist = train_interactions['event'].value_counts(normalize=True)
test_event_dist = test_interactions['event'].value_counts(normalize=True)

print(f"\nEvent type distribution:")
print("Train:")
for event_type, proportion in train_event_dist.items():
    print(f"  {event_type}: {proportion:.3f}")
print("Test:")
for event_type, proportion in test_event_dist.items():
    print(f"  {event_type}: {proportion:.3f}")

# Create train dataset
print("\nCreating train dataset...")
train_dataset = Dataset.construct(
    interactions_df=train_interactions[[Columns.User, Columns.Item, Columns.Weight, Columns.Datetime]]
)

print(f"Train dataset created")
print(f"User count: {train_dataset.user_id_map.size}")
print(f"Item count: {train_dataset.item_id_map.size}")

# Matrix density
total_possible = train_dataset.user_id_map.size * train_dataset.item_id_map.size
density = (len(train_interactions) / total_possible) * 100

print(f"\nTrain Matrix Density: {density:.4f}%")
print(f"Sparsity: {100 - density:.4f}%")

# Train PureSVD model with reasonable hyperparameters
print("\nTraining PureSVD model...")
model = PureSVDModel(
    factors=128,
    tol=1e-4,
    maxiter=100,
    random_state=42,
    verbose=1
)
model.fit(train_dataset)

# Evaluate on test set ONLY
print("\n=== Model Evaluation ===")

# Get test users that exist in training data (can't recommend for cold users)
test_users_in_train = test_interactions[
    test_interactions[Columns.User].isin(train_dataset.user_id_map.external_ids)
][Columns.User].unique()

print(f"Test users in training data: {len(test_users_in_train)}")

# Generate recommendations for test users
test_recos = model.recommend(
    users=test_users_in_train,
    dataset=train_dataset,
    k=10,
    filter_viewed=True,
)

# Create ground truth from test interactions
test_ground_truth_list = []
for user_id, items in test_interactions[
    test_interactions[Columns.User].isin(test_users_in_train)
].groupby(Columns.User)[Columns.Item].apply(list).items():
    for item_id in items:
        test_ground_truth_list.append({
            Columns.User: user_id,
            Columns.Item: item_id
        })
test_ground_truth_df = pd.DataFrame(test_ground_truth_list)

# Calculate metrics
from rectools.metrics import Precision, Recall, MAP, NDCG

precision = Precision(k=10)
recall = Recall(k=10)
map_metric = MAP(k=10)
ndcg = NDCG(k=10)

precision_score = precision.calc(test_recos, test_ground_truth_df)
recall_score = recall.calc(test_recos, test_ground_truth_df)
map_score = map_metric.calc(test_recos, test_ground_truth_df)
ndcg_score = ndcg.calc(test_recos, test_ground_truth_df)

print(f"\nEvaluation Metrics (Top-10):")
print(f"Precision@10: {precision_score:.4f}")
print(f"Recall@10: {recall_score:.4f}")
print(f"MAP@10: {map_score:.4f}")
print(f"NDCG@10: {ndcg_score:.4f}")

# Coverage analysis
print(f"\nCoverage Analysis:")
total_items = train_dataset.item_id_map.size
recommended_items = set(test_recos[Columns.Item].unique())
coverage = len(recommended_items) / total_items
print(f"Item Coverage: {coverage:.4f} ({len(recommended_items)}/{total_items} items)")

# Diversity analysis
print(f"\nDiversity Analysis:")
user_recommendations = test_recos.groupby(Columns.User)[Columns.Item].apply(list)
avg_unique_items_per_user = user_recommendations.apply(len).mean()
print(f"Average unique items per user: {avg_unique_items_per_user:.1f}")

# Cold-start analysis
cold_users = len(test_users) - len(test_users_in_train)
print(f"\nCold-Start Analysis:")
print(f"Cold users (not in training): {cold_users}")
print(f"Coverage: {len(test_users_in_train)/len(test_users)*100:.1f}% of test users")

# Sample recommendations
print(f"\nSample Recommendations (5 test users):")
sample_users = test_users_in_train[:5]
sample_recos = model.recommend(
    users=sample_users,
    dataset=train_dataset,
    k=10,
    filter_viewed=True,
)
print(sample_recos)

print("\n=== Training Complete ===")