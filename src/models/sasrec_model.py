import numpy as np
import os
import pandas as pd
import torch
torch.backends.mps.is_available = lambda: False
import warnings
from datetime import datetime
from lightning_fabric import seed_everything

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import SASRecModel, load_model
from rectools.models.nn.item_net import CatFeaturesItemNet, IdEmbeddingsItemNet

warnings.simplefilter("ignore")

# Enable deterministic behaviour with CUDA >= 10.2
#os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Random seed
RANDOM_STATE = 60
torch.use_deterministic_algorithms(True)
seed_everything(RANDOM_STATE, workers=True)

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"sasrec_training_{timestamp}.log"

def log_message(message):
    """Log message to both console and file"""
    print(message)
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

# Load retail rocket data
log_message("Loading retail rocket data...")

# Load events data
events = pd.read_csv(
    "retail-rocket/raw/events.csv",
    dtype={
        'visitorid': 'int32',
        'itemid': 'int32', 
        'transactionid': 'object'
    }
)
log_message(f"Events shape: {events.shape}")

# Process interactions
log_message("\nProcessing interactions...")
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# Create interaction weights
interaction_weights = {
    'view': 1.0,
    'addtocart': 3.0, 
    'transaction': 5.0
}
events['weight'] = events['event'].map(interaction_weights)

log_message("\nPreparing raw sequential interactions (NO aggregation)...")

interactions = events.rename(columns={
    'visitorid': Columns.User,
    'itemid': Columns.Item,
    'timestamp': Columns.Datetime,
    'weight': Columns.Weight
})[[Columns.User, Columns.Item, Columns.Datetime, Columns.Weight]]

# Sort strictly by time (CRITICAL)
interactions = interactions.sort_values(Columns.Datetime).reset_index(drop=True)

log_message(f"Raw interactions for SASRec: {interactions.shape}")


# Filter for active users and items
log_message("\nFiltering and splitting data...")
from rectools.model_selection import TimeRangeSplitter
from rectools.dataset import Interactions

log_message("\nPerforming time-based split...")

interactions_rt = Interactions(interactions)

splitter = TimeRangeSplitter(test_size="30D", n_splits=1)

train_idx, test_idx, _ = next(splitter.split(interactions_rt))

train_interactions = interactions.iloc[train_idx].copy()
future_interactions = interactions.iloc[test_idx].copy()

log_message(f"Training interactions end at: {train_interactions[Columns.Datetime].max()}")
log_message(f"Training interactions: {len(train_interactions):,}")
log_message(f"Future interactions: {len(future_interactions):,}")
assert train_interactions[Columns.Datetime].max() < future_interactions[Columns.Datetime].min()
log_message("✓ Time split verified: no overlap between train and future")

# Save future interactions for ranking stage
future_interactions.to_csv(
    "future_interactions_ground_truth2.csv",
    index=False
)

log_message(f"Future interactions saved to: future_interactions_ground_truth2.csv")

log_message(f"Original interactions: {len(interactions)}")
log_message(f"Training interactions (cut at {train_interactions[Columns.Datetime].max()}): {len(train_interactions)}")

# Filter for active users and items within the TRAINING window
# ===============================
# FILTER ACTIVE USERS / ITEMS
# ===============================

min_user_interactions = 2
min_item_interactions = 3

user_counts = train_interactions.groupby(Columns.User).size()
item_counts = train_interactions.groupby(Columns.Item).size()

active_users = user_counts[user_counts >= min_user_interactions].index
popular_items = item_counts[item_counts >= min_item_interactions].index

train_interactions = train_interactions[
    train_interactions[Columns.User].isin(active_users) &
    train_interactions[Columns.Item].isin(popular_items)
]

log_message(f"Filtered training interactions: {train_interactions.shape}")
log_message(f"Active users: {len(active_users):,}")
log_message(f"Popular items: {len(popular_items):,}")


log_message(f"Filtered Training interactions: {train_interactions.shape}")
# Create user features (using ONLY training data)
log_message("\nCreating user features from training data...")
user_stats = train_interactions.groupby(Columns.User).agg({
    Columns.Weight: ['sum', 'count'],
    Columns.Datetime: ['min', 'max']
}).reset_index()
user_stats.columns = [Columns.User, 'total_weight', 'interaction_count', 'first_interaction', 'last_interaction']

# SIMPLIFY: Use just activity_level based on interaction count
# Use quantiles for balanced bins
if len(user_stats) >= 3:
    quantiles = user_stats['interaction_count'].quantile([0.33, 0.66]).values
    if len(quantiles) == 2 and quantiles[0] != quantiles[1]:
        bins = [0, quantiles[0], quantiles[1], float('inf')]
    else:
        bins = [0, 5, 10, float('inf')]  # Fallback
else:
    bins = [0, 3, 6, float('inf')]  # Simple bins for few users

user_stats['activity_level'] = pd.cut(
    user_stats['interaction_count'],
    bins=bins,
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

user_features_frames = []
frame = user_stats[[Columns.User, 'activity_level']].copy()
frame.columns = ["id", "value"]
frame["feature"] = "activity_level"
user_features_frames.append(frame)

user_features = pd.concat(user_features_frames)
log_message(f"Created activity_level feature for {len(user_features)} users")

# Create item features (using ONLY training data)
log_message("\nCreating item features from training data...")

# Get all items in training
train_items = train_interactions[Columns.Item].unique()
train_items_set = set(train_items)
log_message(f"Number of unique items in training: {len(train_items_set)}")

# Create a dictionary to store categories as we find them
# CRITICAL: We need the LATEST category value that is <= train_end_ts
train_end_ts_ms = int(train_interactions[Columns.Datetime].max().timestamp() * 1000)
log_message(f"Research cutoff timestamp (ms): {train_end_ts_ms}")

category_dict = {}
found_items = set()

# List of item property files
item_property_files = [
    "retail-rocket/raw/item_properties_part1.csv",
    "retail-rocket/raw/item_properties_part2.csv"
]

# Track missing files
missing_files = []

for file_path in item_property_files:
    # Check if file exists
    if not os.path.exists(file_path):
        log_message(f"WARNING: File {file_path} not found, skipping")
        missing_files.append(file_path)
        continue
    
    # If we already found all items, skip remaining files
    if len(found_items) == len(train_items_set):
        log_message(f"Found categories for all {len(train_items_set)} items, skipping {file_path}")
        break
    
    log_message(f"Processing {file_path}...")
    file_start_time = datetime.now()
    chunks_processed = 0
    items_found_in_file = 0
    
    try:
        # Process files in reverse to find latest property values first if files were sorted
        # but since they aren't strictly sorted by item+time, we'll process normally 
        # and store ALL valid instances, then pick the latest.
        # Actually, for memory efficiency, we'll stick to the current loop but update if a newer valid timestamp is found.
        
        for chunk in pd.read_csv(file_path, chunksize=100000):
            # Filter for category information AND check timestamp (LEAKAGE FIX)
            chunk = chunk[
                (chunk['property'] == 'categoryid') & 
                (chunk['timestamp'] <= train_end_ts_ms)
            ].copy()
            
            # Further filter for items in training
            chunk = chunk[chunk['itemid'].isin(train_items_set)]
            
            if not chunk.empty:
                # Store latest category per item
                for _, row in chunk.iterrows():
                    item_id = row['itemid']
                    cat_val = row['value']
                    ts = row['timestamp']
                    
                    # If first time seeing item or found a newer valid version
                    if item_id not in category_dict or ts > category_dict[item_id]['ts']:
                        category_dict[item_id] = {'val': cat_val, 'ts': ts}
                        found_items.add(item_id)
                        items_found_in_file += 1
            
            chunks_processed += 1
            if chunks_processed % 20 == 0:
                log_message(f"  Processed {chunks_processed} chunks, found {len(found_items)}/{len(train_items_set)} items")
            
            # Early stopping: if we found all items
            if len(found_items) == len(train_items_set):
                log_message(f"  Found all items after {chunks_processed} chunks")
                break
        
        file_time = (datetime.now() - file_start_time).total_seconds()
        log_message(f"Finished {file_path}: {chunks_processed} chunks, found {items_found_in_file} new items in {file_time:.1f}s")
        
    except Exception as e:
        log_message(f"ERROR processing {file_path}: {str(e)}")
        continue

# Report on missing files
if missing_files:
    log_message(f"WARNING: {len(missing_files)} files missing: {missing_files}")

# Convert dictionary to DataFrame
if category_dict:
    item_categories = pd.DataFrame({
        'item_id': list(category_dict.keys()),
        'category_id': [v['val'] for v in category_dict.values()]
    })
    log_message(f"Found valid categories for {len(item_categories)} unique items")
else:
    item_categories = pd.DataFrame(columns=['item_id', 'category_id'])
    log_message("WARNING: No valid category data found in any file within temporal constraints!")

# Clean and convert category IDs
item_categories['category_id'] = pd.to_numeric(item_categories['category_id'], errors='coerce')
item_categories = item_categories.dropna(subset=['category_id'])
item_categories['category_id'] = item_categories['category_id'].astype(int)

# CRITICAL: Ensure ALL training items have a category feature
# Create a DataFrame with ALL training items
all_items_df = pd.DataFrame({'item_id': train_items})

# Merge with found categories (left join keeps all items)
item_categories_complete = all_items_df.merge(
    item_categories,
    on='item_id',
    how='left'
)

# Fill missing categories with -1 (unknown category)
item_categories_complete['category_id'] = item_categories_complete['category_id'].fillna(-1).astype(str)

log_message(f"Category coverage: {len(category_dict)}/{len(train_items)} items have real categories")
log_message(f"{len(train_items) - len(category_dict)} items will use -1 (unknown category)")

# Prepare item features in rectools format - ONLY CATEGORY, NO POPULARITY!
item_features_frames = []

# Category feature (now includes ALL items)
cat_frame = item_categories_complete[['item_id', 'category_id']].copy()
cat_frame.columns = ["id", "value"]
cat_frame["feature"] = "category"
item_features_frames.append(cat_frame)
log_message(f"✓ Category feature: {len(cat_frame)} rows")
log_message(f"✓ Dropped popularity feature - SASRec learns item importance from interaction patterns")

# REMOVED ALL POPULARITY FEATURE CODE!

# Combine item features (only category now)
item_features = pd.concat(item_features_frames, ignore_index=True)

# Verify ALL items have features
items_in_features = item_features['id'].nunique()
if items_in_features == len(train_items):
    log_message(f"✓ SUCCESS: All {len(train_items)} items have category features")
else:
    log_message(f"✗ WARNING: Only {items_in_features}/{len(train_items)} items have features")
    # This should not happen with our complete approach
    
    # Find missing items
    all_item_ids = set(train_items)
    featured_item_ids = set(item_features['id'].unique())
    missing_items = all_item_ids - featured_item_ids
    
    if missing_items:
        log_message(f"Missing features for {len(missing_items)} items")
        # Add dummy features for missing items
        for item_id in list(missing_items)[:10]:  # Log first 10
            log_message(f"  - Item {item_id}")

# Add this BEFORE Dataset.construct():
user_coverage = user_features['id'].nunique() / train_interactions[Columns.User].nunique() * 100
item_coverage = item_features['id'].nunique() / train_interactions[Columns.Item].nunique() * 100

log_message(f"User feature coverage: {user_coverage:.2f}%")
log_message(f"Item feature coverage: {item_coverage:.2f}%")

if user_coverage < 99.9 or item_coverage < 99.9:
    log_message("ERROR: Incomplete feature coverage!")
    # Handle appropriately
# Construct dataset
log_message("\nConstructing dataset with clean features...")
dataset = Dataset.construct(
    interactions_df=train_interactions,
    user_features_df=user_features,
    cat_user_features=["activity_level"],
    item_features_df=item_features,
    cat_item_features=["category"],
)

# CRITICAL: Save dataset to preserve ID mappings (User/Item -> internal_id)
import pickle
dataset_filename = "sasrec_dataset5.pkl"
with open(dataset_filename, 'wb') as f:
    pickle.dump(dataset, f)
log_message(f"Dataset saved as: {dataset_filename}")

# Train SASRec Model
log_message("\nTraining sasrec_gbce_ids_and_cat...")
# Calculate optimal sequence length
user_seq_lengths = train_interactions.groupby(Columns.User).size()

#avg_seq_len = user_seq_lengths.mean()
#p95_seq_len = user_seq_lengths.quantile(0.95)
#session_max_len = min(50, int(p95_seq_len))

#log_message(
#f"Sequence stats — Avg: {avg_seq_len:.1f}, "
#f"P95: {p95_seq_len}, Using max_len: {session_max_len}"
#)

avg_seq_len = user_seq_lengths.mean()
p95_seq_len = user_seq_lengths.quantile(0.95)
session_max_len = min(50, int(p95_seq_len))  # Cap at 50, use 95th percentile
log_message(f"Sequence stats - Avg: {avg_seq_len:.1f}, P95: {p95_seq_len}, Using: {session_max_len}")

model = SASRecModel(
    deterministic=True,
    loss="gBCE",
    epochs=10,
    n_negatives=200,
    gbce_t=0.1,
    session_max_len=session_max_len,
    n_factors=256,
    n_blocks=2,
    n_heads=4,
    dropout_rate=0.2,
    lr=0.0005,
    batch_size=128,
    verbose=1,
    item_net_block_types=(IdEmbeddingsItemNet, CatFeaturesItemNet)
)

model.fit(dataset)

# Save model
model_filename = "sasrec_gbce_model5.pkl"
model.save(model_filename)
log_message(f"\nModel saved as: {model_filename}")

# Quick verify

# CRITICAL: Pick user from TRAIN set
test_user = train_interactions[Columns.User].iloc[0]

recos = model.recommend(
    users=[test_user],
    dataset=dataset,
    k=5,
    filter_viewed=True
)

log_message(f"Recommendations for user {test_user}:\n{recos}")

log_message("\n=== TRAINING COMPLETE ===")