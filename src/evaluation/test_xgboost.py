import numpy as np
import pandas as pd
from datetime import datetime
from rectools import Columns
from xgboost_ranker import XGBoostFeatureBuilder, XGBoostRanker

print("="*70)
print("TRAINING XGBOOST RANKER FROM SASREC CANDIDATES")
print("="*70)

# ===== STEP 1: Load SASRec candidates =====
print("\nLoading SASRec candidates...")
candidates = pd.read_csv("sasrec_candidates_causal_14d_20260528_010447.csv")
print(f"Loaded {len(candidates)} candidates")
print(f"Users: {candidates['user_id'].nunique()}, Items: {candidates['item_id'].nunique()}")
print(f"Score range: [{candidates['sasrec_score'].min():.4f}, {candidates['sasrec_score'].max():.4f}]")

# ===== STEP 2: Load interactions (for features + targets) =====
print("\nLoading interactions...")
events = pd.read_csv("retail-rocket/raw/events.csv", 
                     dtype={'visitorid': 'int32', 'itemid': 'int32'})
events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')

# Create interaction weights
interaction_weights = {'view': 1.0, 'addtocart': 3.0, 'transaction': 5.0}
events['weight'] = events['event'].map(interaction_weights)

# Aggregate interactions
interactions = events.groupby(['visitorid', 'itemid']).agg({
    'weight': 'sum', 'timestamp': 'max'
}).reset_index().rename(columns={
    'visitorid': Columns.User, 'itemid': Columns.Item,
    'weight': Columns.Weight, 'timestamp': Columns.Datetime
})

# Filter active users/items (same as training)
min_interactions = 5
user_counts = interactions[Columns.User].value_counts()
item_counts = interactions[Columns.Item].value_counts()
interactions = interactions[
    interactions[Columns.User].isin(user_counts[user_counts >= min_interactions].index) &
    interactions[Columns.Item].isin(item_counts[item_counts >= min_interactions].index)
]

print(f"Loaded {len(interactions)} interactions")

# ===== STEP 3: Split train/test (temporal) =====
print("\nSplitting train/test...")
interactions = interactions.sort_values(Columns.Datetime)
split_idx = int(len(interactions) * 0.8)
train_interactions = interactions.iloc[:split_idx]
test_interactions = interactions.iloc[split_idx:]
current_time = train_interactions[Columns.Datetime].max()

print(f"Train: {len(train_interactions)}, Test: {len(test_interactions)}")

# ===== STEP 4: Add ground truth targets to candidates =====
print("\nAdding target labels...")
# Get test interactions for candidate users
test_user_items = test_interactions.groupby(Columns.User)[Columns.Item].apply(set).to_dict()
test_weights = test_interactions.groupby([Columns.User, Columns.Item])[Columns.Weight].sum().to_dict()

# Add targets
def get_target(row):
    user_items = test_user_items.get(row['user_id'], set())
    if row['item_id'] in user_items:
        weight = test_weights.get((row['user_id'], row['item_id']), 0)
        return min(weight, 5)  # Clip to 5 for NDCG
    return 0

candidates['target'] = candidates.apply(get_target, axis=1)

print(f"Positive targets: {(candidates['target'] > 0).sum()}/{len(candidates)} "
      f"({100*(candidates['target'] > 0).sum()/len(candidates):.1f}%)")

# ===== STEP 5: Build XGBoost features =====
print("\nBuilding features...")
# Load item features if available
try:
    item_props = pd.read_csv("retail-rocket/raw/item_properties_part1.csv", nrows=500000)
    category_tree = pd.read_csv("retail-rocket/raw/category_tree.csv")
    
    # Create minimal item features
    item_categories = item_props[item_props['property'] == 'categoryid'].copy()
    item_categories = item_categories[['itemid', 'value']].rename(
        columns={'itemid': 'item_id', 'value': 'category_id'})
    item_categories['category_id'] = pd.to_numeric(item_categories['category_id'], errors='coerce')
    items_df = item_categories.set_index('item_id')
except:
    print("Warning: Could not load item features, using None")
    items_df = None

# Initialize feature builder
feature_builder = XGBoostFeatureBuilder(train_interactions, items_df, None)

# Build features per user (vectorized)
training_data_list = []
candidate_users = candidates['user_id'].unique()

for user_id in candidate_users:
    user_candidates = candidates[candidates['user_id'] == user_id].copy()
    
    # Compute per-user normalization
    max_score = user_candidates['sasrec_score'].max()
    min_score = user_candidates['sasrec_score'].min()
    
    # Build features
    features_df = feature_builder.build_features_vectorized(
        user_candidates,
        current_time=current_time,
        max_score=max_score,
        min_score=min_score
    )
    training_data_list.append(features_df)

training_data = pd.concat(training_data_list, ignore_index=True)
# Fill NaN datetime columns that cause XGBoost to fail
datetime_cols = ['item_item_first_interaction', 'item_item_last_interaction']
for col in datetime_cols:
    if col in training_data.columns:
        training_data[col] = pd.to_numeric(training_data[col], errors='coerce').fillna(0)
print(f"Training data shape: {training_data.shape}")
print(f"Feature count: {len([c for c in training_data.columns if c not in ['user_id', 'item_id', 'target']])}")

# ===== STEP 6: Train XGBoost ranker =====
print("\nTraining XGBoost ranker...")

# Split train/validation by users
train_users = candidate_users[:int(len(candidate_users)*0.8)]
val_users = candidate_users[int(len(candidate_users)*0.8):]

train_df = training_data[training_data['user_id'].isin(train_users)]
val_df = training_data[training_data['user_id'].isin(val_users)]

print(f"Train: {len(train_df)} ({train_df['user_id'].nunique()} users)")
print(f"Val: {len(val_df)} ({val_df['user_id'].nunique()} users)")

# Initialize ranker
ranker = XGBoostRanker(params={
    'objective': 'rank:ndcg',
    'eval_metric': 'ndcg@10',
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'verbosity': 1
})

# Prepare validation data
val_X, val_y, val_groups = ranker.prepare_data(val_df)

# Train
print("\nTraining...")
results = ranker.train(train_df, validation_data=(val_X, val_y, val_groups), 
                       early_stopping_rounds=10)

# ===== STEP 7: Evaluate =====
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

metrics = ranker.evaluate(val_df, k=10)
print(f"\nNDCG@10: {metrics['ndcg@10']:.4f}")
print(f"Users evaluated: {metrics['num_users_evaluated']}")

# Feature importance
print("\nTop 15 Features:")
importance = ranker.get_feature_importance()
print(importance.head(15).to_string(index=False))

# ===== STEP 8: Test on sample user =====
print("\n" + "="*70)
print("SAMPLE PREDICTION")
print("="*70)

test_user = val_users[0]
user_data = training_data[training_data['user_id'] == test_user].copy()
predictions = ranker.predict(user_data)
user_data['xgb_score'] = predictions

print(f"\nUser {test_user} - Top 10 by XGBoost:")
top_xgb = user_data.sort_values('xgb_score', ascending=False).head(10)
print(top_xgb[['item_id', 'sasrec_score', 'xgb_score', 'rank_position', 'target']].to_string(index=False))

print(f"\nUser {test_user} - Top 10 by SASRec (original):")
top_sasrec = user_data.sort_values('rank_position').head(10)
print(top_sasrec[['item_id', 'sasrec_score', 'xgb_score', 'rank_position', 'target']].to_string(index=False))

print(f"\nRelevant items in XGBoost top-10: {(top_xgb['target'] > 0).sum()}/10")
print(f"Relevant items in SASRec top-10: {(top_sasrec['target'] > 0).sum()}/10")

# Save model
print("\nSaving model...")
ranker.save_model("xgboost_ranker_model.pkl")
print("✓ Model saved to: xgboost_ranker_model.pkl")

print("\n" + "="*70)
print("✓ XGBoost training complete!")
print("="*70)