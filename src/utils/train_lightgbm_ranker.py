"""
Leakage-Free LightGBM Ranker Training Pipeline
Strictly enforces temporal causality across all feature construction
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from rectools import Columns
from lightgbm_ranker import LightGBMFeatureBuilder, LightGBMRanker

print("="*70)
print("TRAINING LIGHTGBM RANKER - LEAKAGE-FREE PIPELINE")
print("="*70)

# ===== STEP 1: Load Prepared Data =====
print("\nLoading prepared ranker training data...")
train_candidates = pd.read_parquet("ranker_train_data_part1.parquet")
print(f"Train candidates: {len(train_candidates):,} rows")
print(f"Positive samples: {train_candidates['target'].sum():,} ({train_candidates['target'].mean():.2%})")

# ===== STEP 2: Load Historical Context (for features) =====
print("\nLoading historical interactions for feature building...")
with open("sasrec_dataset6.pkl", "rb") as f:
    dataset = pickle.load(f)

history_interactions = dataset.interactions.df

# Get item features - handle SparseFeatures object
try:
    # Try to get sparse features as a dataframe
    items_sparse = dataset.item_features
    if hasattr(items_sparse, 'df'):
        items_df = items_sparse.df
    else:
        # Convert sparse features to dataframe
        items_df = pd.DataFrame({
            'item_id': items_sparse.ids,
            'feature': items_sparse.names,
            'value': items_sparse.values.flatten() if hasattr(items_sparse.values, 'flatten') else items_sparse.values
        })
        # Pivot to get one row per item
        items_df = items_df.pivot(index='item_id', columns='feature', values='value').reset_index()
except Exception as e:
    print(f"Warning: Could not load item features: {e}")
    items_df = pd.DataFrame()

print(f"History interactions: {len(history_interactions):,}")
print(f"History ends at: {history_interactions[Columns.Datetime].max()}")

# ===== STEP 3: Define Temporal Windows (CRITICAL for No Leakage) =====
print("\nDefining temporal boundaries...")
# Part 1 window: Days 30-15 from the end
max_date = history_interactions[Columns.Datetime].max()
part1_end = max_date + pd.Timedelta(days=15)  # End of Part 1

print(f"Training window ends: {max_date}")
print(f"Part 1 (Ranker Train) ends: {part1_end}")
print(f"Feature cutoff for training: {part1_end}")

# ===== STEP 4: Build Features (Causal Feature Builder) =====
print("\nBuilding causal features for training...")
# Feature builder sees ONLY history (pre-Aug 20)
# current_time set to Part 1 end for recency features
fb_train = LightGBMFeatureBuilder(
    history_interactions,
    items_df,
    None,  # No SASRec model needed for feature building
    current_time=part1_end
)

# Build features per user (vectorized)
print("Extracting features...")
train_features = []
unique_users = train_candidates[Columns.User].unique()

for user in unique_users:
    user_cands = train_candidates[train_candidates[Columns.User] == user].copy()
    
    # Vectorized feature building with correct normalization
    features = fb_train.build_features_vectorized(
        user_cands,
        current_time=part1_end,
        max_score=user_cands['sasrec_score'].max(),
        min_score=user_cands['sasrec_score'].min()
    )
    train_features.append(features)

train_feat_df = pd.concat(train_features, ignore_index=True)
print(f"Feature matrix shape: {train_feat_df.shape}")

# ===== STEP 5: Train LightGBM Ranker =====
print("\nTraining LightGBM ranker...")
ranker = LightGBMRanker(params={
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'random_state': 42,
    'ndcg_eval_at': [10],
    'min_data_in_leaf': 1
})

# Train (no validation for now, we'll use Part 2 for final test)
# CRITICAL FIX: Add class weighting to handle 0.12% positive rate
pos_count = (train_feat_df['target'] == 1).sum()
neg_count = (train_feat_df['target'] == 0).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"Class balance - Positives: {pos_count}, Negatives: {neg_count}")
print(f"Using scale_pos_weight: {scale_pos_weight:.1f}")

ranker.params['scale_pos_weight'] = scale_pos_weight
ranker.train(train_feat_df, target_col='target')

# ===== STEP 6: Feature Importance =====
print("\nTop 15 Most Important Features:")
importance = ranker.get_feature_importance()
print(importance.head(15).to_string(index=False))

# ===== STEP 7: Save Model =====
print("\nSaving trained ranker...")
ranker.save_model("lightgbm_ranker_causal.pkl")
print("✓ Model saved to: lightgbm_ranker_causal.pkl")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE - Ready for Part 2 evaluation")
print("="*70)
