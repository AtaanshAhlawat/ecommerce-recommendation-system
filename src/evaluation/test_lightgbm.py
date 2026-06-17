import numpy as np
import pandas as pd
from datetime import datetime
from rectools import Columns
from lightgbm_ranker import LightGBMFeatureBuilder, LightGBMRanker

print("="*70)
print("TRAINING LIGHTGBM RANKER FROM SASREC CANDIDATES")
print("="*70)

# ===== STEP 1: Load Causal Artifacts (The Source of Truth) =====
print("Loading causal artifacts from SASRec stage...")
import os
import glob

# 1. Load the exact Dataset object (maps the "History" world)
import pickle
if not os.path.exists("sasrec_dataset2.pkl"):
    print("WARNING: sasrec_dataset2.pkl not found. Falling back to raw loading for now.")
    # Fallback logic if user hasn't run the updated sasrec_model.py yet
    dataset = None
    history_interactions = pd.DataFrame() 
else:
    with open("sasrec_dataset2.pkl", 'rb') as f:
        dataset = pickle.load(f)
    history_interactions = dataset.interactions.df

# 2. Load the future ground truth (The "Future" world)
future_path = "future_interactions_ground_truth.csv"
if not os.path.exists(future_path):
    print(f"WARNING: {future_path} not found.")
    future_interactions_all = pd.DataFrame()
else:
    future_interactions_all = pd.read_csv(future_path)
    future_interactions_all[Columns.Datetime] = pd.to_datetime(future_interactions_all[Columns.Datetime])

# 3. Load latest generated candidates
print("\nLoading latest causal candidates...")
candidate_files = glob.glob("sasrec_candidates_causal_7d_*.csv")
if not candidate_files:
    print("WARNING: No causal candidate files found. Using older ones if possible.")
    candidate_files = glob.glob("sasrec_candidates_all_users_*.csv")

if candidate_files:
    latest_candidate_file = max(candidate_files)
    candidates = pd.read_csv(latest_candidate_file)
    print(f"Loaded {len(candidates)} candidates from {latest_candidate_file}")
else:
    candidates = pd.DataFrame(columns=['user_id', 'item_id', 'sasrec_score', 'rank_position'])
    print("CRITICAL: No candidates found at all.")

# ===== STEP 2: Partition Future Window into Train/Eval Labels =====
print("\nPartitioning Future window for disjoint Ranker Training and Evaluation...")

if future_interactions_all.empty:
    print("WARNING: future_interactions_all is empty. Pipeline will fail.")
    train_label_interactions = pd.DataFrame()
    eval_label_interactions = pd.DataFrame()
else:
    future_interactions_all = future_interactions_all.sort_values(Columns.Datetime)
    t_min = future_interactions_all[Columns.Datetime].min()
    t_max = future_interactions_all[Columns.Datetime].max()
    mid_point = t_min + (t_max - t_min) / 2

    train_label_interactions = future_interactions_all[future_interactions_all[Columns.Datetime] < mid_point].copy()
    eval_label_interactions = future_interactions_all[future_interactions_all[Columns.Datetime] >= mid_point].copy()

    # Ensure ID types match
    train_label_interactions[Columns.User] = train_label_interactions[Columns.User].astype(np.int64)
    train_label_interactions[Columns.Item] = train_label_interactions[Columns.Item].astype(np.int64)
    eval_label_interactions[Columns.User] = eval_label_interactions[Columns.User].astype(np.int64)
    eval_label_interactions[Columns.Item] = eval_label_interactions[Columns.Item].astype(np.int64)

    print(f"History data ends at: {history_interactions[Columns.Datetime].max()}")
    print(f"Ranker Train Labels: {train_label_interactions[Columns.Datetime].min()} to {train_label_interactions[Columns.Datetime].max()} ({len(train_label_interactions)} rows)")
    print(f"Ranker Eval Labels: {eval_label_interactions[Columns.Datetime].min()} to {eval_label_interactions[Columns.Datetime].max()} ({len(eval_label_interactions)} rows)")

# ===== STEP 3: Build Training & Evaluation Sets =====
print("\nBuilding Ranker datasets...")

def get_targets_from_interactions(df_interactions):
    if df_interactions.empty: return {}, {}
    weights = df_interactions.groupby([Columns.User, Columns.Item])[Columns.Weight].sum().to_dict()
    user_items = df_interactions.groupby(Columns.User)[Columns.Item].apply(set).to_dict()
    return weights, user_items

train_weights, train_user_items = get_targets_from_interactions(train_label_interactions)
eval_weights, eval_user_items = get_targets_from_interactions(eval_label_interactions)

# Select users for Pilot
all_candidate_users = candidates['user_id'].unique() if not candidates.empty else []
train_users_potential = [u for u in all_candidate_users if u in train_user_items]
eval_users_potential = [u for u in all_candidate_users if u in eval_user_items]

# Disjoint users for training and validation
train_users = train_users_potential[:200]
eval_users = [u for u in eval_users_potential if u not in train_users][:50]

print(f"PILOT RUN: {len(train_users)} train users, {len(eval_users)} eval users")

# Filter candidates and apply labels
train_data = candidates[candidates['user_id'].isin(train_users)].copy()
train_data['target'] = train_data.apply(
    lambda r: min(train_weights.get((r['user_id'], r['item_id']), 0), 5), axis=1
) if not train_data.empty else pd.Series()

eval_candidates = candidates[candidates['user_id'].isin(eval_users)].copy()
eval_candidates['target'] = eval_candidates.apply(
    lambda r: min(eval_weights.get((r['user_id'], r['item_id']), 0), 5), axis=1
) if not eval_candidates.empty else pd.Series()

# DIAGNOSTIC
def check_hits(df, user_items_dict, name):
    if df.empty: return
    hits = 0
    total_users = df['user_id'].nunique()
    for user_id in df['user_id'].unique():
        user_cands = set(df[df['user_id'] == user_id]['item_id'])
        actual_items = user_items_dict.get(user_id, set())
        if user_cands.intersection(actual_items):
            hits += 1
    print(f"{name} Hit@500: {hits}/{total_users} users ({100*hits/total_users:.1f}%)")

check_hits(train_data, train_user_items, "Train")
check_hits(eval_candidates, eval_user_items, "Eval")

# Positives augmentation for training
positives_train = train_label_interactions[train_label_interactions[Columns.User].isin(train_users)][[Columns.User, Columns.Item, Columns.Weight]].copy()
if not positives_train.empty:
    positives_train = positives_train.rename(columns={Columns.User: 'user_id', Columns.Item: 'item_id', Columns.Weight: 'target'})
    positives_train['target'] = positives_train['target'].clip(upper=5).astype(int)
    positives_train['sasrec_score'] = train_data['sasrec_score'].min() - 1.0 if not train_data.empty else 0
    positives_train['rank_position'] = train_data['rank_position'].max() + 1 if not train_data.empty else 1
    train_data = pd.concat([train_data, positives_train], ignore_index=True)
    train_data = train_data.sort_values('target', ascending=False).drop_duplicates(['user_id', 'item_id'])

print(f"Final Train data: {len(train_data)} rows")

# ===== STEP 4: Build Features (Causal 3-Window Split) =====
print("\nBuilding features with correct historical context...")

# Load item features/metadata if available
try:
    item_props = pd.read_csv("retail-rocket/raw/item_properties_part1.csv", nrows=500000)
    item_categories = item_props[item_props['property'] == 'categoryid'].copy()
    item_categories = item_categories[['itemid', 'value']].rename(columns={'itemid': 'item_id', 'value': 'category_id'})
    item_categories['category_id'] = pd.to_numeric(item_categories['category_id'], errors='coerce')
    items_df = item_categories.set_index('item_id')
except:
    items_df = None

# Feature Builder for Training: Views only History (Window 1)
fb_train = LightGBMFeatureBuilder(history_interactions, items_df, None)
current_time_train = history_interactions[Columns.Datetime].max()

# Feature Builder for Eval: Views History + Train Window (Window 1 + Window 2)
eval_history = pd.concat([history_interactions, train_label_interactions])
fb_eval = LightGBMFeatureBuilder(eval_history, items_df, None)
current_time_eval = eval_history[Columns.Datetime].max()

def build_features_for_df(df, fb, current_time):
    if df.empty: return pd.DataFrame()
    data_list = []
    unique_users = df['user_id'].unique()
    for i, user_id in enumerate(unique_users):
        user_cands = df[df['user_id'] == user_id].copy()
        features = fb.build_features_vectorized(
            user_cands, current_time=current_time,
            max_score=user_cands['sasrec_score'].max(),
            min_score=user_cands['sasrec_score'].min()
        )
        data_list.append(features)
    return pd.concat(data_list, ignore_index=True)

print("Building train features...")
train_feat_df = build_features_for_df(train_data, fb_train, current_time_train)

print("Building eval features...")
eval_feat_df = build_features_for_df(eval_candidates, fb_eval, current_time_eval)

# Ensure sort by user for LightGBM
if not train_feat_df.empty:
    train_feat_df = train_feat_df.sort_values('user_id').reset_index(drop=True)
if not eval_feat_df.empty:
    eval_feat_df = eval_feat_df.sort_values('user_id').reset_index(drop=True)

print(f"Train features: {train_feat_df.shape if not train_feat_df.empty else '0'}")
print(f"Eval features: {eval_feat_df.shape if not eval_feat_df.empty else '0'}")

# ===== STEP 4.5: Align Columns (CRITICAL for LightGBM) =====
print("\nAligning feature columns across datasets...")
# Get union of all feature columns (excluding metadata)
exclude_cols = {'user_id', 'item_id', 'target'}
all_cols = sorted(list(set(train_feat_df.columns).union(set(eval_feat_df.columns))))

# Ensure both have all columns in the same order
for df in [train_feat_df, eval_feat_df]:
    if df.empty: continue
    for col in all_cols:
        if col not in df.columns:
            df[col] = 0
            
# Select columns in identical order
train_feat_df = train_feat_df[all_cols] if not train_feat_df.empty else train_feat_df
eval_feat_df = eval_feat_df[all_cols] if not eval_feat_df.empty else eval_feat_df

print(f"Aligned Train features: {train_feat_df.shape}")
print(f"Aligned Eval features: {eval_feat_df.shape}")

# ===== STEP 5: Train LightGBM ranker =====
print("\nTraining LightGBM ranker...")

ranker = LightGBMRanker(params={
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42,
    'min_data_in_leaf': 1
})

# Prepare validation data
val_X, val_y, val_groups = ranker.prepare_data(eval_feat_df)

# Train
print("\nTraining...")
results = ranker.train(train_feat_df, validation_data=(val_X, val_y, val_groups), 
                       early_stopping_rounds=10)

# ===== STEP 6: Evaluate =====
print("\n" + "="*70)
print("EVALUATION")
print("="*70)

metrics = ranker.evaluate(eval_feat_df, k=10)
print(f"\nNDCG@10: {metrics['ndcg@10']:.4f}")
print(f"Users evaluated: {metrics['num_users_evaluated']}")

# Feature importance
print("\nTop 15 Features:")
importance = ranker.get_feature_importance()
print(importance.head(15).to_string(index=False))

# ===== STEP 7: Test on sample user =====
print("\n" + "="*70)
print("SAMPLE PREDICTION")
print("="*70)

test_user = eval_feat_df['user_id'].iloc[0] if not eval_feat_df.empty else None
if test_user:
    user_data = eval_feat_df[eval_feat_df['user_id'] == test_user].copy()
    predictions = ranker.predict(user_data)
    user_data['lgb_score'] = predictions

    print(f"\nUser {test_user} - Top 10 by LightGBM:")
    top_lgb = user_data.sort_values('lgb_score', ascending=False).head(10)
    print(top_lgb[['item_id', 'sasrec_score', 'lgb_score', 'rank_position', 'target']].to_string(index=False))

    print(f"\nUser {test_user} - Top 10 by SASRec (original):")
    top_sasrec = user_data.sort_values('rank_position').head(10)
    print(top_sasrec[['item_id', 'sasrec_score', 'lgb_score', 'rank_position', 'target']].to_string(index=False))

    print(f"\nRelevant items in LightGBM top-10: {(top_lgb['target'] > 0).sum()}/10")
    print(f"Relevant items in SASRec top-10: {(top_sasrec['target'] > 0).sum()}/10")

# Save model
print("\nSaving model...")
ranker.save_model("lightgbm_ranker_model2.pkl")
print("✓ Model saved to: lightgbm_ranker_model.pkl")

print("\n" + "="*70)
print("✓ LightGBM training complete!")
print("="*70)
