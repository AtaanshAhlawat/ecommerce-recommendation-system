"""
Part 2 Evaluation: Measure LightGBM Ranker Performance
Compares SASRec baseline vs LightGBM re-ranking on held-out Part 2 users
"""
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from rectools import Columns
from lightgbm_ranker import LightGBMFeatureBuilder, LightGBMRanker
from generate_candidate import generate_candidates_for_users, load_data_and_model

print("="*70)
print("PART 2 EVALUATION: LIGHTGBM RANKER VS SASREC BASELINE")
print("="*70)

# ===== STEP 1: Load Trained Ranker =====
print("\nLoading trained LightGBM ranker...")
ranker = LightGBMRanker()
ranker.load_model("lightgbm_ranker_causal.pkl")
print("✓ Ranker loaded")

# ===== STEP 2: Load Part 2 Ground Truth =====
print("\nLoading Part 2 ground truth...")
part2_gt = pd.read_csv("ranker_test_gt_part2.csv")
part2_gt[Columns.Datetime] = pd.to_datetime(part2_gt[Columns.Datetime])
part2_users = list(part2_gt[Columns.User].unique())
print(f"Part 2 users: {len(part2_users):,}")
print(f"Part 2 interactions: {len(part2_gt):,}")

# ===== STEP 3: Generate Candidates for Part 2 =====
print("\nGenerating SASRec candidates for Part 2 users...")
sasrec_model, dataset, train_interactions, future_interactions = load_data_and_model()

part2_candidates = generate_candidates_for_users(
    sasrec_model, dataset, train_interactions, part2_gt,
    target_users=part2_users, k=500
)

print(f"Generated {len(part2_candidates):,} candidates for {part2_candidates[Columns.User].nunique()} users")

# ===== STEP 4: Build Features for Part 2 (SAME Context as Training) =====
print("\nBuilding features with SAME context as training (History only - NO Part 1)...")
# CRITICAL FIX: Use ONLY history, just like training
# DO NOT add Part 1 data - it creates feature distribution mismatch

# Get items
try:
    items_sparse = dataset.item_features
    items_df = pd.DataFrame()  # Simplified
except:
    items_df = pd.DataFrame()

# Feature builder with SAME context as training
part2_end = train_interactions[Columns.Datetime].max() + pd.Timedelta(days=30)  # End of Part 2
fb_eval = LightGBMFeatureBuilder(train_interactions, items_df, None, current_time=part2_end)

# Build features per user
print("Extracting features...")
eval_features = []
unique_users = part2_candidates[Columns.User].unique()

for user in unique_users:
    user_cands = part2_candidates[part2_candidates[Columns.User] == user].copy()
    features = fb_eval.build_features_vectorized(
        user_cands,
        current_time=part2_end,
        max_score=user_cands['sasrec_score'].max(),
        min_score=user_cands['sasrec_score'].min()
    )
    eval_features.append(features)

eval_feat_df = pd.concat(eval_features, ignore_index=True)
print(f"Feature matrix shape: {eval_feat_df.shape}")

# ===== STEP 5: Re-rank with LightGBM =====
print("\nRe-ranking candidates with LightGBM...")
lgb_scores = ranker.predict(eval_feat_df)
eval_feat_df['lgb_score'] = lgb_scores

# ===== STEP 6: Evaluate HR@10 =====
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

# Ground truth set
gt_pairs = set(zip(part2_gt[Columns.User], part2_gt[Columns.Item]))

def calculate_hitrate(df, score_col, k=10):
    hits = 0
    total_users = 0
    
    for user in df[Columns.User].unique():
        user_data = df[df[Columns.User] == user].copy()
        top_k = user_data.nlargest(k, score_col)
        
        user_hits = sum(1 for _, row in top_k.iterrows() 
                       if (row[Columns.User], row[Columns.Item]) in gt_pairs)
        
        if user_hits > 0:
            hits += 1
        total_users += 1
    
    return hits / total_users if total_users > 0 else 0

# Calculate metrics
hr10_sasrec = calculate_hitrate(eval_feat_df, 'sasrec_score', k=10)
hr10_lgb = calculate_hitrate(eval_feat_df, 'lgb_score', k=10)

print(f"\n📊 HitRate@10 Results:")
print(f"  SASRec Baseline:  {hr10_sasrec:.4f} ({hr10_sasrec*100:.2f}%)")
print(f"  LightGBM Ranker:  {hr10_lgb:.4f} ({hr10_lgb*100:.2f}%)")
print(f"  Improvement:      {(hr10_lgb - hr10_sasrec):.4f} ({((hr10_lgb/hr10_sasrec - 1)*100):.1f}%)")

# ===== STEP 7: Sample Comparison =====
print("\n" + "="*70)
print("SAMPLE USER COMPARISON")
print("="*70)

sample_user = eval_feat_df[Columns.User].iloc[0]
user_data = eval_feat_df[eval_feat_df[Columns.User] == sample_user].copy()

print(f"\nUser {sample_user} - Top 10 by LightGBM:")
top_lgb = user_data.nlargest(10, 'lgb_score')
for idx, (_, row) in enumerate(top_lgb.iterrows(), 1):
    hit = "✓" if (row[Columns.User], row[Columns.Item]) in gt_pairs else " "
    print(f"  {idx}. Item {row[Columns.Item]} - LGB: {row['lgb_score']:.3f}, SASRec: {row['sasrec_score']:.3f} {hit}")

print(f"\nUser {sample_user} - Top 10 by SASRec:")
top_sasrec = user_data.nlargest(10, 'sasrec_score')
for idx, (_, row) in enumerate(top_sasrec.iterrows(), 1):
    hit = "✓" if (row[Columns.User], row[Columns.Item]) in gt_pairs else " "
    print(f"  {idx}. Item {row[Columns.Item]} - SASRec: {row['sasrec_score']:.3f}, LGB: {row['lgb_score']:.3f} {hit}")

print("\n" + "="*70)
print("✓ EVALUATION COMPLETE")
print("="*70)