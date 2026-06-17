import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from scipy.sparse.linalg import svds
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

RANDOM_STATE = 42

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading Dataset...")
ratings = pd.read_csv(
    'data/ratings_Electronics.csv', 
    names=['userId', 'productId', 'rating', 'timestamp'],
    on_bad_lines='skip'
)

print(f"Initial dataset shape: {ratings.shape}")
print(f"Columns: {ratings.columns.tolist()}")

ratings_original = ratings.copy(deep=True)

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n[2/7] Preprocessing Data...")

# Convert timestamp
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
print(f"Date range: {ratings.timestamp.min()} to {ratings.timestamp.max()}")

# Check for duplicates and nulls
duplicates = ratings.duplicated(subset=['userId', 'productId'], keep=False).sum()
print(f"Duplicate user-product pairs: {duplicates}")
print(f"Missing values:\n{ratings.isnull().sum()}")

print(f"\nRating distribution:")
print(ratings.rating.value_counts().sort_index())

print(f"\nUnique users: {ratings.userId.nunique()}")
print(f"Unique products: {ratings.productId.nunique()}")

# ============================================================================
# 3. REDUCE SPARSITY - OPTIMIZED
# ============================================================================
print("\n[3/7] Reducing Sparsity (Optimized)...")

original_users = ratings.userId.nunique()
original_items = ratings.productId.nunique()
original_density = (len(ratings) / (original_users * original_items)) * 100

print(f"\nOriginal Matrix:")
print(f"  Users: {original_users:,}")
print(f"  Products: {original_items:,}")
print(f"  Interactions: {len(ratings):,}")
print(f"  Density: {original_density:.5f}%")

# OPTIMIZATION 1: Filter users with at least 50 ratings
user_counts = ratings.groupby('userId').size()
eligible_users = user_counts[user_counts >= 50].index
ratings = ratings[ratings.userId.isin(eligible_users)]

# OPTIMIZATION 2: Filter products with at least 10 ratings
product_counts = ratings.groupby('productId').size()
popular_products = product_counts[product_counts >= 10].index
ratings = ratings[ratings.productId.isin(popular_products)]

print(f"\nFiltering Strategy:")
print(f"  - Keep users with ≥50 ratings")
print(f"  - Keep products with ≥10 ratings")

filtered_users = ratings.userId.nunique()
filtered_items = ratings.productId.nunique()
filtered_density = (len(ratings) / (filtered_users * filtered_items)) * 100

print(f"\nFiltered Matrix:")
print(f"  Users: {filtered_users:,}")
print(f"  Products: {filtered_items:,}")
print(f"  Interactions: {len(ratings):,}")
print(f"  Density: {filtered_density:.5f}%")
print(f"  Density increase: {(filtered_density/original_density - 1)*100:.1f}%")

# ============================================================================
# 4. TRAIN/TEST SPLIT - PER-USER TEMPORAL
# ============================================================================
print("\n[4/7] Creating Train/Test Split (Per-User Temporal)...")

# Sort by user and timestamp for temporal split
ratings = ratings.sort_values(['userId', 'timestamp'])

train_list = []
test_list = []

for user_id, user_data in ratings.groupby('userId'):
    n_ratings = len(user_data)
    split_idx = int(n_ratings * 0.7)
    
    if split_idx == 0:
        split_idx = 1
    if split_idx == n_ratings:
        split_idx = n_ratings - 1
    
    train_list.append(user_data.iloc[:split_idx])
    test_list.append(user_data.iloc[split_idx:])

train_ratings = pd.concat(train_list, ignore_index=True)
test_ratings = pd.concat(test_list, ignore_index=True)

print(f"Train set: {len(train_ratings):,} ({len(train_ratings)/len(ratings)*100:.1f}%)")
print(f"Test set: {len(test_ratings):,} ({len(test_ratings)/len(ratings)*100:.1f}%)")

# Create user-item matrix for training
print("\nCreating user-item matrix...")
train_matrix = train_ratings.pivot(
    index='userId', 
    columns='productId', 
    values='rating'
).fillna(0)

print(f"Train matrix shape: {train_matrix.shape}")
print(f"Train matrix density: {(train_matrix > 0).sum().sum() / train_matrix.size * 100:.2f}%")

# ============================================================================
# 5. TRAIN SVD MODEL WITH OPTIMIZATION
# ============================================================================
print("\n[5/7] Training SVD Model...")

# OPTIMIZATION 3: Center the ratings (mean normalization)
user_means = train_matrix.mean(axis=1)
train_matrix_centered = train_matrix.sub(user_means, axis=0)

# Try different numbers of latent factors
k_values = [50, 100, 150]
results = []

for k in k_values:
    print(f"\nTraining with k={k} latent factors...")
    
    # Perform SVD on centered matrix
    U, sigma, Vt = svds(train_matrix_centered.values, k=k)
    sigma = np.diag(sigma)
    
    # Predict ratings (add back user means)
    predictions_centered = np.dot(np.dot(U, sigma), Vt)
    predictions = predictions_centered + user_means.values.reshape(-1, 1)
    
    # OPTIMIZATION 4: Clip predictions to valid rating range [1, 5]
    predictions = np.clip(predictions, 1, 5)
    
    pred_df = pd.DataFrame(
        predictions, 
        index=train_matrix.index, 
        columns=train_matrix.columns
    )
    
    # Evaluate on test set
    test_predictions = []
    test_actuals = []
    
    for _, row in test_ratings.iterrows():
        user = row['userId']
        product = row['productId']
        actual = row['rating']
        
        if user in pred_df.index and product in pred_df.columns:
            pred = pred_df.loc[user, product]
            test_predictions.append(pred)
            test_actuals.append(actual)
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_actuals, test_predictions))
    
    results.append({
        'k': k,
        'rmse': rmse,
        'predictions': pred_df,
        'test_coverage': len(test_predictions) / len(test_ratings) * 100
    })
    
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Test coverage: {len(test_predictions)/len(test_ratings)*100:.1f}%")

# Select best model
best_model = min(results, key=lambda x: x['rmse'])
best_k = best_model['k']
best_rmse = best_model['rmse']
best_predictions = best_model['predictions']

print(f"\n{'='*60}")
print(f"BEST MODEL: k={best_k}, RMSE={best_rmse:.4f}")
print(f"{'='*60}")

# ============================================================================
# 6. GENERATE RECOMMENDATIONS
# ============================================================================
print("\n[6/7] Generating Recommendations...")

def recommend_items(user_id, orig_matrix, pred_matrix, top_n=10):
    """Generate top-N recommendations for a user"""
    
    if user_id not in pred_matrix.index:
        return None
    
    user_predictions = pred_matrix.loc[user_id]
    user_actuals = orig_matrix.loc[user_id]
    
    # Filter out already rated items
    recommendations = user_predictions[user_actuals == 0]
    
    # Sort by predicted rating
    recommendations = recommendations.sort_values(ascending=False)
    
    return recommendations.head(top_n)

# Sample recommendations
sample_users = train_matrix.index[:5].tolist()

print(f"\nTop-10 Recommendations for Sample Users:")
print("="*60)

for user in sample_users:
    print(f"\nUser: {user}")
    recs = recommend_items(user, train_matrix, best_predictions, top_n=10)
    
    if recs is not None:
        for i, (product, score) in enumerate(recs.items(), 1):
            print(f"  {i}. {product}: {score:.3f}")

# ============================================================================
# 7. COMPREHENSIVE EVALUATION METRICS
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION METRICS")
print("="*80)

print("\nGenerating recommendations for all test users...")
K = 10

# Get test users in training data
test_users_in_train = test_ratings[
    test_ratings.userId.isin(train_matrix.index)
].userId.unique()

print(f"Test users in training data: {len(test_users_in_train)}")

# Generate top-K recommendations for each test user
all_recommendations = {}
for user in test_users_in_train:
    recs = recommend_items(user, train_matrix, best_predictions, top_n=K)
    if recs is not None:
        all_recommendations[user] = list(recs.index)  # Keep as list for order

# Create ground truth
ground_truth = {}
for user in test_users_in_train:
    user_test_items = test_ratings[test_ratings.userId == user].productId.values
    ground_truth[user] = set(user_test_items)

# METRIC 1: Precision@K
precisions = []
for user in all_recommendations:
    if user in ground_truth:
        recommended = set(all_recommendations[user])
        relevant = ground_truth[user]
        
        if len(recommended) > 0:
            hits = len(recommended & relevant)
            precision = hits / len(recommended)
            precisions.append(precision)

precision_at_k = np.mean(precisions) if precisions else 0

# METRIC 2: Recall@K
recalls = []
for user in all_recommendations:
    if user in ground_truth:
        recommended = set(all_recommendations[user])
        relevant = ground_truth[user]
        
        if len(relevant) > 0:
            hits = len(recommended & relevant)
            recall = hits / len(relevant)
            recalls.append(recall)

recall_at_k = np.mean(recalls) if recalls else 0

# METRIC 3: F1-Score@K
f1_at_k = 0
if precision_at_k + recall_at_k > 0:
    f1_at_k = 2 * (precision_at_k * recall_at_k) / (precision_at_k + recall_at_k)

# METRIC 4: MAP@K (Mean Average Precision)
def average_precision_at_k(recommended_list, relevant_set, k):
    """Calculate Average Precision at K"""
    if len(relevant_set) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0.0
    
    for i, item in enumerate(recommended_list[:k]):
        if item in relevant_set:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    
    return score / min(len(relevant_set), k)

map_scores = []
for user in all_recommendations:
    if user in ground_truth:
        recommended_list = all_recommendations[user]
        relevant = ground_truth[user]
        ap = average_precision_at_k(recommended_list, relevant, K)
        map_scores.append(ap)

map_at_k = np.mean(map_scores) if map_scores else 0

# METRIC 5: NDCG@K (Normalized Discounted Cumulative Gain)
def ndcg_at_k(recommended_list, relevant_set, k):
    """Calculate NDCG at K"""
    if len(relevant_set) == 0:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(recommended_list[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)
    
    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_set), k)))
    
    return dcg / idcg if idcg > 0 else 0.0

ndcg_scores = []
for user in all_recommendations:
    if user in ground_truth:
        recommended_list = all_recommendations[user]
        relevant = ground_truth[user]
        ndcg = ndcg_at_k(recommended_list, relevant, K)
        ndcg_scores.append(ndcg)

ndcg_at_k_score = np.mean(ndcg_scores) if ndcg_scores else 0

# METRIC 6: Hit Rate@K
hits = []
for user in all_recommendations:
    if user in ground_truth:
        recommended = set(all_recommendations[user])
        relevant = ground_truth[user]
        hit = 1 if len(recommended & relevant) > 0 else 0
        hits.append(hit)

hit_rate = np.mean(hits) if hits else 0

# METRIC 7: Coverage Analysis
total_items = train_matrix.shape[1]
all_recommended_items = set()
for recs in all_recommendations.values():
    all_recommended_items.update(recs)

catalog_coverage = len(all_recommended_items) / total_items * 100

# METRIC 8: Diversity (Average dissimilarity)
avg_unique_per_user = np.mean([len(set(recs)) for recs in all_recommendations.values()])

# Print Results
print("\n" + "="*80)
print(f"RANKING METRICS @K={K}")
print("="*80)

print(f"\nAccuracy Metrics:")
print(f"  Precision@{K}:     {precision_at_k:.4f}")
print(f"  Recall@{K}:        {recall_at_k:.4f}")
print(f"  F1-Score@{K}:      {f1_at_k:.4f}")
print(f"  Hit Rate@{K}:      {hit_rate:.4f}")

print(f"\nRanking Quality:")
print(f"  MAP@{K}:           {map_at_k:.4f}")
print(f"  NDCG@{K}:          {ndcg_at_k_score:.4f}")

print(f"\nPrediction Error:")
print(f"  RMSE:              {best_rmse:.4f}")

print(f"\nCoverage & Diversity:")
print(f"  Catalog Coverage:  {catalog_coverage:.2f}% ({len(all_recommended_items)}/{total_items} items)")
print(f"  Avg Unique/User:   {avg_unique_per_user:.1f}")

print(f"\nCold-Start Performance:")
cold_start_users = len(test_ratings.userId.unique()) - len(test_users_in_train)
print(f"  Total test users:  {len(test_ratings.userId.unique())}")
print(f"  Covered users:     {len(test_users_in_train)} ({len(test_users_in_train)/len(test_ratings.userId.unique())*100:.1f}%)")
print(f"  Cold-start users:  {cold_start_users}")

# Model comparison across k values
print(f"\n" + "="*80)
print("MODEL COMPARISON (Different Latent Factors)")
print("="*80)
print(f"\n{'k':>5} | {'RMSE':>8} | {'Prec@10':>8} | {'Rec@10':>8} | {'MAP@10':>8} | {'NDCG@10':>8}")
print("-" * 60)

for result in results:
    k_val = result['k']
    pred_df = result['predictions']
    
    # Calculate metrics for this k
    k_recs = {}
    for user in test_users_in_train:
        recs = recommend_items(user, train_matrix, pred_df, top_n=K)
        if recs is not None:
            k_recs[user] = list(recs.index)
    
    k_precisions = []
    k_recalls = []
    k_map_scores = []
    k_ndcg_scores = []
    
    for user in k_recs:
        if user in ground_truth:
            recommended_set = set(k_recs[user])
            recommended_list = k_recs[user]
            relevant = ground_truth[user]
            
            # Precision
            if len(recommended_set) > 0:
                hits = len(recommended_set & relevant)
                k_precisions.append(hits / len(recommended_set))
            
            # Recall
            if len(relevant) > 0:
                hits = len(recommended_set & relevant)
                k_recalls.append(hits / len(relevant))
            
            # MAP
            k_map_scores.append(average_precision_at_k(recommended_list, relevant, K))
            
            # NDCG
            k_ndcg_scores.append(ndcg_at_k(recommended_list, relevant, K))
    
    k_prec = np.mean(k_precisions) if k_precisions else 0
    k_rec = np.mean(k_recalls) if k_recalls else 0
    k_map = np.mean(k_map_scores) if k_map_scores else 0
    k_ndcg = np.mean(k_ndcg_scores) if k_ndcg_scores else 0
    
    print(f"{k_val:>5} | {result['rmse']:>8.4f} | {k_prec:>8.4f} | {k_rec:>8.4f} | {k_map:>8.4f} | {k_ndcg:>8.4f}")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
