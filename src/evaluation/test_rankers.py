import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import SASRecModel

from lightgbm_ranker import LightGBMRanker, FeatureBuilder, create_training_data
from xgboost_ranker import XGBoostRanker, XGBoostFeatureBuilder, create_xgb_training_data, compare_models

# Enable deterministic behaviour
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def create_sample_data():
    """Create sample interaction and item data for testing based on Retail Rocket structure"""
    
    # Sample items data (Retail Rocket style)
    items_data = {
        Columns.Item: range(1000),
        'categoryid': np.random.choice([1, 2, 3, 4, 5], 1000),
        'parentid': np.random.choice([0, 1, 2], 1000),
        'price': np.random.uniform(10, 500, 1000)
    }
    items_df = pd.DataFrame(items_data)
    items_df = items_df.set_index(Columns.Item)
    
    # Sample interactions data (Retail Rocket style)
    n_users = 500
    n_interactions = 10000
    
    # Create interaction weights similar to Retail Rocket (view=1, addtocart=3, transaction=5)
    event_types = np.random.choice(['view', 'addtocart', 'transaction'], n_interactions, p=[0.7, 0.2, 0.1])
    interaction_weights = {'view': 1.0, 'addtocart': 3.0, 'transaction': 5.0}
    
    interactions_data = {
        Columns.User: np.random.randint(1, n_users + 1, n_interactions),
        Columns.Item: np.random.randint(1, 1000, n_interactions),
        Columns.Datetime: [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_interactions)],
        Columns.Weight: [interaction_weights[event] for event in event_types],
        'event': event_types
    }
    
    interactions_df = pd.DataFrame(interactions_data)
    
    # Remove duplicates and aggregate by user-item (like Retail Rocket processing)
    interactions_df = interactions_df.groupby([Columns.User, Columns.Item]).agg({
        Columns.Weight: 'sum',
        Columns.Datetime: 'max',
        'event': lambda x: ','.join(sorted(set(x)))  # Combine event types
    }).reset_index()
    
    print(f"Created sample data:")
    print(f"Items: {len(items_df)}")
    print(f"Interactions: {len(interactions_df)}")
    print(f"Users: {interactions_df[Columns.User].nunique()}")
    print(f"Event distribution: {interactions_df['event'].value_counts()}")
    
    return interactions_df, items_df


def train_sasrec_model(interactions_df):
    """Train a simple SASRec model for testing"""
    
    # Create dataset
    dataset = Dataset.construct(interactions_df)
    
    # Train SASRec model
    sasrec_params = {
        'embedding_dim': 32,
        'hidden_dim': 64,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_len': 50,
        'dropout': 0.1,
        'batch_size': 128,
        'learning_rate': 0.001,
        'n_epochs': 5,  # Small number for testing
        'random_state': 42
    }
    
    model = SASRecModel(**sasrec_params)
    model.fit(dataset)
    
    print("SASRec model trained successfully")
    return model


def test_lightgbm_ranker(interactions_df, items_df, sasrec_model):
    """Test LightGBM ranker implementation"""
    
    print("\n=== Testing LightGBM Ranker ===")
    
    # Create training data
    train_data, feature_builder = create_training_data(
        interactions_df, items_df, sasrec_model, test_size=0.2
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data columns: {list(train_data.columns)}")
    
    # Split into train and validation
    users = train_data['user_id'].unique()
    np.random.shuffle(users)
    split_idx = int(len(users) * 0.8)
    train_users = users[:split_idx]
    val_users = users[split_idx:]
    
    train_df = train_data[train_data['user_id'].isin(train_users)]
    val_df = train_data[train_data['user_id'].isin(val_users)]
    
    # Initialize and train LightGBM ranker
    lgb_ranker = LightGBMRanker()
    
    # Train model
    feature_importance = lgb_ranker.train(
        train_df, 
        target_col='target',
        validation_data=(val_df.drop('target', axis=1), val_df['target'], val_df['user_id'].values)
    )
    
    print("LightGBM model trained successfully")
    
    # Evaluate on validation set
    metrics = lgb_ranker.evaluate(val_df, target_col='target', k=10)
    print(f"LightGBM NDCG@10: {metrics['ndcg@10']:.4f}")
    
    # Show feature importance
    print("\nTop 10 LightGBM features:")
    feature_names = lgb_ranker.feature_cols
    importance_scores = feature_importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print(feature_importance_df.head(10).to_string(index=False))
    
    return lgb_ranker, feature_builder


def test_xgboost_ranker(interactions_df, items_df, sasrec_model):
    """Test XGBoost ranker implementation"""
    
    print("\n=== Testing XGBoost Ranker ===")
    
    # Create training data
    train_data, feature_builder = create_xgb_training_data(
        interactions_df, items_df, sasrec_model, test_size=0.2
    )
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data columns: {list(train_data.columns)}")
    
    # Split into train and validation
    users = train_data['user_id'].unique()
    np.random.shuffle(users)
    split_idx = int(len(users) * 0.8)
    train_users = users[:split_idx]
    val_users = users[split_idx:]
    
    train_df = train_data[train_data['user_id'].isin(train_users)]
    val_df = train_data[train_data['user_id'].isin(val_users)]
    
    # Initialize and train XGBoost ranker
    xgb_ranker = XGBoostRanker()
    
    # Train model
    evals_result = xgb_ranker.train(
        train_df, 
        target_col='target',
        validation_data=(val_df.drop('target', axis=1), val_df['target'], val_df['user_id'].values),
        early_stopping_rounds=5
    )
    
    print("XGBoost model trained successfully")
    
    # Evaluate on validation set
    metrics = xgb_ranker.evaluate(val_df, target_col='target', k=10)
    print(f"XGBoost NDCG@10: {metrics['ndcg@10']:.4f}")
    
    # Show feature importance
    print("\nTop 10 XGBoost features:")
    feature_importance = xgb_ranker.get_feature_importance()
    print(feature_importance.head(10).to_string(index=False))
    
    return xgb_ranker, feature_builder


def test_production_pipeline(interactions_df, items_df, sasrec_model, lgb_ranker, xgb_ranker):
    """Test the complete production pipeline with Retail Rocket data"""
    
    print("\n=== Testing Production Pipeline ===")
    
    # Create a test dataset
    test_interactions = interactions_df.sample(frac=0.1, random_state=42)
    test_dataset = Dataset.construct(test_interactions)
    
    # Get a sample user
    sample_user = test_interactions[Columns.User].iloc[0]
    print(f"Testing pipeline for user: {sample_user}")
    
    # Step 1: Get SASRec candidates
    candidates, sasrec_scores = sasrec_model.recommend(
        users=np.array([sample_user]),
        dataset=test_dataset,
        k=500,
        filter_viewed=True
    )
    
    print(f"SASRec generated {len(candidates[0])} candidates")
    
    # Step 2: Create candidate DataFrame using Retail Rocket columns
    candidate_df = pd.DataFrame({
        Columns.User: sample_user,
        Columns.Item: candidates[0],
        'sasrec_score': sasrec_scores[0],
        'rank_position': range(1, len(candidates[0]) + 1)
    })
    
    # Step 3: Build features for LightGBM
    lgb_feature_builder = FeatureBuilder(test_interactions, items_df, sasrec_model)
    lgb_features = lgb_feature_builder.build_features(candidate_df)
    
    # Step 4: Get LightGBM predictions
    lgb_scores = lgb_ranker.predict(lgb_features)
    
    # Step 5: Build features for XGBoost
    xgb_feature_builder = XGBoostFeatureBuilder(test_interactions, items_df, sasrec_model)
    xgb_features = xgb_feature_builder.build_features(candidate_df)
    
    # Step 6: Get XGBoost predictions
    xgb_scores = xgb_ranker.predict(xgb_features)
    
    # Step 7: Combine and rank
    final_results = pd.DataFrame({
        Columns.Item: candidates[0],
        'sasrec_score': sasrec_scores[0],
        'sasrec_rank': range(1, len(candidates[0]) + 1),
        'lgb_score': lgb_scores,
        'lgb_rank': pd.Series(lgb_scores).rank(ascending=False, method='first'),
        'xgb_score': xgb_scores,
        'xgb_rank': pd.Series(xgb_scores).rank(ascending=False, method='first')
    })
    
    # Get top-10 recommendations from each model
    top_10_sasrec = final_results.nsmallest(10, 'sasrec_rank')[Columns.Item].tolist()
    top_10_lgb = final_results.nsmallest(10, 'lgb_rank')[Columns.Item].tolist()
    top_10_xgb = final_results.nsmallest(10, 'xgb_rank')[Columns.Item].tolist()
    
    print(f"\nTop-10 Recommendations:")
    print(f"SASRec: {top_10_sasrec}")
    print(f"LightGBM: {top_10_lgb}")
    print(f"XGBoost: {top_10_xgb}")
    
    # Calculate overlap
    lgb_xgb_overlap = len(set(top_10_lgb) & set(top_10_xgb))
    sasrec_lgb_overlap = len(set(top_10_sasrec) & set(top_10_lgb))
    sasrec_xgb_overlap = len(set(top_10_sasrec) & set(top_10_xgb))
    
    print(f"\nOverlap in top-10:")
    print(f"LightGBM-XGBoost: {lgb_xgb_overlap}/10")
    print(f"SASRec-LightGBM: {sasrec_lgb_overlap}/10")
    print(f"SASRec-XGBoost: {sasrec_xgb_overlap}/10")
    
    # Show some feature examples
    print(f"\nSample features for top LightGBM recommendation:")
    top_lgb_item = top_10_lgb[0]
    item_features = lgb_features[lgb_features[Columns.Item] == top_lgb_item].iloc[0]
    print(f"Item {top_lgb_item} features:")
    print(f"  SASRec score: {item_features['sasrec_score']:.4f}")
    print(f"  Item popularity: {item_features.get('item_popularity', 'N/A')}")
    print(f"  User interactions: {item_features.get('user_total_interactions', 'N/A')}")
    print(f"  Activity level: {item_features.get('activity_level', 'N/A')}")
    
    return final_results


def main():
    """Main test function"""
    
    print("Starting Production Architecture Test")
    print("=" * 50)
    
    # Create sample data
    interactions_df, items_df = create_sample_data()
    
    # Train SASRec model
    sasrec_model = train_sasrec_model(interactions_df)
    
    # Test LightGBM ranker
    lgb_ranker, lgb_feature_builder = test_lightgbm_ranker(interactions_df, items_df, sasrec_model)
    
    # Test XGBoost ranker
    xgb_ranker, xgb_feature_builder = test_xgboost_ranker(interactions_df, items_df, sasrec_model)
    
    # Test production pipeline
    final_results = test_production_pipeline(interactions_df, items_df, sasrec_model, lgb_ranker, xgb_ranker)
    
    # Compare models
    print("\n=== Model Comparison ===")
    
    # Create some test data for comparison
    test_data, _ = create_training_data(interactions_df, items_df, sasrec_model, test_size=0.1)
    
    comparison = compare_models(lgb_ranker, xgb_ranker, test_data)
    
    print(f"LightGBM NDCG@10: {comparison['lightgbm']['ndcg@10']:.4f}")
    print(f"XGBoost NDCG@10: {comparison['xgboost']['ndcg@10']:.4f}")
    
    print("\nTop XGBoost Features:")
    for feat in comparison['xgb_feature_importance'][:5]:
        print(f"  {feat['feature']}: {feat['importance']:.4f}")
    
    print("\nTest completed successfully!")
    print("\nProduction Architecture Summary:")
    print("1. ✅ SASRec model trained and generating candidates")
    print("2. ✅ LightGBM ranker implemented and tested")
    print("3. ✅ XGBoost ranker implemented and tested")
    print("4. ✅ Feature engineering pipeline working")
    print("5. ✅ Complete production pipeline functional")
    
    return {
        'sasrec_model': sasrec_model,
        'lgb_ranker': lgb_ranker,
        'xgb_ranker': xgb_ranker,
        'results': final_results
    }


if __name__ == "__main__":
    results = main()
