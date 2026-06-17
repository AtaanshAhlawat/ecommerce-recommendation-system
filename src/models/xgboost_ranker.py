import os
import typing as tp
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import LabelEncoder
from rectools import Columns
from typing import Dict, Any
from rectools.dataset import Dataset
from rectools.models import SASRecModel

# Suppress pandas FutureWarnings
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')


class XGBoostFeatureBuilder:
    """Build features for XGBoost ranking model from SASRec candidates and Retail Rocket interaction data"""
    
    def __init__(self, interactions: pd.DataFrame, items: pd.DataFrame, sasrec_model: SASRecModel):
        self.interactions = interactions
        self.items = items
        self.sasrec_model = sasrec_model
        self.user_features = {}
        self.item_features = {}
        self._precompute_features()
    
    def _precompute_features(self):
        """Precompute user and item level features for Retail Rocket data"""
        # User features - using visitorid as user_id
        user_stats = self.interactions.groupby(Columns.User).agg({
            Columns.Item: 'count',
            Columns.Weight: ['sum', 'mean'],
            Columns.Datetime: ['min', 'max']
        }).reset_index()
        user_stats.columns = [Columns.User, 'user_total_interactions', 'user_total_weight', 
                            'user_avg_weight', 'user_first_interaction', 'user_last_interaction']
        
        # Add user activity level similar to retail_rocket_6.py
        user_stats['user_activity_span_days'] = (
            user_stats['user_last_interaction'] - user_stats['user_first_interaction']
        ).dt.days.fillna(0)
        user_stats['user_interactions_per_day'] = np.where(
            user_stats['user_activity_span_days'] > 0,
            user_stats['user_total_interactions'] / user_stats['user_activity_span_days'],
            user_stats['user_total_interactions']
        )
        
        # Add activity level categorization
        user_stats['activity_level'] = pd.cut(user_stats['user_total_interactions'], 
                                     bins=[0, 5, 20, float('inf')], 
                                     labels=['Low', 'Medium', 'High'])
        
        # Add interaction category similar to retail_rocket_6.py
        user_stats['interaction_category'] = pd.cut(user_stats['user_total_interactions'],
                                          bins=[0, 10, 50, float('inf')],
                                          labels=['Casual', 'Regular', 'Frequent'])
        
        self.user_features = user_stats.set_index(Columns.User).to_dict('index')
        
        # Item features - using itemid as item_id
        item_stats = self.interactions.groupby(Columns.Item).agg({
            Columns.User: 'count',
            Columns.Weight: ['sum', 'mean'],
            Columns.Datetime: ['min', 'max']
        }).reset_index()
        item_stats.columns = [Columns.Item, 'item_popularity', 'item_total_weight',
                            'item_avg_weight', 'item_first_interaction', 'item_last_interaction']
        
        # Add popularity level similar to retail_rocket_6.py
        item_stats['popularity_level'] = pd.cut(item_stats['item_popularity'],
                                            bins=[0, 10, 50, float('inf')],
                                            labels=['Unpopular', 'Popular', 'Very_Popular'])
        
        # Create popularity buckets and percentiles
        item_stats['popularity_bucket'] = pd.qcut(item_stats['item_popularity'], 
                                                 q=10, labels=False, duplicates='drop')
        item_stats['popularity_percentile'] = item_stats['item_popularity'].rank(pct=True)
        
        # Add item age features
        current_time = datetime.now()
        item_stats['item_age_days'] = (
            current_time - item_stats['item_first_interaction']
        ).dt.days.fillna(0)
        
        # Merge with item metadata if available
        if self.items is not None and len(self.items) > 0:
            item_stats = item_stats.merge(self.items, left_on=Columns.Item, right_index=True, how='left')
        
        # Ensure unique item IDs before setting index
        item_stats = item_stats.drop_duplicates(subset=[Columns.Item], keep='first')
        self.item_features = item_stats.set_index(Columns.Item).to_dict('index')
    
    def get_user_features(self, user_id: int) -> dict:
        """Get precomputed user features"""
        return self.user_features.get(user_id, {})
    
    def get_item_features(self, item_id: int) -> dict:
        """Get precomputed item features"""
        return self.item_features.get(item_id, {})
    
    def build_features_vectorized(self, candidates_df: pd.DataFrame, current_time: datetime = None,
                              max_score: float = None, min_score: float = None) -> pd.DataFrame:
        """
        Build features for XGBoost ranking model - FULLY VECTORIZED (no Python loops)
        
        Args:
            candidates_df: DataFrame with columns [user_id, item_id, sasrec_score, rank_position]
            current_time: Reference time for recency features
            max_score: Pre-computed max SASRec score for this user batch
            min_score: Pre-computed min SASRec score for this user batch
            
        Returns:
            DataFrame with all features for ranking
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Use provided normalization stats or compute per batch
        if max_score is None or min_score is None:
            max_score = candidates_df['sasrec_score'].max()
            min_score = candidates_df['sasrec_score'].min()
        
        max_rank = candidates_df['rank_position'].max()
        
        # FIXED: Fully vectorized feature computation - ZERO Python loops!
        features_df = candidates_df.copy()
        
        # SASRec features with CORRECT per-user normalization (vectorized)
        features_df['sasrec_score_norm'] = (features_df['sasrec_score'] - min_score) / (max_score - min_score + 1e-8)
        # Keep original negative scores (they have meaning!)
        features_df['sasrec_score_raw'] = features_df['sasrec_score']
        # Apply log only to shifted positive version if needed
        features_df['sasrec_score_shifted'] = features_df['sasrec_score'] - min_score + 1
        features_df['sasrec_score_log'] = np.log1p(features_df['sasrec_score_shifted'])
        features_df['rank_position_norm'] = features_df['rank_position'] / (max_rank + 1)
        features_df['rank_position_log'] = np.log1p(features_df['rank_position'])
        features_df['inverse_rank'] = 1.0 / (features_df['rank_position'] + 1)
        
        # FIXED: Vectorized user features lookup using merge (no loops) - CORRECT COLUMN NAMING
        user_feat_df = pd.DataFrame.from_dict(self.user_features, orient='index')
        user_feat_df.index.name = Columns.User
        user_feat_df = user_feat_df.reset_index()
        # Add prefix to non-ID columns only
        user_feat_df = user_feat_df.rename(columns={
            col: f'user_{col}' for col in user_feat_df.columns if col != Columns.User
        })
        
        # FIXED: Vectorized item features lookup using merge (no loops) - CORRECT COLUMN NAMING
        item_feat_df = pd.DataFrame.from_dict(self.item_features, orient='index')
        item_feat_df.index.name = Columns.Item
        item_feat_df = item_feat_df.reset_index()
        # Add prefix to non-ID columns only
        item_feat_df = item_feat_df.rename(columns={
            col: f'item_{col}' for col in item_feat_df.columns if col != Columns.Item
        })
        
        # Merge features
        features_df = features_df.merge(user_feat_df, on=Columns.User, how='left')
        features_df = features_df.merge(item_feat_df, on=Columns.Item, how='left')
        
        # FIXED: Fill NaN AFTER all merges (with proper dtype handling)
        features_df = features_df.fillna(0).infer_objects(copy=False)
        
        # FIXED: Convert all object and datetime columns to numeric for XGBoost
        for col in features_df.columns:
            if col in ['user_id', 'item_id', 'target']:
                continue  # Skip ID and target columns
            
            if features_df[col].dtype == 'object':
                # Convert categorical objects to numeric codes
                if features_df[col].nunique() < 100:  # Only if reasonable cardinality
                    features_df[col] = pd.Categorical(features_df[col]).codes
                else:
                    # For high cardinality, drop the column
                    features_df = features_df.drop(columns=[col])
            elif features_df[col].dtype == 'datetime64[ns]':
                # Convert datetime to timestamp (seconds since epoch)
                features_df[col] = features_df[col].astype('int64') // 10**9
            elif pd.api.types.is_numeric_dtype(features_df[col]):
                # Ensure numeric columns are float
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        # FIXED: Time features with correct column references
        if 'user_last_interaction' in features_df.columns:
            user_last_interaction = pd.to_datetime(features_df['user_last_interaction'])
            time_since_user = (current_time - user_last_interaction).dt.total_seconds() / 3600
            features_df['time_since_user_interaction_hrs'] = time_since_user.clip(upper=1e6)  # Cap instead of inf
            features_df['time_since_user_interaction_days'] = features_df['time_since_user_interaction_hrs'] / 24
        else:
            features_df['time_since_user_interaction_hrs'] = 1e6
            features_df['time_since_user_interaction_days'] = 1e6 / 24
        
        if 'item_last_interaction' in features_df.columns:
            item_last_interaction = pd.to_datetime(features_df['item_last_interaction'])
            time_since_item = (current_time - item_last_interaction).dt.total_seconds() / 3600
            features_df['time_since_item_interaction_hrs'] = time_since_item.clip(upper=1e6)  # Cap instead of inf
            features_df['time_since_item_interaction_days'] = features_df['time_since_item_interaction_hrs'] / 24
        else:
            features_df['time_since_item_interaction_hrs'] = 1e6
            features_df['time_since_item_interaction_days'] = 1e6 / 24
        
        # FIXED: Interaction features with correct column references
        features_df['sasrec_user_interactions'] = (
            features_df['sasrec_score'] * np.log1p(np.maximum(features_df.get('user_total_interactions', 0), 0))
        )
        features_df['sasrec_item_popularity'] = (
            features_df['sasrec_score'] * np.log1p(np.maximum(features_df.get('item_popularity', 0), 0))
        )
        features_df['user_weight_to_popularity_ratio'] = (
            features_df.get('user_total_weight', 1) / (features_df.get('item_popularity', 1) + 1e-8)
        )
        features_df['item_weight_ratio'] = (
            features_df.get('item_avg_weight', 0) / (features_df.get('user_avg_weight', 1) + 1e-8)
        )
        
        return features_df


class XGBoostRanker:
    """XGBoost ranking model for recommendation re-ranking"""
    
    def __init__(self, params: dict = None):
        self.params = params or {
            'objective': 'rank:ndcg',  # FIXED: Use LambdaRank-style objective
            'eval_metric': 'ndcg@10',
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'n_estimators': 100,
            'verbosity': 0
        }
        self.model = None
        self.feature_cols = None
        self.categorical_encoders = {}
    
    def prepare_data(self, features_df: pd.DataFrame, target_col: str = 'target') -> tuple:
        """Prepare data for XGBoost training - REMOVED dangerous categorical encoding + FIXED validation"""
        # Sort by user to maintain group order for XGBoost
        features_df = features_df.sort_values(Columns.User).reset_index(drop=True)
        
        # FIXED: Remove all categorical features to avoid dangerous LabelEncoder mutation
        # Focus on numerical features that matter most for ranking
        categorical_cols = ['category', 'parent_category', 'activity_level', 'interaction_category', 'popularity_level']
        df_clean = features_df.drop(columns=[col for col in categorical_cols if col in features_df.columns])
        
        # Select feature columns (only numerical)
        exclude_cols = [Columns.User, Columns.Item, target_col]
        self.feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[self.feature_cols]
        y = features_df[target_col] if target_col in features_df.columns else None
        groups = features_df[Columns.User].values if Columns.User in features_df.columns else None
        
        # FIXED: Replace inf before validation (time features use large values)
        X = X.replace([np.inf, -np.inf], 1e6)
        
        if X.isnull().any().any():
            raise ValueError(f"NaN in features: {X.columns[X.isnull().any()].tolist()}")
        
        return X, y, groups
    
    def train(self, features_df: pd.DataFrame, target_col: str = 'target', 
              validation_data: tuple = None, early_stopping_rounds: int = 10) -> dict:
        """Train the XGBoost ranking model"""
        X, y, groups = self.prepare_data(features_df, target_col)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set group information for ranking
        if groups is not None:
            group_sizes = self._get_group_sizes(groups)
            dtrain.set_group(group_sizes)
        
        evals_result = {}
        
        if validation_data:
            val_X, val_y, val_groups = validation_data
            dval = xgb.DMatrix(val_X, label=val_y)
            
            if val_groups is not None:
                val_group_sizes = self._get_group_sizes(val_groups)
                dval.set_group(val_group_sizes)
            
            evals = [(dtrain, 'train'), (dval, 'valid')]
        else:
            evals = [(dtrain, 'train')]
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get('n_estimators', 1000),  # Default to 1000 if not specified
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if validation_data else None,
            verbose_eval=False
        )
        
        return evals_result
    
    def _get_group_sizes(self, groups: np.ndarray) -> list:
        """Convert group array to group sizes for XGBoost - FIXED empty groups"""
        if groups is None or len(groups) == 0:
            return []  # FIXED: Return empty list, not [0]
        
        sizes = []
        prev = groups[0]
        count = 1
        
        for g in groups[1:]:
            if g == prev:
                count += 1
            else:
                sizes.append(count)
                count = 1
                prev = g
        
        sizes.append(count)  # Add last group
        return sizes
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _, _ = self.prepare_data(features_df)
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)
    
    def evaluate(self, features_df: pd.DataFrame, target_col: str = 'target', k: int = 10) -> dict:
        """Evaluate model using NDCG@k - FIXED single-item user handling"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, y, groups = self.prepare_data(features_df, target_col)
        predictions = self.predict(features_df)
        
        # Calculate NDCG@k per user
        ndcg_scores = []
        
        if groups is not None:
            start_idx = 0
            for i, group_size in enumerate(self._get_group_sizes(groups)):
                if group_size == 0:
                    continue
                    
                end_idx = start_idx + group_size
                user_preds = predictions[start_idx:end_idx]
                user_targets = y[start_idx:end_idx]
                
                # FIXED: Include single-item users if they have positive targets
                if len(user_targets) >= 1 and user_targets.sum() > 0:
                    # Convert to numpy arrays and reshape for sklearn
                    if hasattr(user_preds, 'values'):
                        user_preds_reshaped = user_preds.values.reshape(1, -1)
                    else:
                        user_preds_reshaped = user_preds.reshape(1, -1)
                    
                    if hasattr(user_targets, 'values'):
                        user_targets_reshaped = user_targets.values.reshape(1, -1)
                    else:
                        user_targets_reshaped = user_targets.reshape(1, -1)
                    
                    try:
                        ndcg = ndcg_score(user_targets_reshaped, user_preds_reshaped, k=k)
                        ndcg_scores.append(ndcg)
                    except ValueError:
                        # Handle edge cases
                        pass
                
                start_idx = end_idx
        
        return {
            f'ndcg@{k}': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'num_users_evaluated': len(ndcg_scores)
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_scores = self.model.get_score(importance_type='gain')
        
        # Create DataFrame with all features
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': [importance_scores.get(f'f{i}', 0) for i in range(len(self.feature_cols))]
        })
        
        return importance_df.sort_values('importance', ascending=False)
    
    def save_model(self, path: str):
        """Save model and encoders"""
        import pickle
        model_data = {
            'model': self.model,
            'feature_cols': self.feature_cols,
            'categorical_encoders': self.categorical_encoders,
            'params': self.params
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, path: str):
        """Load model and encoders"""
        import pickle
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_cols = model_data['feature_cols']
        self.categorical_encoders = model_data['categorical_encoders']
        self.params = model_data['params']


def create_xgb_training_data(interactions: pd.DataFrame, items: pd.DataFrame, 
                           sasrec_model: SASRecModel, test_size: float = 0.2) -> tuple:
    """Create training data for XGBoost ranking model - FIXED per-user normalization"""
    
    # Split interactions by time for temporal validation
    interactions = interactions.sort_values(Columns.Datetime)
    split_point = int(len(interactions) * (1 - test_size))
    
    train_interactions = interactions.iloc[:split_point]
    test_interactions = interactions.iloc[split_point:]
    
    # FIXED: Use max train timestamp for time features
    current_time = train_interactions[Columns.Datetime].max()
    
    # Create datasets
    train_dataset = Dataset.construct(train_interactions)
    test_dataset = Dataset.construct(test_interactions)
    
    # Get unique users and sample by activity bucket (FIXED: avoid power user bias)
    user_activity = test_interactions.groupby(Columns.User).size().reset_index()
    user_activity.columns = [Columns.User, 'interaction_count']
    
    # Create activity buckets (cold, warm, heavy)
    user_activity['activity_bucket'] = pd.qcut(
        user_activity['interaction_count'], 
        q=3, 
        labels=['cold', 'warm', 'heavy']
    )
    
    # Sample users from each activity bucket
    sampled_users = []
    for bucket in ['cold', 'warm', 'heavy']:
        bucket_users = user_activity[user_activity['activity_bucket'] == bucket][Columns.User].tolist()
        # Sample up to 34 users per bucket (total ~100)
        sample_size = min(34, len(bucket_users))
        if bucket_users:
            sampled_users.extend(np.random.choice(bucket_users, sample_size, replace=False).tolist())
    
    test_users = sampled_users[:100]  # Ensure max 100 users
    test_items = test_interactions[Columns.Item].unique()
    
    # Generate candidates using SASRec
    feature_builder = XGBoostFeatureBuilder(train_interactions, items, sasrec_model)
    
    training_data = []
    
    # FIXED: Process users individually to ensure per-user normalization
    for user_id in test_users[:100]:  # Limit for demo
        # Get SASRec recommendations
        candidates, sasrec_scores = sasrec_model.recommend(
            users=np.array([user_id]),
            dataset=train_dataset,
            k=500,
            filter_viewed=True
        )
        
        # Get weighted targets from test interactions
        user_test_interactions = test_interactions[test_interactions[Columns.User] == user_id]
        item_weights = user_test_interactions.groupby(Columns.Item)[Columns.Weight].sum().to_dict()
        
        # Create candidate DataFrame for this user only
        candidate_df = pd.DataFrame({
            Columns.User: user_id,
            Columns.Item: candidates[0],
            'sasrec_score': sasrec_scores[0],
            'rank_position': range(1, len(candidates[0]) + 1)
        })
        
        # FIXED: Use graded relevance (0-5) instead of raw summed weights
        candidate_df['target'] = candidate_df[Columns.Item].map(item_weights).fillna(0)
        candidate_df['target'] = np.clip(candidate_df['target'], 0, 5)  # Cap at 5 for NDCG
        
        # FIXED: Compute per-user normalization stats BEFORE building features
        max_score = candidate_df['sasrec_score'].max()
        min_score = candidate_df['sasrec_score'].min()
        
        # Build features with per-user normalization (VECTORIZED)
        features_df = feature_builder.build_features_vectorized(
            candidate_df, 
            current_time=current_time,
            max_score=max_score, 
            min_score=min_score
        )
        training_data.append(features_df)
    
    return pd.concat(training_data, ignore_index=True), feature_builder


def compare_models(lightgbm_ranker, xgboost_ranker, test_data: pd.DataFrame) -> dict:
    """Compare LightGBM and XGBoost models"""
    
    # Evaluate LightGBM
    lgb_metrics = lightgbm_ranker.evaluate(test_data)
    
    # Evaluate XGBoost
    xgb_metrics = xgboost_ranker.evaluate(test_data)
    
    # Get feature importance
    xgb_importance = xgboost_ranker.get_feature_importance()
    
    comparison = {
        'lightgbm': lgb_metrics,
        'xgboost': xgb_metrics,
        'xgb_feature_importance': xgb_importance.head(10).to_dict('records')
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    print("XGBoost Ranker Implementation")
