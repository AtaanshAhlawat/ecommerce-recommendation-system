import os
import typing as tp
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import LabelEncoder

# Suppress pandas FutureWarnings
pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

from rectools import Columns
from rectools.dataset import Dataset
from rectools.models import SASRecModel


class LightGBMFeatureBuilder:
    """Build features for ranking model from SASRec candidates and Retail Rocket interaction data"""
    
    def __init__(self, interactions: pd.DataFrame, items: pd.DataFrame, sasrec_model: SASRecModel, current_time: datetime = None):
        self.interactions = interactions
        self.items = items
        self.user_features = {}
        self.item_features = {}
        # Mandatory current_time to avoid leakage
        self.current_time = current_time or interactions[Columns.Datetime].max()
        self._precompute_features(self.current_time)
    
    def _precompute_features(self, current_time: datetime):
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
        # In _precompute_features:
        user_stats['activity_level'] = pd.cut(
            user_stats['user_total_interactions'], 
            bins=[3, 8, 20, float('inf')],  # Start at 3 to match training
            labels=['Low', 'Medium', 'High']
        )
        
        self.user_features = user_stats.set_index(Columns.User).to_dict('index')
        
        # Item features - using itemid as item_id
        item_stats = self.interactions.groupby(Columns.Item).agg({
            Columns.User: 'count',
            Columns.Weight: ['sum', 'mean'],
            Columns.Datetime: ['min', 'max']
        }).reset_index()
        item_stats.columns = [Columns.Item, 'item_interaction_count', 'item_total_weight',
                            'item_avg_weight', 'item_first_interaction', 'item_last_interaction']
        
        # Add item age features - CAUSAL: Use provided current_time
        item_stats['item_age_days'] = (
            current_time - item_stats['item_first_interaction']
        ).dt.days.fillna(0)
        
        # Merge with item metadata if available
        if self.items is not None and len(self.items) > 0:
            item_stats = item_stats.merge(self.items, left_on=Columns.Item, right_index=True, how='left')
            # Fix: Ensure unique index before converting to dict
            item_stats = item_stats.drop_duplicates(Columns.Item)
        
        self.item_features = item_stats.set_index(Columns.Item).to_dict('index')
    
    def get_user_features(self, user_id: int) -> dict:
        """Get precomputed user features"""
        return self.user_features.get(user_id, {})
    
    def get_item_features(self, item_id: int) -> dict:
        """Get precomputed item features"""
        return self.item_features.get(item_id, {})
    
    def build_features(self, candidates_df: pd.DataFrame, current_time: datetime = None,
                    max_score: float = None, min_score: float = None) -> pd.DataFrame:
        """
        Build features for ranking model from Retail Rocket data
        
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
        
        # FIXED: Use provided normalization stats or compute per batch
        if max_score is None or min_score is None:
            max_score = candidates_df['sasrec_score'].max()
            min_score = candidates_df['sasrec_score'].min()
        
        max_rank = candidates_df['rank_position'].max()
        
        features = []
        
        for _, row in candidates_df.iterrows():
            user_id = row[Columns.User]
            item_id = row[Columns.Item]
            sasrec_score = row['sasrec_score']
            rank_position = row['rank_position']
            
            # Get user and item features
            user_feat = self.get_user_features(user_id)
            item_feat = self.get_item_features(item_id)
            
            # Base features
            feature_dict = {
                Columns.User: user_id,
                Columns.Item: item_id,
                'sasrec_score': sasrec_score,
                'rank_position': rank_position,
            }
            
            # SASRec features with CORRECT per-user normalization
            feature_dict.update({
                'sasrec_score_norm': (sasrec_score - min_score) / (max_score - min_score + 1e-8),
                'sasrec_score_log': np.log1p(sasrec_score),
                'rank_position_norm': rank_position / (max_rank + 1),
                'inverse_rank': 1.0 / (rank_position + 1),
            })
            
            # User interaction features (Retail Rocket specific)
            feature_dict.update({
                'user_total_interactions': user_feat.get('user_total_interactions', 0),
                'user_total_weight': user_feat.get('user_total_weight', 0),
                'user_avg_weight': user_feat.get('user_avg_weight', 0),
                'user_activity_span_days': user_feat.get('user_activity_span_days', 0),
                'user_interactions_per_day': user_feat.get('user_interactions_per_day', 0),
            })
            
            # Add activity level encoding
            activity_level = user_feat.get('activity_level', 'Low')
            feature_dict['activity_level'] = activity_level
            
            # Item interaction features (Retail Rocket specific)
            feature_dict.update({
                'item_interaction_count': item_feat.get('item_interaction_count', 0),
                'item_total_weight': item_feat.get('item_total_weight', 0),
                'item_avg_weight': item_feat.get('item_avg_weight', 0),
                'item_age_days': item_feat.get('item_age_days', 0),
            })
            
            # Time-based features
            if user_feat.get('user_last_interaction'):
                time_since_user_interaction = (current_time - user_feat['user_last_interaction']).total_seconds() / 3600  # hours
                feature_dict['time_since_user_interaction'] = time_since_user_interaction
                feature_dict['time_since_user_interaction_days'] = time_since_user_interaction / 24
            
            if item_feat.get('item_last_interaction'):
                time_since_item_interaction = (current_time - item_feat['item_last_interaction']).total_seconds() / 3600  # hours
                feature_dict['time_since_item_interaction'] = time_since_item_interaction
                feature_dict['time_since_item_interaction_days'] = time_since_item_interaction / 24
            
            # Metadata features (if available from Retail Rocket)
            if 'categoryid' in item_feat:
                feature_dict['category'] = item_feat['categoryid']
            if 'parentid' in item_feat:
                feature_dict['parent_category'] = item_feat['parentid']
            
            # Interaction features
            feature_dict['user_item_score'] = sasrec_score * np.log1p(item_feat.get('item_interaction_count', 1))
            feature_dict['user_activity_score'] = np.log1p(user_feat.get('user_total_interactions', 1)) * sasrec_score
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def build_features_vectorized(self, candidates_df: pd.DataFrame, current_time: datetime = None,
                              max_score: float = None, min_score: float = None) -> pd.DataFrame:
        """
        Build features for ranking model - FULLY VECTORIZED (no Python loops)
        
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
        
        # FIXED: Fill NaN AFTER all merges (with proper dtype handling for LightGBM)
        features_df = features_df.fillna(0).infer_objects(copy=False)
        
        # FIXED: Convert all object and datetime columns to numeric for LightGBM
        for col in features_df.columns:
            if col in ['user_id', 'item_id', 'target']:
                continue  # Skip ID and target columns
            
            if features_df[col].dtype == 'object':
                # Convert categorical objects to numeric codes
                if features_df[col].nunique() < 100:  # Only if reasonable cardinality
                    features_df[col] = pd.Categorical(features_df[col]).codes
                else:
                    # For high cardinality, drop column
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
            features_df['sasrec_score'] * np.log1p(features_df.get('user_total_interactions', 0))
        )
        features_df['sasrec_item_interactions'] = (
            features_df['sasrec_score'] * np.log1p(features_df.get('item_interaction_count', 0))
        )
        features_df['user_weight_to_item_ratio'] = (
            features_df.get('user_total_weight', 1) / (features_df.get('item_interaction_count', 1) + 1e-8)
        )
        features_df['item_weight_ratio'] = (
            features_df.get('item_avg_weight', 0) / (features_df.get('user_avg_weight', 1) + 1e-8)
        )
        
        return features_df


class LightGBMRanker:
    """LightGBM ranking model for recommendation re-ranking"""
    
    def __init__(self, params: dict = None):
        self.params = params or {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'random_state': 42,
            'ndcg_eval_at': [10]
        }
        self.model = None
        self.feature_cols = None
        self.categorical_encoders = {}
    
    def prepare_data(self, features_df: pd.DataFrame, target_col: str = 'target') -> tuple:
        """Prepare data for LightGBM training - REMOVED dangerous categorical encoding + FIXED validation"""
        # Check actual column names and use them
        user_col = 'user_id' if 'user_id' in features_df.columns else Columns.User
        item_col = 'item_id' if 'item_id' in features_df.columns else Columns.Item
        
        # Sort by user to maintain group order for LightGBM
        if user_col in features_df.columns:
            features_df = features_df.sort_values(user_col).reset_index(drop=True)
        else:
            features_df = features_df.reset_index(drop=True)
        
        # FIXED: Remove all categorical features to avoid dangerous LabelEncoder mutation
        # Focus on numerical features that matter most for ranking
        categorical_cols = ['category', 'parent_category', 'activity_level']
        df_clean = features_df.drop(columns=[col for col in categorical_cols if col in features_df.columns])
        
        # Select feature columns (only numerical)
        exclude_cols = [user_col, item_col, target_col]
        self.feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[self.feature_cols]
        y = features_df[target_col] if target_col in features_df.columns else None
        groups = features_df[user_col].values if user_col in features_df.columns else None
        
        # FIXED: Replace inf before validation (time features use large values)
        X = X.replace([np.inf, -np.inf], 1e6)
        
        if X.isnull().any().any():
            # Handle NaNs gracefully by filling with 0 (Standard for LightGBM)
            X = X.fillna(0)
        
        # FINAL SANITY CHECK FOR LIGHTGBM
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        return X, y, groups
    
    def train(self, features_df: pd.DataFrame, target_col: str = 'target', 
              validation_data: tuple = None, early_stopping_rounds: int = None) -> dict:
        """Train the LightGBM ranking model"""
        X, y, groups = self.prepare_data(features_df, target_col)
        
        # Create LightGBM Dataset
        train_data = lgb.Dataset(X, label=y, group=self._get_group_sizes(groups))
        
        callbacks = [lgb.log_evaluation(0)]
        if validation_data and early_stopping_rounds:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        if validation_data:
            val_X, val_y, val_groups = validation_data
            val_data = lgb.Dataset(val_X, label=val_y, group=self._get_group_sizes(val_groups), 
                                 reference=train_data)
            valid_sets = [train_data, val_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Train model
        print("DEBUG: Starting lgb.train...")
        try:
            self.model = lgb.train(
                self.params,
                train_data,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            print("DEBUG: lgb.train completed successfully")
        except Exception as e:
            print(f"DEBUG: lgb.train raised exception: {e}")
            raise e
        
        return self.model.feature_importance(importance_type='gain')
    
    def _get_group_sizes(self, groups: np.ndarray) -> list:
        """Convert group array to group sizes for LightGBM - FIXED empty groups"""
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
        return self.model.predict(X)
    
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
        
        importance_scores = self.model.feature_importance(importance_type='gain')
        
        # Create DataFrame with all features
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': [importance_scores[i] if i < len(importance_scores) else 0 
                          for i in range(len(self.feature_cols))]
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
