"""
Generate candidates for 100 users using trained SASRec model from retail_rocket_6.py
"""

import numpy as np
import pandas as pd
from rectools import Columns
import os
import torch
import warnings
import typing as tp
from datetime import datetime
from lightning_fabric import seed_everything
from pathlib import Path

from rectools.dataset import Dataset
from rectools.models import load_model

warnings.simplefilter("ignore")

# Enable deterministic behaviour with CUDA >= 10.2
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Random seed
RANDOM_STATE=60
torch.use_deterministic_algorithms(True)
seed_everything(RANDOM_STATE, workers=True)

def load_data_and_model():
    """Load the causal model, preserved dataset, and ground truth labels"""
    print("Loading causal artifacts...")
    
    # 1. Load the trained SASRec model
    model_path = "sasrec_gbce_model6.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run sasrec_model.py.")
    sasrec_model = load_model(model_path)
    
    # 2. Load the exact Dataset object used in training (maps IDs to embeddings correctly)
    import pickle
    dataset_path = "sasrec_dataset6.pkl"
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}. Run sasrec_model.py.")
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # 3. Load the ground truth interactions for labeling/overlap check
    future_path = "future_interactions_ground_truth2.csv"
    if not os.path.exists(future_path):
        raise FileNotFoundError(f"Future interactions file not found: {future_path}. Run sasrec_model.py.")
    future_interactions = pd.read_csv(future_path)
    future_interactions[Columns.Datetime] = pd.to_datetime(future_interactions[Columns.Datetime])
    
    # We also need the history interactions to identify which users can be recommended to
    # These are part of the dataset object, but for simplicity in overlap check, we can use dataset.interactions
    train_interactions = dataset.interactions.df
    
    print(f"Model loaded: {type(sasrec_model)}")
    print(f"Dataset loaded: {dataset.user_id_map.size} users, {dataset.item_id_map.size} items")
    print(f"Future interactions loaded: {len(future_interactions)} rows")
    
    return sasrec_model, dataset, train_interactions, future_interactions

import time

# Add validation after loading:
def validate_compatibility(dataset, model):
    """Check that dataset and model are compatible"""
    # Check user/item mappings exist
    if dataset.user_id_map is None or dataset.item_id_map is None:
        raise ValueError("Dataset missing ID mappings")
    
    # Check model was trained on compatible feature dimensions
    if hasattr(model, 'item_net_block_types'):
        print(f"Model uses item nets: {model.item_net_block_types}")
    
    # Quick test with a sample user
    test_user = list(dataset.user_id_map.external_ids[:1])[0]
    try:
        test_recos = model.recommend(
            users=[test_user],
            dataset=dataset,
            k=1,
            filter_viewed=True
        )
        print(f"✓ Compatibility test passed: Generated {len(test_recos)} test recommendations")
    except Exception as e:
        raise ValueError(f"Model-dataset compatibility check failed: {e}")

def generate_candidates_for_users(sasrec_model, dataset, train_interactions, future_interactions, 
                                 target_users: tp.Optional[tp.List[int]] = None,
                                 n_users: tp.Optional[int] = None, k=500):
    """Generate candidates for users who appear in both train and future windows."""
    
    # Identify users who exist in BOTH history and future
    users_with_history = set(train_interactions[Columns.User].unique())
    
    if target_users is not None:
        # If specific target users provided (e.g. from a specific split)
        test_users = [u for u in target_users if u in users_with_history]
        print(f"Using {len(test_users)} provided target users (filtered for history)")
    else:
        # Default behavior: all users with future activity
        users_with_future = set(future_interactions[Columns.User].unique())
        test_users = list(users_with_history.intersection(users_with_future))
    
    # Filter users with at least N future interactions for meaningful evaluation
    MIN_FUTURE_INTERACTIONS = 1 # Relaxed for candidate generation
    future_user_counts = future_interactions[Columns.User].value_counts()
    active_future_users = future_user_counts[future_user_counts >= MIN_FUTURE_INTERACTIONS].index

    # Get overlap with sufficient future activity
    test_users = list(set(test_users) & set(active_future_users))
    if not test_users:
        print("WARNING: No users with both history and sufficient future interactions!")
        print(f"  Users with history: {len(users_with_history)}")
        print(f"  Users with future: {len(users_with_future)}")
        print(f"  Users with ≥{MIN_FUTURE_INTERACTIONS} future interactions: {len(active_future_users)}")
        return pd.DataFrame()
    if n_users:
        # Sort by future activity count (most active first)
        test_users = sorted(test_users, 
                        key=lambda u: future_user_counts.get(u, 0), 
                        reverse=True)[:n_users]
        
    print(f"\nGenerating candidates for {len(test_users)} overlap users (History + Future)")
    print(f"Users: {test_users[:5]}...")  # Show first 5 users
    
    start_time = time.time()
    
    try:
        print("\nStarting batch recommendation...")
        # Generate candidates for all users at once
        recommendations = sasrec_model.recommend(
            users=test_users,
            dataset=dataset,
            k=k,
            filter_viewed=True,
            on_unsupported_targets="ignore"
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        if not recommendations.empty:
            # Rename for consistency with ranker expectations if needed
            # Currently it returns Columns.User, Columns.Item, score, rank
            final_candidates = recommendations.rename(columns={
                'score': 'sasrec_score',
                'rank': 'rank_position'
            })
            
            print(f"\nSuccessfully generated {len(final_candidates)} total candidates")
            print(f"Time taken: {duration:.2f} seconds")
            print(f"Throughput: {len(test_users) / duration:.2f} users/second")
            print(f"Candidates per user: {k}")
            print(f"Users processed: {final_candidates[Columns.User].nunique()}/{len(test_users)}")
        else:
            final_candidates = pd.DataFrame(columns=[Columns.User, Columns.Item, 'sasrec_score', 'rank_position'])
            print("No candidates generated!")
            
    except Exception as e:
        print(f"Error during batch generation: {e}")
        print("Falling back to chunks if it was a memory issue...")
        # Optional: Implement chunked processing here if the full batch fails
        final_candidates = pd.DataFrame(columns=[Columns.User, Columns.Item, 'sasrec_score', 'rank_position'])
        raise e
    
    return final_candidates

def main():
    """Main function to generate candidates"""
    print("=" * 70)
    print("SASREC CANDIDATE GENERATION FOR ALL ACTIVE USERS")
    print("=" * 70)
    
    try:
        # Load model and data
        sasrec_model, dataset, train_interactions, future_interactions = load_data_and_model()
        validate_compatibility(dataset, sasrec_model)
        # Generate candidates for overlap users
        candidates_df = generate_candidates_for_users(
            sasrec_model, dataset, train_interactions, future_interactions,
            n_users=None, k=500
        )
        
        if not candidates_df.empty:
            # Save candidates
            output_file = f"sasrec_candidates_causal_14d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            candidates_df.to_csv(output_file, index=False)
            print(f"\nCandidates saved to: {output_file}")
            
            # Show summary statistics
            print("\n" + "=" * 70)
            print("CANDIDATE GENERATION SUMMARY")
            print("=" * 70)
            
            print(f"\nDataset shape: {candidates_df.shape}")
            print(f"\nUnique users: {candidates_df[Columns.User].nunique()}")
            print(f"Unique items: {candidates_df[Columns.Item].nunique()}")
            print(f"Candidates per user: {candidates_df.groupby(Columns.User).size().mean():.1f}")
            
            print(f"\nScore statistics:")
            print(candidates_df['sasrec_score'].describe())
            
            print(f"\nFirst 10 candidates:")
            print(candidates_df.head(10))
            
            # Show sample for a few users
            print(f"\nSample candidates by user:")
            sample_users = candidates_df[Columns.User].unique()[:3]
            for user_id in sample_users:
                user_candidates = candidates_df[candidates_df[Columns.User] == user_id]
                print(f"\nUser {user_id} - Top 5 candidates:")
                print(user_candidates[['rank_position', Columns.Item, 'sasrec_score']].head())
            
            print(f"\n✓ Candidate generation completed successfully!")
            print(f"✓ Ready for XGBoost feature building and ranking")
            
        else:
            print("✗ No candidates were generated!")
            
    except Exception as e:
        print(f"✗ Error during candidate generation: {e}")
        raise

if __name__ == "__main__":
    main()