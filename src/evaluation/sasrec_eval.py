import pickle
import pandas as pd
import numpy as np
from rectools import Columns
from rectools.metrics import Recall, NDCG, HitRate
from rectools.dataset import Dataset
from rectools.models import load_model

# ===============================
# CONFIG
# ===============================
DATASET_PATH = "sasrec_dataset5.pkl"
MODEL_PATH = "sasrec_gbce_model6.pkl"
FUTURE_PATH = "future_interactions_ground_truth2.csv"

TOP_K = [5, 10, 20, 500, 1000]
MIN_GT_INTERACTIONS = 1  # minimum future interactions per user

# ===============================
# LOAD ARTIFACTS
# ===============================
print("Loading dataset...")
with open(DATASET_PATH, "rb") as f:
    dataset: Dataset = pickle.load(f)

print("Loading trained model...")
model = load_model(MODEL_PATH)

print("Loading future interactions...")
future = pd.read_csv(
    FUTURE_PATH,
    parse_dates=[Columns.Datetime]
)

# ===============================
# FILTER FUTURE DATA (CRITICAL)
# ===============================
print("\nFiltering future interactions...")

# Keep only users/items known to the dataset
known_users = set(dataset.user_id_map.external_ids)
known_items = set(dataset.item_id_map.external_ids)

future = future[
    future[Columns.User].isin(known_users) &
    future[Columns.Item].isin(known_items)
]

# Remove duplicates (user,item) in future window
future = (
    future.sort_values(Columns.Datetime)
          .drop_duplicates([Columns.User, Columns.Item], keep="first")
)

# Users with enough future interactions
user_future_counts = future.groupby(Columns.User).size()
eligible_users = user_future_counts[user_future_counts >= MIN_GT_INTERACTIONS].index

future = future[future[Columns.User].isin(eligible_users)]

print(f"Users with future interactions: {future[Columns.User].nunique():,}")
print(f"Future interactions used: {len(future):,}")

# ===============================
# BUILD GROUND TRUTH
# ===============================
# rectools expects:
# user_id | item_id | relevance
gt = future[[Columns.User, Columns.Item]].copy()
gt["relevance"] = 1.0

# ===============================
# GENERATE RECOMMENDATIONS
# ===============================
print("\nGenerating recommendations...")

users = gt[Columns.User].unique().tolist()

reco = model.recommend(
    users=users,
    dataset=dataset,
    k=max(TOP_K),
    filter_viewed=True,
    on_unsupported_targets="ignore"
)

print(f"Generated {len(reco):,} recommendations")

# ===============================
# EVALUATE METRICS
# ===============================
print("\nEvaluating ranking metrics...\n")

metrics = []
for k in TOP_K:
    metrics.extend([
        Recall(k=k),
        NDCG(k=k),
        HitRate(k=k)
    ])

results = {}
for metric in metrics:
    value = metric.calc(
        reco=reco,
        interactions=gt,
    )
    results[str(metric)] = value

# ===============================
# REPORT
# ===============================
print("==== SASRec Evaluation Results ====")
for name, value in results.items():
    print(f"{name:<15}: {value:.5f}")

# ===============================
# OPTIONAL: PER-USER DIAGNOSTICS
# ===============================
print("\nBasic diagnostics:")
print(f"Users evaluated: {len(users):,}")
print(f"Avg GT items/user: {gt.groupby(Columns.User).size().mean():.2f}")
print(f"Avg reco items/user: {reco.groupby(Columns.User).size().mean():.2f}")

print("\n=== EVALUATION COMPLETE ===")
