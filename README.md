# E-Commerce Recommendation System

**Atanshu Ahlawat · 102216100 · Thapar Institute of Engineering and Technology**  
**AI/ML Intern · Tecorb Technologies, Noida · December 2025 – June 2026**

---

An end-to-end personalised product recommendation system built on the [Retail Rocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset). The system implements a two-stage retrieval-and-re-ranking pipeline — a SASRec Transformer retrieves top-500 candidates per user, which are then re-ranked by gradient-boosted rankers. Classical baselines (Implicit ALS, PureSVD) are included for comparison.

All models are evaluated offline on held-out future interactions using standard recommendation metrics.

---

## Project Structure

```
reccomender-tools/
├── src/
│   ├── models/          # Model training scripts
│   │   ├── iALS.py              # Implicit ALS (256 factors)
│   │   ├── svd-model.py         # PureSVD on Retail Rocket
│   │   ├── svd-amazon.py        # PureSVD on Amazon Electronics (RMSE eval)
│   │   ├── sasrec_model.py      # SASRec training (26.3M params, 10 epochs)
│   │   └── i2i.py               # Item-to-Item similarity model
│   ├── data/            # Pipeline scripts
│   │   ├── generate_candidate.py      # SASRec → top-500 candidates per user
│   │   ├── prepare_ranker_data.py     # Label candidates from ground truth
│   │   └── enrich_ranker_features.py  # Build feature set for rankers
│   ├── evaluation/      # Evaluation scripts
│   │   ├── sasrec_eval.py             # SASRec multi-K evaluation
│   │   ├── test_xgboost.py            # XGBoost ranker training + eval
│   │   └── train_lightgbm_ranker.py   # LightGBM ranker training
│   └── utils/           # Helpers and utilities
├── app.py               # Streamlit demo interface
├── app2.py              # Streamlit demo interface (updated)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Models

| Model        | Type                    | Key Config                                 |
| ------------ | ----------------------- | ------------------------------------------ |
| Implicit ALS | Collaborative filtering | 256 factors, alpha=40, reg=0.05, 30 iter   |
| PureSVD      | Matrix factorisation    | 128 factors, RecTools                      |
| SASRec       | Sequential Transformer  | 26.3M params, 2 blocks, 4 heads, gBCE loss |
| LightGBM     | Learning-to-rank        | lambdarank, 100 trees, 26 features         |
| XGBoost      | Learning-to-rank        | rank:ndcg, 36 features                     |
| I2I          | Item similarity         | SASRec embeddings, cosine similarity       |

---

## Results

### Retrieval Models (Retail Rocket, temporal 70/30 split)

| Model        | Precision@10 | Recall@10 | MAP@10 | NDCG@10   | Coverage   |
| ------------ | ------------ | --------- | ------ | --------- | ---------- |
| Implicit ALS | **4.08%**    | 2.50%     | 1.11%  | **4.53%** | **22.75%** |
| PureSVD      | 1.03%        | 0.42%     | 0.21%  | 1.24%     | 4.99%      |
| SASRec       | —            | **4.36%** | —      | 0.89%     | —          |

### SASRec at Multiple K (8,931 test users)

| K   | Recall@K | NDCG@K | HitRate@K |
| --- | -------- | ------ | --------- |
| 10  | 4.36%    | 0.89%  | 6.57%     |
| 500 | 22.70%   | —      | 28.50%    |

Recall@500 = 22.70% is the retrieval ceiling — the maximum recall the two-stage pipeline can achieve.

### Re-ranking

| Model           | NDCG@10   | Notes                               |
| --------------- | --------- | ----------------------------------- |
| XGBoost ranker  | **6.89%** | 73 validation users, 36 features    |
| LightGBM ranker | —         | HitRate eval incomplete (env issue) |

### SVD on Amazon Electronics (explicit ratings, k-sweep)

| k         | RMSE   | NDCG@10 |
| --------- | ------ | ------- |
| 50 (best) | 3.4794 | 0.81%   |
| 100       | 3.4805 | 0.76%   |
| 150       | 3.4810 | 0.72%   |

---

## Dataset

**Retail Rocket E-Commerce Dataset** — 2.75M interaction events (views, add-to-cart, transactions) across 1.4M visitors and 235K items, May–September 2015. Fully anonymised — no product names, images, or descriptions.

Download from Kaggle and place files in `src/models/retail-rocket/raw/`:

- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

**Note:** Large data files and model checkpoints (sasrec_gbce_model6.pkl, sasrec_dataset6.pkl, events.csv, item_properties files) are excluded from this repository due to size. See `.gitignore`.

---

## Installation

```bash
git clone <repo-url>
cd reccomender-tools

python3 -m venv rec_env
source rec_env/bin/activate

pip install -r requirements.txt
```

### Key dependencies

```
rectools==0.17.0
implicit==0.7.2
lightgbm==4.5.0
xgboost==2.1.1
torch==2.2.2
pytorch-lightning==2.2.2
scipy==1.11.4
streamlit
pandas
numpy
```

---

## Running the Pipeline

```bash
# 1. Train baseline models
python src/models/iALS.py
python src/models/svd-model.py

# 2. Train SASRec retrieval model
python src/models/sasrec_model.py

# 3. Evaluate SASRec
python src/evaluation/sasrec_eval.py

# 4. Generate candidates (top-500 per user)
python src/data/generate_candidate.py

# 5. Prepare and enrich ranker training data
python src/data/prepare_ranker_data.py
python src/data/enrich_ranker_features.py

# 6. Train re-rankers
python src/evaluation/train_lightgbm_ranker.py
python src/evaluation/test_xgboost.py

# 7. Train item-to-item similarity
python src/models/i2i.py
```

---

## Demo Interface

A five-page Streamlit interface demonstrates all models:

```bash
streamlit run app2.py
```

Opens at `http://localhost:8501`. Pages:

- **Model Metrics** — full results dashboard with all evaluation numbers
- **Product Search** — browse items by inferred category
- **User Recommendations** — five tabs, one per model (SASRec, LightGBM, XGBoost, ALS, SVD), each producing independent top-10 recommendations for a given user
- **Item Similarity** — cosine similarity via SASRec embeddings
- **Model Comparison** — side-by-side chart of all models

---

## Key Findings

- **ALS vs SVD:** ALS outperforms SVD ~4× on the same data because it is specifically designed for implicit feedback — confidence-weighted factorisation beats plain matrix approximation on sparse data.
- **SASRec vs ALS:** SASRec achieves higher Recall@10 (4.36% vs 2.50%) by capturing sequential patterns, but lower NDCG@10 (0.89% vs 4.53%). This complementarity motivates the two-stage architecture.
- **Re-ranking ceiling:** Recall@500 = 22.7% is the hard ceiling. The re-ranker improves ordering within those 500 candidates but cannot surface items outside them.
- **Feature importance:** The dominant LightGBM feature is `sasrec_score` (gain 5964), confirming the ranker refines rather than overrides the retrieval signal.

---

## Limitations

- Dataset is fully anonymised — no product names, images, or content features
- LightGBM HitRate@10 evaluation incomplete due to environment constraints
- Offline evaluation only — no online A/B testing
- No cold-start handling for new users

---

## Mentors

- **Industry Mentor:** Mr. Jai Rajput, Tecorb Technologies
- **Faculty Mentor:** Dr. Sandeep Verma, TIET Patiala
