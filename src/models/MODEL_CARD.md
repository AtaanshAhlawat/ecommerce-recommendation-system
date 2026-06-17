# Model Artifacts — Verification & Reference
**Atanshu Ahlawat · 102216100 · Tecorb Technologies**

This document records the trained model artifacts, their configuration, and verification status. All models below were confirmed to load successfully and are usable for inference.

## 1. LightGBM Ranker

| Property | Value |
|----------|-------|
| File | lightgbm_ranker_causal.pkl |
| Status | Loads successfully |
| Model type | Booster |
| Objective | lambdarank |
| Metric | ndcg |
| Trees | 100 |
| Features | 30 |
| Learning rate | 0.05 |
| Num leaves | 31 |
| scale_pos_weight | 859.37 |

### Top 10 Features by Gain

| Rank | Feature | Gain |
|------|---------|------|
| 1 | sasrec_score | 5964.3 |
| 2 | rank_position | 5019.9 |
| 3 | sasrec_score_shifted | 3804.6 |
| 4 | sasrec_score_norm | 3720.1 |
| 5 | inverse_rank | 2735.7 |
| 6 | user_user_first_interaction | 2681.8 |
| 7 | user_user_last_interaction | 2343.9 |
| 8 | user_user_interactions_per_day | 1445.7 |
| 9 | item_item_age_days | 1377.3 |
| 10 | item_item_interaction_count | 898.0 |

The dominant feature is the SASRec retrieval score, confirming that the ranker learns to refine SASRec's ordering rather than override it.

## 2. XGBoost Ranker

| Property | Value |
|----------|-------|
| File | xgboost_ranker_model.pkl |
| Status | Loads successfully |
| Model type | Booster |
| Objective | rank:ndcg |
| Features | 36 |
| NDCG@10 (validation) | 6.89% |

## 3. SASRec Model Configuration

Hyperparameters from the training run and hparams.yaml:

| Hyperparameter | Value |
|----------------|-------|
| Architecture | Self-Attentive Sequential Recommendation |
| Total parameters | 26.3 million |
| Factors (embedding dim) | 256 |
| Transformer blocks | 2 |
| Attention heads | 4 |
| Loss function | gBCE (generalised BCE) |
| gBCE temperature | 0.1 |
| Logits temperature | 1.0 |
| Learning rate | 0.0005 |
| Adam betas | (0.9, 0.98) |
| Epochs | 10 |
| Session max length | 10 |
| Dropout rate | 0.1 |

## 4. Data Artifacts

| File | Rows | Columns | Description |
|------|------|---------|-------------|
| ranker_train_data_part1.parquet | 647,000 | 5 | Labeled candidates |
| ranker_train_features_part1.parquet | 647,000 | 9 | Enriched features |
| sasrec_dataset6.pkl | — | — | RecTools Dataset object (49 MB) |

Positive rate: 752/647,000 = 0.116%