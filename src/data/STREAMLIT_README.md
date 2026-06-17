# E-Commerce Recommendation System — Demo App

**Atanshu Ahlawat · 102216100 · Tecorb Technologies**

A Streamlit interface for the two-stage recommendation pipeline (SASRec retrieval + LightGBM/XGBoost re-ranking) built on the Retail Rocket dataset.

## Setup

```bash
pip install streamlit pandas numpy plotly rectools==0.17.0 lightgbm==4.5.0
```

## Run

Place `app.py` in the repository root (`reccomender-tools/`) and run:

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## File placement

The app auto-detects files in these locations (no config needed):

| File | Expected location |
|------|-------------------|
| `events.csv` | `src/models/retail-rocket/raw/` |
| `item_properties_part1.csv` | `src/models/retail-rocket/raw/` |
| `sasrec_candidates_*.csv` | `src/data/` or `src/evaluation/` |
| `sasrec_gbce_model6.pkl` | `src/models/` |
| `sasrec_dataset6.pkl` | `src/models/` |
| `lightgbm_ranker_causal.pkl` | `src/evaluation/` or `src/utils/` |

The sidebar shows a **live Data Status panel** — green dots for loaded files, grey for missing. The app runs in metrics-only mode if data files are absent, so it never crashes during a demo.

## Pages

1. **Model Metrics** — full results dashboard across all models
2. **Product Search** — search by inferred category, browse popular items
3. **User Recommendations** — per-user SASRec top-10 + LightGBM re-ranking
4. **Item Similarity (I2I)** — SASRec embedding cosine similarity
5. **Model Comparison** — side-by-side metrics chart and insights

## Note on data

The Retail Rocket dataset is fully anonymised. Item IDs and category IDs are integers with no disclosed meaning. Category labels in the UI (e.g. "Electronics") are **inferred** from numeric IDs and clearly marked as such throughout. Real product names and images would require a named dataset (Amazon Products 2023, H&M Fashion) — noted as future work.
