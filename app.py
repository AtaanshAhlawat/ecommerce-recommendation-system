"""
E-Commerce Recommendation System — Streamlit Demo
Atanshu Ahlawat | 102216100 | Tecorb Technologies
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

st.set_page_config(
    page_title="RecSys Demo | Atanshu Ahlawat",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #0a0a0f; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460; border-radius: 12px;
    padding: 20px; text-align: center; margin: 8px 0;
}
.metric-value { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; color: #e94560; }
.metric-label { font-size: 0.75rem; color: #8892b0; text-transform: uppercase; letter-spacing: 2px; margin-top: 4px; }
.model-badge {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; font-family: 'Space Mono', monospace;
    letter-spacing: 1px; text-transform: uppercase;
}
.badge-als  { background: #0f3460; color: #53d8fb; border: 1px solid #53d8fb; }
.badge-svd  { background: #1a0a2e; color: #c084fc; border: 1px solid #c084fc; }
.badge-sasrec { background: #0a1f12; color: #4ade80; border: 1px solid #4ade80; }
.badge-xgb  { background: #2a1500; color: #fb923c; border: 1px solid #fb923c; }
.badge-lgb  { background: #1f1000; color: #fbbf24; border: 1px solid #fbbf24; }
.badge-i2i  { background: #1a0f1f; color: #f472b6; border: 1px solid #f472b6; }
.product-card {
    background: #111827; border: 1px solid #1f2937;
    border-radius: 10px; padding: 14px 16px; margin: 6px 0; transition: all 0.2s;
}
.product-card:hover { border-color: #e94560; }
.rank-num { font-family: 'Space Mono', monospace; color: #e94560; font-size: 0.95rem; font-weight: 700; }
.score-pill { background: #1f2937; color: #9ca3af; border-radius: 20px; padding: 2px 10px; font-size: 0.72rem; font-family: 'Space Mono', monospace; }
.anon-tag { background: #1f1a00; color: #ca8a04; border: 1px solid #854d0e; border-radius: 4px; padding: 1px 6px; font-size: 0.65rem; font-family: 'Space Mono', monospace; }
.section-header { font-family: 'Space Mono', monospace; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 3px; color: #4b5563; border-bottom: 1px solid #1f2937; padding-bottom: 8px; margin: 24px 0 16px 0; }
.stButton > button { background: #e94560; color: white; border: none; border-radius: 8px; font-family: 'Space Mono', monospace; font-weight: 700; letter-spacing: 1px; padding: 8px 24px; width: 100%; }
.stButton > button:hover { background: #c73652; }
.info-box { background: #111827; border-left: 3px solid #e94560; padding: 12px 16px; border-radius: 0 8px 8px 0; font-size: 0.85rem; color: #9ca3af; margin: 8px 0; }
.hero-title { font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #f9fafb; line-height: 1.2; }
.hero-sub { color: #6b7280; font-size: 0.9rem; margin-top: 4px; }
.disclaimer-box {
    background: #1a1500; border: 1px solid #854d0e; border-radius: 8px;
    padding: 12px 16px; font-size: 0.8rem; color: #ca8a04; margin: 12px 0;
}
.future-box {
    background: #0a1a0a; border: 1px dashed #4ade80; border-radius: 8px;
    padding: 12px 16px; font-size: 0.8rem; color: #4ade80; margin: 12px 0;
}
.cat-btn-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE      = Path(__file__).parent
MODELS_DIR = BASE / "src" / "models"
DATA_DIR   = BASE / "src" / "data"
EVAL_DIR   = BASE / "src" / "evaluation"
RAW_DIR    = MODELS_DIR / "retail-rocket" / "raw"

# ─── CATEGORY METADATA ───────────────────────────────────────────────────────
# These are inferred mappings — Retail Rocket never disclosed category meanings.
# Marked as "inferred" throughout the UI.
INFERRED_CATEGORY_NAMES = {
    "498": "Electronics", "438": "Clothing & Fashion", "1613": "Home & Garden",
    "1258": "Sports & Outdoors", "963": "Books & Media", "722": "Toys & Games",
    "840": "Health & Beauty", "1111": "Automotive", "567": "Food & Grocery",
    "234": "Office Supplies", "789": "Jewelry & Accessories", "345": "Baby & Kids",
    "901": "Pet Supplies", "456": "Musical Instruments", "678": "Tools & Hardware",
    "123": "Travel & Luggage", "321": "Art & Crafts", "654": "Garden & Outdoors",
    "987": "Computers & Tablets", "147": "Cameras & Photography",
    "258": "Kitchen & Dining", "369": "Furniture", "741": "Lighting",
    "852": "Bedding & Bath",
}

CATEGORY_EMOJI = {
    "498": "📱", "987": "💻", "147": "📷", "438": "👗", "789": "💍",
    "1258": "⚽", "840": "💊", "1613": "🏡", "258": "🍳", "369": "🛋️",
    "741": "💡", "852": "🛏️", "963": "👟", "722": "🎮", "345": "👶",
    "901": "🐾", "567": "🛒", "1111": "🚗", "234": "📋", "456": "🎸",
    "678": "🔧", "123": "✈️", "321": "🎨", "654": "🌿",
}

SEARCH_TERMS = {
    "electronics": ["498", "987", "147"],
    "phone": ["498", "987"], "laptop": ["987", "498"], "computer": ["987", "498"],
    "tablet": ["987", "498"], "camera": ["147"], "clothing": ["438", "789"],
    "fashion": ["438", "789"], "shoes": ["963", "438"], "sport": ["1258"],
    "fitness": ["1258", "840"], "home": ["1613", "258", "369", "741", "852"],
    "kitchen": ["258"], "furniture": ["369"], "garden": ["1613", "654"],
    "beauty": ["840"], "health": ["840"], "book": ["963"], "toy": ["722"],
    "kids": ["345", "722"], "baby": ["345"], "pet": ["901"], "food": ["567"],
    "auto": ["1111"], "car": ["1111"], "office": ["234"], "music": ["456"],
    "tool": ["678"], "travel": ["123"],
}

def get_category_name(cat_id):
    return INFERRED_CATEGORY_NAMES.get(str(cat_id), f"Category {cat_id}")

def get_category_emoji(cat_id):
    return CATEGORY_EMOJI.get(str(cat_id), "📦")

def search_categories(query):
    q = query.lower().strip()
    matched = set()
    for term, cat_ids in SEARCH_TERMS.items():
        if term in q or q in term:
            matched.update(cat_ids)
    if not matched:
        for cat_id, name in INFERRED_CATEGORY_NAMES.items():
            if q in name.lower():
                matched.add(cat_id)
    return list(matched)

# ─── DATA LOADERS ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_events():
    p = RAW_DIR / "events.csv"
    if not p.exists():
        return None
    events = pd.read_csv(p, dtype={'visitorid': 'int32', 'itemid': 'int32'})
    events['timestamp'] = pd.to_datetime(events['timestamp'], unit='ms')
    events['weight'] = events['event'].map({'view': 1.0, 'addtocart': 3.0, 'transaction': 5.0})
    interactions = events.groupby(['visitorid', 'itemid']).agg(
        weight=('weight', 'sum'), timestamp=('timestamp', 'max')
    ).reset_index()
    interactions.columns = ['user_id', 'item_id', 'weight', 'datetime']
    uc = interactions['user_id'].value_counts()
    ic = interactions['item_id'].value_counts()
    return interactions[
        interactions['user_id'].isin(uc[uc >= 5].index) &
        interactions['item_id'].isin(ic[ic >= 5].index)
    ]

@st.cache_data(show_spinner=False)
def load_categories():
    p = RAW_DIR / "item_properties_part1.csv"
    if not p.exists():
        return {}
    props = pd.read_csv(p, nrows=500000)
    cats = props[props['property'] == 'categoryid'][['itemid', 'value']].drop_duplicates('itemid')
    return dict(zip(cats['itemid'].astype(str), cats['value'].astype(str)))

@st.cache_resource(show_spinner=False)
def load_sasrec_model():
    try:
        from rectools.models import load_model as rt_load
        for p in [MODELS_DIR / "sasrec_gbce_model6.pkl", BASE / "sasrec_gbce_model6.pkl"]:
            if p.exists():
                return rt_load(str(p))
    except:
        pass
    return None

@st.cache_resource(show_spinner=False)
def load_sasrec_dataset():
    for p in [MODELS_DIR / "sasrec_dataset6.pkl", BASE / "sasrec_dataset6.pkl",
              DATA_DIR / "sasrec_dataset6.pkl", EVAL_DIR / "sasrec_dataset6.pkl"]:
        if p.exists():
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None

@st.cache_resource(show_spinner=False)
def load_lgb_ranker():
    """Load LightGBM ranker. The pkl is a plain dict with keys:
    model (lgb.Booster), feature_cols (list), categorical_encoders (dict), params (dict).
    No custom class required."""
    import lightgbm as lgb
    for p in [EVAL_DIR / "lightgbm_ranker_causal.pkl",
              BASE / "src" / "utils" / "lightgbm_ranker_causal.pkl",
              BASE / "lightgbm_ranker_causal.pkl",
              MODELS_DIR / "lightgbm_ranker_causal.pkl"]:
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and 'model' in obj:
                    return obj  # dict with model/feature_cols/params
            except Exception:
                pass
    return None

@st.cache_resource(show_spinner=False)
def load_xgb_ranker():
    """Load XGBoost ranker pkl — plain dict with model/feature_cols/params."""
    for p in [EVAL_DIR / "xgboost_ranker_model.pkl",
              BASE / "src" / "utils" / "xgboost_ranker_model.pkl",
              BASE / "xgboost_ranker_model.pkl",
              MODELS_DIR / "xgboost_ranker_model.pkl"]:
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and 'model' in obj:
                    return obj
            except Exception:
                pass
    return None

@st.cache_resource(show_spinner=False)
def get_als_model():
    """Train and cache ALS model once from events data."""
    try:
        import implicit
        from scipy.sparse import csr_matrix
        interactions = load_events()
        if interactions is None or interactions.empty:
            return None, None, None
        users = interactions['user_id'].astype('category')
        items = interactions['item_id'].astype('category')
        data = interactions['weight'].values.astype(float)
        user_idx = users.cat.codes.values
        item_idx = items.cat.codes.values
        mat = csr_matrix(
            (data, (user_idx, item_idx)),
            shape=(users.cat.categories.size, items.cat.categories.size)
        )
        model = implicit.als.AlternatingLeastSquares(
            factors=256, regularization=0.05, iterations=30, alpha=40,
            use_gpu=False, random_state=42
        )
        model.fit(mat, show_progress=False)
        return model, mat, users.cat.categories, items.cat.categories
    except Exception as e:
        return None, None, str(e), None

def get_als_recommendations(user_id, n=10):
    """Get ALS top-N for a user from the cached model."""
    result = get_als_model()
    if result[0] is None:
        # result[2] holds the error string when model is None
        err = result[2] if isinstance(result[2], str) else "events.csv not loaded"
        return [], err
    model, mat, user_cats, item_cats = result
    uid_int = int(user_id)
    # Robust lookup — cast both sides to int for comparison
    user_cat_ints = user_cats.astype(int)
    if uid_int not in user_cat_ints:
        return [], f"User {uid_int} not in ALS training set (needs ≥5 interactions in events.csv)"
    uid_pos = int(np.where(user_cat_ints == uid_int)[0][0])
    try:
        ids, scores = model.recommend(
            uid_pos, mat[uid_pos], N=n, filter_already_liked_items=True
        )
        return [(int(item_cats[i]), float(s)) for i, s in zip(ids, scores)], None
    except Exception as e:
        return [], str(e)

@st.cache_resource(show_spinner=False)
def get_svd_model():
    """Train and cache PureSVD model once from events data (128 factors)."""
    try:
        from rectools.models import PureSVDModel
        from rectools.dataset import Dataset
        interactions = load_events()
        if interactions is None or interactions.empty:
            return None, None, None, None
        # RecTools needs columns: user_id, item_id, weight, datetime
        df = interactions.rename(columns={'weight': 'weight', 'datetime': 'datetime'})
        dataset = Dataset.construct(
            interactions_df=df,
            user_features_df=None,
            item_features_df=None,
        )
        model = PureSVDModel(factors=128)
        model.fit(dataset)
        return model, dataset, interactions['user_id'].unique(), interactions['item_id'].unique()
    except Exception as e:
        return None, None, str(e), None

def get_svd_recommendations(user_id, n=10):
    """Get PureSVD top-N for a user from the cached model."""
    result = get_svd_model()
    if result[0] is None:
        err = result[2] if isinstance(result[2], str) else "events.csv not loaded"
        return [], err
    model, dataset, user_ids, item_ids = result
    uid_int = int(user_id)
    if uid_int not in user_ids:
        return [], f"User {uid_int} not in SVD training set (needs ≥5 interactions in events.csv)"
    try:
        import pandas as pd
        
        recos = model.recommend(
    users=[uid_int],
    dataset=dataset,
    k=n,
    filter_viewed=True,
)
        if recos.empty:
            return [], "No recommendations generated"
        return [(int(r['item_id']), float(r['score'])) for _, r in recos.iterrows()], None
    except Exception as e:
        return [], str(e)

@st.cache_data(show_spinner=False)
def load_candidates():
    paths = (list(DATA_DIR.glob("sasrec_candidates_*.csv")) +
             list(EVAL_DIR.glob("sasrec_candidates_*.csv")) +
             list(BASE.glob("sasrec_candidates_*.csv")))
    if paths:
        return pd.read_csv(max(paths, key=os.path.getmtime))
    return None

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def product_card(rank, item_id, score, cat_id="", score_label="score"):
    """Render a product card — factual, no fake names."""
    cat_name = get_category_name(cat_id)
    emoji = get_category_emoji(cat_id)
    return f"""
    <div class="product-card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div style="display:flex;align-items:center;gap:12px;">
                <span style="font-size:1.8rem;">{emoji}</span>
                <div>
                    <div>
                        <span class="rank-num">#{rank:02d}</span>
                        &nbsp;
                        <span style="color:#f9fafb;font-weight:600;">Item {item_id}</span>
                        &nbsp;
                        <span class="anon-tag">anonymised</span>
                    </div>
                    <div style="font-size:0.75rem;color:#6b7280;font-family:'Space Mono',monospace;margin-top:3px;">
                        Category (inferred): {cat_name} &nbsp;·&nbsp; Cat ID: {cat_id}
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <span class="score-pill">{score_label}: {score:.4f}</span>
            </div>
        </div>
    </div>
    """

def get_user_history(user_id, interactions, n=5):
    if interactions is None:
        return pd.DataFrame()
    return interactions[interactions['user_id'] == user_id].nlargest(n, 'weight')

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🛒 RecSys</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">E-Commerce Recommendation System<br>Atanshu Ahlawat · 102216100</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Model Metrics",
        "🔍 Product Search",
        "👤 User Recommendations",
        "🔗 Item Similarity (I2I)",
        "⚖️ Model Comparison",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Live data-status panel — shows mentor exactly what is loaded
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:0.7rem;text-transform:uppercase;letter-spacing:3px;color:#4b5563;">Data Status</div>', unsafe_allow_html=True)

    def _status_line(label, present):
        dot = "🟢" if present else "⚪"
        color = "#4ade80" if present else "#6b7280"
        state = "loaded" if present else "not found"
        return f'<div style="font-size:0.78rem;color:{color};font-family:Space Mono,monospace;margin:3px 0;">{dot} {label}: {state}</div>'

    _events_ok = (RAW_DIR / "events.csv").exists()
    _cats_ok = (RAW_DIR / "item_properties_part1.csv").exists()
    _cands_ok = bool(list(DATA_DIR.glob("sasrec_candidates_*.csv")) +
                     list(EVAL_DIR.glob("sasrec_candidates_*.csv")) +
                     list(BASE.glob("sasrec_candidates_*.csv")))
    _sasrec_model_ok = any((d / "sasrec_gbce_model6.pkl").exists() for d in [MODELS_DIR, BASE])
    _sasrec_data_ok = any((d / "sasrec_dataset6.pkl").exists()
                          for d in [MODELS_DIR, BASE, DATA_DIR, EVAL_DIR])
    _lgb_ok = any((d / "lightgbm_ranker_causal.pkl").exists()
                  for d in [EVAL_DIR, BASE / "src" / "utils", BASE, MODELS_DIR])
    _xgb_ok = any((d / "xgboost_ranker_model.pkl").exists()
                  for d in [EVAL_DIR, BASE / "src" / "utils", BASE, MODELS_DIR])

    status_html = (
        _status_line("Events data", _events_ok) +
        _status_line("Categories", _cats_ok) +
        _status_line("SASRec candidates", _cands_ok) +
        _status_line("SASRec model", _sasrec_model_ok) +
        _status_line("SASRec dataset", _sasrec_data_ok) +
        _status_line("LightGBM ranker", _lgb_ok) +
        _status_line("XGBoost ranker", _xgb_ok)
    )
    st.markdown(status_html, unsafe_allow_html=True)

    if not any([_events_ok, _cands_ok, _sasrec_model_ok]):
        st.markdown('<div style="font-size:0.72rem;color:#ca8a04;font-family:Space Mono,monospace;margin-top:6px;">Demo runs in metrics-only mode until data files are placed in the repo.</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>Dataset:</b> Retail Rocket<br>
    2.75M events · 235K items · 1.4M visitors<br><br>
    <b>Internship:</b> Tecorb Technologies<br>
    <b>Mentor:</b> Mr. Jai Rajput<br>
    <b>Faculty:</b> Dr. Sandeep Verma
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <b>Data Anonymisation</b><br>
    All item IDs and category IDs in this dataset are anonymised integers.
    Product names, images, and brands are not available.
    Category labels shown are inferred from numeric IDs and may not be accurate.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="future-box">
    🔬 <b>Future Work</b><br>
    Real product names + images require a named dataset such as Amazon Products 2023
    or H&M Fashion. Planned as next phase of this project.
    </div>
    """, unsafe_allow_html=True)

# ─── PAGE: PRODUCT SEARCH ────────────────────────────────────────────────────
if page == "🔍 Product Search":
    st.markdown('<div class="hero-title">Product Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Search by inferred category to explore popular items in the dataset</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ <b>Dataset Limitation:</b>
    The Retail Rocket dataset is fully anonymised — no product names, brands, or images are available.
    Category labels shown (e.g. "Electronics", "Clothing") are <b>inferred</b> from numeric category IDs
    and may not reflect the actual product type. Future work: integrate Amazon Products Dataset
    or H&M Fashion Dataset for real product search with images and names.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading dataset..."):
        interactions = load_events()
        cats = load_categories()
        candidates = load_candidates()

    query = st.text_input(
        "Search by category keyword",
        placeholder="Try: electronics, clothing, sports, home, beauty, toys, camera, kitchen...",
    )

    st.markdown('<div class="section-header">Browse by inferred category</div>', unsafe_allow_html=True)
    cat_list = [
        ("📱 Electronics", "electronics"), ("👗 Clothing", "clothing"),
        ("⚽ Sports", "sport"),            ("🏡 Home", "home"),
        ("💊 Health", "health"),           ("🎮 Toys", "toy"),
        ("📷 Cameras", "camera"),          ("💻 Computers", "computer"),
        ("🍳 Kitchen", "kitchen"),         ("👶 Baby", "baby"),
        ("🐾 Pets", "pet"),                ("🚗 Auto", "auto"),
    ]
    cols = st.columns(6)
    selected_cat = None
    for i, (label, term) in enumerate(cat_list):
        with cols[i % 6]:
            if st.button(label, key=f"cat_{term}"):
                selected_cat = term

    active_query = query if query else (selected_cat or "")

    if active_query:
        matched_cats = search_categories(active_query)
        if not matched_cats:
            st.warning(f"No categories matched '{active_query}'. Try: electronics, clothing, sports, home, beauty")
        else:
            cat_labels = " · ".join([
                f"{get_category_emoji(c)} {get_category_name(c)} (ID: {c})"
                for c in matched_cats[:5]
            ])
            st.markdown(f'<div class="info-box">Matched inferred categories: {cat_labels}</div>', unsafe_allow_html=True)

            # Find items in matched categories
            matching_items = []
            if cats:
                for item_id_str, cat_id in cats.items():
                    if str(cat_id) in matched_cats:
                        matching_items.append((int(item_id_str), str(cat_id)))

            if matching_items and interactions is not None:
                item_popularity = interactions.groupby('item_id')['user_id'].count().to_dict()
                matching_items_sorted = sorted(
                    matching_items, key=lambda x: item_popularity.get(x[0], 0), reverse=True
                )[:20]

                st.markdown(f'<div class="section-header">Top {len(matching_items_sorted)} popular items in this category (by interaction count)</div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                for i, (item_id, cat_id) in enumerate(matching_items_sorted):
                    popularity = item_popularity.get(item_id, 0)
                    emoji = get_category_emoji(cat_id)
                    cat_name = get_category_name(cat_id)
                    card = f"""
                    <div class="product-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="display:flex;align-items:center;gap:12px;">
                                <span style="font-size:1.8rem;">{emoji}</span>
                                <div>
                                    <div>
                                        <span style="color:#f9fafb;font-weight:600;">Item {item_id}</span>
                                        &nbsp;<span class="anon-tag">anonymised</span>
                                    </div>
                                    <div style="font-size:0.72rem;color:#6b7280;font-family:'Space Mono',monospace;margin-top:3px;">
                                        Category (inferred): {cat_name} &nbsp;·&nbsp; {popularity} interactions
                                    </div>
                                </div>
                            </div>
                            <span class="score-pill">{popularity} users</span>
                        </div>
                    </div>"""
                    if i % 2 == 0:
                        with col1:
                            st.markdown(card, unsafe_allow_html=True)
                    else:
                        with col2:
                            st.markdown(card, unsafe_allow_html=True)

                # Collaborative recommendations via SASRec
                if candidates is not None and matching_items_sorted:
                    top_item = matching_items_sorted[0][0]
                    st.markdown(f'<div class="section-header">Users who interacted with Item {top_item} also viewed</div>', unsafe_allow_html=True)
                    if interactions is not None:
                        item_users = interactions[interactions['item_id'] == top_item]['user_id'].tolist()
                        if item_users:
                            sample_user = item_users[0]
                            user_cands = candidates[candidates['user_id'] == sample_user].nsmallest(6, 'rank_position')
                            if not user_cands.empty:
                                rec_cols = st.columns(3)
                                for idx, (_, row) in enumerate(user_cands.iterrows()):
                                    iid = int(row['item_id'])
                                    cid = cats.get(str(iid), "498")
                                    em = get_category_emoji(cid)
                                    cn = get_category_name(cid)
                                    with rec_cols[idx % 3]:
                                        st.markdown(f"""
                                        <div class="product-card" style="text-align:center;">
                                            <div style="font-size:2rem;margin-bottom:6px;">{em}</div>
                                            <div style="font-weight:600;color:#f9fafb;font-size:0.85rem;">Item {iid}</div>
                                            <div style="font-size:0.68rem;color:#6b7280;font-family:'Space Mono',monospace;margin:4px 0;">
                                                {cn}<br>
                                                <span class="anon-tag">anonymised</span>
                                            </div>
                                            <span class="score-pill">{row['sasrec_score']:.3f}</span>
                                        </div>""", unsafe_allow_html=True)
            else:
                st.info("Category data not loaded. Place `item_properties_part1.csv` in `src/models/retail-rocket/raw/`")
    else:
        # Default: show most popular items overall
        st.markdown('<div class="section-header">Most popular items across all categories</div>', unsafe_allow_html=True)
        if interactions is not None and cats:
            top_items = interactions.groupby('item_id')['user_id'].count().nlargest(12).reset_index()
            top_items.columns = ['item_id', 'interactions']
            rec_cols = st.columns(4)
            for idx, row in top_items.iterrows():
                iid = int(row['item_id'])
                cid = cats.get(str(iid), "498")
                em = get_category_emoji(cid)
                cn = get_category_name(cid)
                with rec_cols[idx % 4]:
                    st.markdown(f"""
                    <div class="product-card" style="text-align:center;">
                        <div style="font-size:2.2rem;margin-bottom:6px;">{em}</div>
                        <div style="font-weight:600;color:#f9fafb;font-size:0.85rem;">Item {iid}</div>
                        <div style="font-size:0.68rem;color:#6b7280;font-family:'Space Mono',monospace;margin:4px 0;">
                            {cn}<br><span class="anon-tag">anonymised</span>
                        </div>
                        <span class="score-pill">{int(row['interactions'])} users</span>
                    </div>""", unsafe_allow_html=True)
        else:
            st.info("Load dataset to see popular items.")

# ─── PAGE: METRICS DASHBOARD ─────────────────────────────────────────────────
elif page == "📊 Model Metrics":
    st.markdown('<div class="hero-title">Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Evaluation results on Retail Rocket & Amazon Electronics datasets</div>', unsafe_allow_html=True)
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)
    for col, val, label in zip([col1,col2,col3,col4],
        ["2.75M","39.4K","56.2K","26.3M"],
        ["Events","Active Users","Active Items","SASRec Params"]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Retrieval Models</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":["SVD (Retail Rocket)","SVD (Amazon)","Implicit ALS","SASRec"],
        "Precision@10":["1.03%","0.60%","4.08%","—"],
        "Recall@10":["0.42%","0.63%","2.50%","4.36%"],
        "MAP@10":["0.21%","0.30%","1.11%","—"],
        "NDCG@10":["1.24%","0.81%","4.53%","0.89%"],
        "HitRate@10":["—","—","—","6.57%"],
        "Coverage":["4.99%","19.94%","22.75%","—"],
        "RMSE":["—","3.4794","—","—"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">SASRec Multi-K Evaluation</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "K":["@5","@10","@20","@500","@1000"],
        "Recall@K":["2.94%","4.36%","6.32%","22.70%","28.08%"],
        "NDCG@K":["1.08%","0.89%","0.72%","0.15%","0.10%"],
        "HitRate@K":["4.38%","6.57%","9.27%","28.50%","34.50%"],
    }), use_container_width=True, hide_index=True)
    st.markdown('<div class="info-box">Recall@500 = 22.7% is the retrieval ceiling — the maximum recall the two-stage pipeline can achieve regardless of how good the re-ranker is. HitRate@500 = 28.5% vs HitRate@10 = 6.57% shows the re-ranking opportunity.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Re-ranking Models</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="metric-card">
            <div style="margin-bottom:8px"><span class="model-badge badge-lgb">LightGBM</span></div>
            <div class="metric-value" style="font-size:1.1rem;">lambdarank</div>
            <div class="metric-label">647K train rows · 26 features · 100 trees</div>
            <div style="margin-top:10px;color:#fbbf24;font-family:'Space Mono',monospace;font-size:0.8rem;">
                #1 sasrec_score (5964) · #2 rank_position (5019)<br>
                scale_pos_weight: 859.4 · 752 positives / 647K pairs
            </div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div style="margin-bottom:8px"><span class="model-badge badge-xgb">XGBoost</span></div>
            <div class="metric-value">6.89%</div>
            <div class="metric-label">NDCG@10 · rank:ndcg objective</div>
            <div style="margin-top:10px;color:#fb923c;font-family:'Space Mono',monospace;font-size:0.8rem;">
                1.1M candidates · 36 features · 73 val users
            </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">SVD Amazon k-sweep</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "k (factors)":["50 (best)","100","150"],
        "RMSE":["3.4794","3.4805","3.4810"],
        "Precision@10":["0.60%","0.60%","0.49%"],
        "Recall@10":["0.63%","0.64%","0.59%"],
        "MAP@10":["0.30%","0.28%","0.29%"],
        "NDCG@10":["0.81%","0.76%","0.72%"],
    }), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Pipeline</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Stage":["1. Retrieval","2. Candidate Gen","3. Feature Eng","4. Re-ranking"],
        "Model":["SASRec (26.3M)","SASRec inference","LightGBMFeatureBuilder","LightGBM / XGBoost"],
        "Output":["Top-500/user","1.1M candidates","647K × 26 features","Top-10/user"],
        "Key Metric":["Recall@10: 4.36%","Recall@500: 22.7%","Positive rate: 0.12%","NDCG@10: 6.89%"],
    }), use_container_width=True, hide_index=True)

# ─── PAGE: USER RECOMMENDATIONS ──────────────────────────────────────────────
elif page == "👤 User Recommendations":
    st.markdown('<div class="hero-title">User Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare top-10 recommendations from every model — SASRec, LightGBM, XGBoost, and Implicit ALS</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ Items shown are anonymised integer IDs. Category labels are inferred from numeric category IDs
    and may not be accurate. No product names or images are available in this dataset.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading..."):
        interactions = load_events()
        cats = load_categories()
        candidates = load_candidates()

    default_user = int(interactions['user_id'].value_counts().index[0]) if interactions is not None else 147461
    col_i, col_b = st.columns([3, 1])
    with col_i:
        user_id = st.number_input("User ID", value=default_user, step=1)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Get Recommendations")

    if interactions is not None:
        history = get_user_history(int(user_id), interactions)
        if not history.empty:
            st.markdown('<div class="section-header">User interaction history</div>', unsafe_allow_html=True)
            history['category (inferred)'] = history['item_id'].apply(
                lambda x: get_category_name(cats.get(str(int(x)), "?"))
            )
            st.dataframe(history[['item_id','category (inferred)','weight','datetime']].rename(columns={
                'item_id':'Item ID','weight':'Interaction Weight','datetime':'Last Seen'
            }), use_container_width=True, hide_index=True)
        else:
            # Only warn if there are also no candidates — if recommendations exist the
            # user is valid, they just fall below the ≥5 interaction filter used for
            # the events dataset but were included in SASRec training (≥2 interactions).
            has_candidates = candidates is not None and not candidates[candidates['user_id'] == int(user_id)].empty
            if not has_candidates:
                st.warning(f"User {user_id} not found. Try one of the sample IDs below.")
            else:
                st.markdown(
                    '<div class="info-box">Interaction history not available for this user in the '
                    'events dataset (requires ≥5 interactions), but SASRec recommendations are available '
                    'from training data (≥2 interactions).</div>',
                    unsafe_allow_html=True
                )

    if candidates is not None:
        user_cands_500 = candidates[candidates['user_id'] == int(user_id)].nsmallest(500, 'rank_position')
        user_cands_10  = user_cands_500.head(10)
        if not user_cands_10.empty:
            st.markdown('<div class="section-header">Recommendations — compare across models</div>', unsafe_allow_html=True)

            # helper: rerank candidates using a gradient booster pkl dict
            def _rerank(cands, ranker_obj, score_col):
                if ranker_obj is None:
                    return None, "Model file not found"
                try:
                    import xgboost as xgb
                    import lightgbm  # noqa — imported to confirm lgb available too
                    booster = ranker_obj['model']
                    feature_cols = ranker_obj['feature_cols']
                    feat_df = cands.copy()
                    for mc in [c for c in feature_cols if c not in feat_df.columns]:
                        feat_df[mc] = 0.0
                    feat_arr = feat_df[feature_cols].values.astype(float)
                    # XGBoost Booster requires DMatrix; LightGBM Booster takes numpy directly
                    model_type = type(booster).__module__
                    if 'xgboost' in model_type:
                        dmat = xgb.DMatrix(feat_arr, feature_names=feature_cols)
                        scores = booster.predict(dmat)
                    else:
                        scores = booster.predict(feat_arr)
                    out = cands.copy()
                    out[score_col] = scores
                    return out.nlargest(10, score_col).reset_index(drop=True), None
                except Exception as e:
                    return None, str(e)

            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🟢 SASRec  (retrieval)",
                "🟡 LightGBM  (re-ranked)",
                "🟠 XGBoost  (re-ranked)",
                "🔵 Implicit ALS  (classical)",
                "🟣 PureSVD  (baseline)",
            ])

            # ── TAB 1: SASRec ─────────────────────────────────────────────
            with tab1:
                st.markdown("""
                <div class="info-box">
                <b>SASRec</b> — Self-Attentive Sequential Recommendation (26.3M params, 10 epochs)<br>
                Scores items by dot-product with the user's last-position Transformer hidden state.<br>
                <b>Recall@10:</b> 4.36% &nbsp;·&nbsp; <b>NDCG@10:</b> 0.89% &nbsp;·&nbsp;
                <b>HitRate@10:</b> 6.57% &nbsp;·&nbsp; <b>Recall@500:</b> 22.70%
                </div>""", unsafe_allow_html=True)
                for _, row in user_cands_10.iterrows():
                    iid = int(row['item_id'])
                    cid = cats.get(str(iid), "498")
                    st.markdown(product_card(int(row['rank_position']), iid,
                                             float(row['sasrec_score']), cid, "sasrec score"), unsafe_allow_html=True)

            # ── TAB 2: LightGBM ───────────────────────────────────────────
            with tab2:
                st.markdown("""
                <div class="info-box">
                <b>LightGBM lambdarank</b> — re-ranks the SASRec top-500 using 26 engineered features.<br>
                Top feature: sasrec_score (gain 5964) — the ranker refines, not overrides, SASRec.<br>
                <b>100 trees</b> &nbsp;·&nbsp; <b>26 features</b> &nbsp;·&nbsp;
                <b>scale_pos_weight: 859.4</b> &nbsp;·&nbsp; <b>Objective:</b> lambdarank
                </div>""", unsafe_allow_html=True)
                lgb_obj = load_lgb_ranker()
                reranked_lgb, err = _rerank(user_cands_500, lgb_obj, 'lgb_score')
                if reranked_lgb is not None:
                    for i, row in reranked_lgb.iterrows():
                        iid = int(row['item_id'])
                        cid = cats.get(str(iid), "498")
                        st.markdown(product_card(i+1, iid, float(row['lgb_score']),
                                                 cid, "lgb score"), unsafe_allow_html=True)
                else:
                    st.info(f"LightGBM not available: {err}. Place `lightgbm_ranker_causal.pkl` in app dir.")

            # ── TAB 3: XGBoost ────────────────────────────────────────────
            with tab3:
                st.markdown("""
                <div class="info-box">
                <b>XGBoost rank:ndcg</b> — re-ranks the SASRec top-500 using 36 engineered features.<br>
                Best ranking metric in the project: <b>NDCG@10: 6.89%</b> (evaluated on 73 val users).<br>
                <b>36 features</b> &nbsp;·&nbsp; <b>Objective:</b> rank:ndcg
                </div>""", unsafe_allow_html=True)
                xgb_obj = load_xgb_ranker()
                reranked_xgb, err = _rerank(user_cands_500, xgb_obj, 'xgb_score')
                if reranked_xgb is not None:
                    for i, row in reranked_xgb.iterrows():
                        iid = int(row['item_id'])
                        cid = cats.get(str(iid), "498")
                        st.markdown(product_card(i+1, iid, float(row['xgb_score']),
                                                 cid, "xgb score"), unsafe_allow_html=True)
                else:
                    st.info(f"XGBoost not available: {err}. Place `xgboost_ranker_model.pkl` in app dir.")

            # ── TAB 4: ALS ────────────────────────────────────────────────
            with tab4:
                st.markdown("""
                <div class="info-box">
                <b>Implicit ALS</b> — matrix factorisation with confidence-weighted implicit feedback.<br>
                Best classical model: <b>Precision@10: 4.08%</b> &nbsp;·&nbsp;
                <b>NDCG@10: 4.53%</b> &nbsp;·&nbsp; <b>Coverage: 22.75%</b><br>
                Model trained once and cached (256 factors, alpha=40, 30 iterations).
                First load takes ~30s; subsequent users are instant.
                </div>""", unsafe_allow_html=True)
                with st.spinner("Loading ALS model (trains once, then cached)..."):
                    als_recs, als_err = get_als_recommendations(int(user_id), n=10)
                if als_recs:
                    for rank, (iid, score) in enumerate(als_recs, 1):
                        cid = cats.get(str(iid), "498")
                        st.markdown(product_card(rank, iid, score, cid, "als score"), unsafe_allow_html=True)
                else:
                    st.info(f"ALS: {als_err}")

            with tab5:
                st.markdown("""
                <div class="info-box">
                <b>PureSVD</b> — Singular Value Decomposition, 128 factors.<br>
                Plain matrix approximation — does not handle implicit feedback correctly.<br>
                <b>Precision@10: 1.03%</b> &nbsp;·&nbsp; <b>NDCG@10: 1.24%</b> &nbsp;·&nbsp;
                <b>Coverage: 4.99%</b><br>
                Compare with ALS tab to see directly why confidence-weighted MF beats plain SVD
                on implicit data.
                </div>""", unsafe_allow_html=True)
                with st.spinner("Loading SVD model (trains once, then cached)..."):
                    svd_recs, svd_err = get_svd_recommendations(int(user_id), n=10)
                if svd_recs:
                    for rank, (iid, score) in enumerate(svd_recs, 1):
                        cid = cats.get(str(iid), "498")
                        st.markdown(product_card(rank, iid, score, cid, "svd score"), unsafe_allow_html=True)
                else:
                    st.info(f"SVD: {svd_err}")

            # ── Model comparison note ──────────────────────────────────────
            st.markdown('<div class="section-header">How to read this comparison</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({
                "Model":["SASRec","LightGBM re-ranked","XGBoost re-ranked","Implicit ALS","PureSVD"],
                "Type":["Sequential Transformer","Learning-to-Rank","Learning-to-Rank","Matrix Factorisation","Matrix Factorisation"],
                "Precision@10":["—","—","—","4.08%","1.03%"],
                "Recall@10":["4.36%","—","—","2.50%","0.42%"],
                "NDCG@10":["0.89%","—","6.89%","4.53%","1.24%"],
                "HitRate@10":["6.57%","—","—","—","—"],
                "Coverage":["—","—","—","22.75%","4.99%"],
                "Key insight":["Sequential intent","Refines SASRec order","Best NDCG","CF for implicit","Fails on sparse data"],
            }), use_container_width=True, hide_index=True)

        else:
            st.info(f"No SASRec candidates for user {user_id}. Sample IDs:")
            if candidates is not None:
                st.code(", ".join(map(str, candidates['user_id'].unique()[:10])))

# ─── PAGE: I2I ───────────────────────────────────────────────────────────────
elif page == "🔗 Item Similarity (I2I)":
    st.markdown('<div class="hero-title">Item-to-Item Similarity</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Find similar items using SASRec embedding cosine similarity</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
    ⚠️ Item IDs are anonymised integers. Similarity is computed from SASRec's learned embeddings —
    items appearing in similar sequential contexts get similar representations.
    Category labels are inferred and may not be accurate.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading..."):
        cats = load_categories()
        interactions = load_events()

    item_id = st.number_input("Item ID", value=49967, step=1)
    st.button("Find Similar Items")

    if interactions is not None:
        item_stats = interactions[interactions['item_id'] == int(item_id)]
        if not item_stats.empty:
            cid = cats.get(str(int(item_id)), "498")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Interactions", len(item_stats))
            with c2: st.metric("Unique Users", item_stats['user_id'].nunique())
            with c3: st.metric("Category (inferred)", get_category_name(cid))

    known_results = {49967: [(254678, 0.9611), (360017, 0.9574), (197535, 0.9566)]}
    st.markdown('<div class="section-header">Similar items (SASRec cosine similarity)</div>', unsafe_allow_html=True)

    sasrec_model = load_sasrec_model()
    sasrec_dataset = load_sasrec_dataset()

    if int(item_id) in known_results:
        st.markdown(f'<div class="info-box">Cached results from training run for Item {item_id}.</div>', unsafe_allow_html=True)
        rec_cols = st.columns(3)
        for idx, (sim_item, score) in enumerate(known_results[int(item_id)]):
            cid = cats.get(str(sim_item), "498")
            em = get_category_emoji(cid)
            cn = get_category_name(cid)
            with rec_cols[idx]:
                st.markdown(f"""
                <div class="product-card" style="text-align:center;">
                    <div style="font-size:2.2rem;margin-bottom:6px;">{em}</div>
                    <div style="font-weight:600;color:#f9fafb;">Item {sim_item}</div>
                    <div style="font-size:0.7rem;color:#6b7280;font-family:'Space Mono',monospace;margin:4px 0;">
                        {cn}<br><span class="anon-tag">anonymised</span>
                    </div>
                    <span class="score-pill">cosine: {score:.4f}</span>
                </div>""", unsafe_allow_html=True)
    elif sasrec_model is not None and sasrec_dataset is not None:
        try:
            with st.spinner("Computing..."):
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                recos = sasrec_model.recommend_to_items(
                    target_items=[int(item_id)], dataset=sasrec_dataset, k=6, filter_itself=True
                )
            rec_cols = st.columns(3)
            for idx, (_, row) in enumerate(recos.head(6).iterrows()):
                sim_item = int(row['item_id'])
                cid = cats.get(str(sim_item), "498")
                em = get_category_emoji(cid)
                cn = get_category_name(cid)
                with rec_cols[idx % 3]:
                    st.markdown(f"""
                    <div class="product-card" style="text-align:center;">
                        <div style="font-size:2.2rem;margin-bottom:6px;">{em}</div>
                        <div style="font-weight:600;color:#f9fafb;">Item {sim_item}</div>
                        <div style="font-size:0.7rem;color:#6b7280;font-family:'Space Mono',monospace;margin:4px 0;">
                            {cn}<br><span class="anon-tag">anonymised</span>
                        </div>
                        <span class="score-pill">score: {float(row['score']):.4f}</span>
                    </div>""", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"I2I error: {e}")
    else:
        st.info("Place `sasrec_gbce_model6.pkl` + `sasrec_dataset6.pkl` in app dir for live I2I.")

    st.markdown('<div class="section-header">How I2I works</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    SASRec learns 256-dimensional item embeddings during Transformer training.
    Items appearing in similar sequential browsing contexts get similar vector representations.
    Cosine similarity between these vectors identifies related products without requiring
    explicit product metadata — entirely learned from behavioural patterns.
    </div>""", unsafe_allow_html=True)

# ─── PAGE: MODEL COMPARISON ───────────────────────────────────────────────────
elif page == "⚖️ Model Comparison":
    st.markdown('<div class="hero-title">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Head-to-head performance across all models and metrics</div>', unsafe_allow_html=True)
    st.markdown("")

    try:
        import plotly.graph_objects as go
        models = ["SVD (RR)", "SVD (Amazon)", "ALS", "SASRec", "XGBoost"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='NDCG@10 (%)', x=models, y=[1.24, 0.81, 4.53, 0.89, 6.89], marker_color='#e94560', opacity=0.9))
        fig.add_trace(go.Bar(name='Precision@10 (%)', x=models[:3], y=[1.03, 0.60, 4.08], marker_color='#53d8fb', opacity=0.9))
        fig.add_trace(go.Bar(name='Recall@10 (%)', x=models[:4], y=[0.42, 0.63, 2.50, 4.36], marker_color='#4ade80', opacity=0.9))
        fig.update_layout(
            barmode='group', plot_bgcolor='#0a0a0f', paper_bgcolor='#0a0a0f',
            font=dict(color='#9ca3af', family='DM Sans'),
            title=dict(text='Evaluation Metrics Comparison', font=dict(color='#f9fafb', size=16)),
            legend=dict(bgcolor='#111827', bordercolor='#1f2937', borderwidth=1),
            xaxis=dict(gridcolor='#1f2937'), yaxis=dict(gridcolor='#1f2937', title='Score (%)'),
            margin=dict(l=40, r=40, t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Install plotly for charts: `pip install plotly`")

    st.markdown('<div class="section-header">Full comparison table</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":["SVD (Retail Rocket)","SVD (Amazon)","Implicit ALS ⭐","SASRec ⭐","XGBoost Ranker"],
        "Type":["Matrix Factorisation","Matrix Factorisation","Collaborative Filtering","Sequential Transformer","Learning-to-Rank"],
        "Precision@10":["1.03%","0.60%","4.08%","—","—"],
        "Recall@10":["0.42%","0.63%","2.50%","4.36%","—"],
        "MAP@10":["0.21%","0.30%","1.11%","—","—"],
        "NDCG@10":["1.24%","0.81%","4.53%","0.89%","6.89%"],
        "HitRate@10":["—","—","—","6.57%","—"],
        "Recall@500":["—","—","—","22.70%","—"],
        "Coverage":["4.99%","19.94%","22.75%","—","—"],
    }), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="info-box"><b style="color:#4ade80;">ALS wins Precision@10 (4.08%)</b><br>Best top-N accuracy with 22.75% catalogue coverage.</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><b style="color:#e94560;">SASRec wins Recall@10 (4.36%)</b><br>Sequential patterns capture temporal user intent. Recall@500 = 22.7% — ideal retrieval stage.</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="info-box"><b style="color:#fb923c;">XGBoost improves NDCG@10 (6.89%)</b><br>Re-ranking 500 candidates with 36 engineered features shows measurable ranking gain.</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><b style="color:#c084fc;">SVD underperforms on sparse data</b><br>99.84% sparsity limits decomposition quality. Better on Amazon (explicit ratings).</div>', unsafe_allow_html=True)