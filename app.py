import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShirtFinder — Visual Search",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* Root Variables */
:root {
    --primary: #1a1a2e;
    --accent: #e94560;
    --accent2: #0f3460;
    --gold: #f5a623;
    --surface: #16213e;
    --card-bg: #0f3460;
    --text-muted: #8892a4;
    --success: #00c9a7;
    --bg: #0d0d1a;
}

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: #e8eaf0 !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem; max-width: 1400px; }

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 50%, #1a1a2e 100%);
    border-bottom: 1px solid rgba(233,69,96,0.3);
    padding: 2.5rem 2rem 2rem;
    margin: -1rem -2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(233,69,96,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #ffffff 30%, #e94560 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem;
    line-height: 1.1;
}
.hero-subtitle {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(233,69,96,0.15);
    border: 1px solid rgba(233,69,96,0.4);
    color: #e94560;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
}

/* ── Upload Panel ── */
.upload-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
    display: block;
}

/* ── Query Preview ── */
.query-container {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
    position: relative;
}
.query-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    padding: 0.8rem 1rem 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ── Results Section ── */
.section-title {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #c8d0de;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0 0 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent);
}

/* ── Result Card ── */
.result-card {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    overflow: hidden;
    transition: all 0.3s ease;
    position: relative;
    height: 100%;
}
.result-card:hover {
    border-color: rgba(233,69,96,0.5);
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px rgba(233,69,96,0.1);
}
.rank-badge {
    position: absolute;
    top: 12px;
    left: 12px;
    z-index: 10;
    background: rgba(13,13,26,0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.12);
    color: #fff;
    font-family: 'Sora', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 3px 8px;
    border-radius: 8px;
}

/* ── Match Score Bar ── */
.match-bar-container {
    padding: 0.9rem 1rem 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
.match-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}
.match-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
}
.match-score {
    font-family: 'Sora', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
}
.bar-track {
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 99px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s ease;
}
.card-meta {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 4px;
}

/* ── Stats Row ── */
.stat-card {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-number {
    font-family: 'Sora', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--text-muted);
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ── Streamlit elements override ── */
.stFileUploader > div {
    background: rgba(15,52,96,0.3) !important;
    border: 1.5px dashed rgba(233,69,96,0.4) !important;
    border-radius: 12px !important;
}
.stFileUploader > div:hover {
    border-color: rgba(233,69,96,0.8) !important;
    background: rgba(233,69,96,0.05) !important;
}
.stButton button {
    background: linear-gradient(135deg, #e94560, #c73652) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(233,69,96,0.35) !important;
}
[data-testid="stCameraInput"] {
    border-radius: 12px !important;
    overflow: hidden !important;
}
div[data-testid="stSelectbox"] > div {
    background: rgba(22,33,62,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# HELPERS — replace these with your actual model & dataset logic
# ────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """Load your embedding model here."""
    try:
        # Example: from sentence_transformers import SentenceTransformer
        # return SentenceTransformer("clip-ViT-B-32")
        return None  # placeholder
    except Exception:
        return None


@st.cache_data
def load_dataset():
    """Load your image dataset + embeddings here.
    Should return a list of dicts: [{id, path, embedding, category, name}, ...]
    """
    # placeholder — returns empty list
    return []


def search_similar(query_image: Image.Image, dataset: list, top_k: int = 10):
    """
    Run similarity search.
    Replace with your actual embedding + cosine similarity logic.
    Returns list of {id, path, score, rank, category}.
    """
    # ── Example placeholder results (remove when wiring real model) ──────────
    placeholder = [
        {"id": 5129,  "rank": 1,  "score": 0.543, "name": "Floral Cotton Shirt",   "category": "Casual"},
        {"id": 14772, "rank": 2,  "score": 0.543, "name": "Slim Fit Oxford Shirt",  "category": "Formal"},
        {"id": 9659,  "rank": 3,  "score": 0.474, "name": "Striped Dress Shirt",    "category": "Formal"},
        {"id": 8940,  "rank": 4,  "score": 0.468, "name": "Mandarin Collar Set",    "category": "Smart"},
        {"id": 7145,  "rank": 5,  "score": 0.438, "name": "Plaid Button-Down",      "category": "Casual"},
        {"id": 3301,  "rank": 6,  "score": 0.421, "name": "Linen Summer Shirt",     "category": "Casual"},
        {"id": 2810,  "rank": 7,  "score": 0.408, "name": "Classic White Formal",   "category": "Formal"},
        {"id": 6644,  "rank": 8,  "score": 0.395, "name": "Denim Western Shirt",    "category": "Casual"},
        {"id": 11020, "rank": 9,  "score": 0.381, "name": "Batik Heritage Shirt",   "category": "Traditional"},
        {"id": 4477,  "rank": 10, "score": 0.364, "name": "Technical Sports Shirt", "category": "Sport"},
    ]
    return placeholder[:top_k]


# ────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ────────────────────────────────────────────────────────────────────────────

# ── Hero Header ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">✦ AI-Powered</div>
    <div class="hero-title">ShirtFinder</div>
    <div class="hero-subtitle">Upload a shirt image — find visually similar styles instantly</div>
</div>
""", unsafe_allow_html=True)

# ── Main columns ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2.6], gap="large")

with col_left:
    # Input mode tabs
    mode = st.radio(
        "Input mode",
        ["📁 Gallery", "📷 Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded_file = None
    if "Gallery" in mode:
        st.markdown('<span class="upload-label">📁 Upload Shirt Image</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload shirt",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
        )
    else:
        st.markdown('<span class="upload-label">📷 Take a Photo</span>', unsafe_allow_html=True)
        uploaded_file = st.camera_input("Camera", label_visibility="collapsed")

    # Settings expander
    with st.expander("⚙️  Search settings", expanded=False):
        top_k = st.slider("Results to show", min_value=3, max_value=20, value=10, step=1)
        min_score = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        category_filter = st.multiselect(
            "Filter by category",
            ["Casual", "Formal", "Smart", "Traditional", "Sport"],
            default=[],
            placeholder="All categories",
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Query preview
    if uploaded_file:
        query_img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="query-label">🔍 QUERY IMAGE</div>', unsafe_allow_html=True)
        st.image(query_img, use_container_width=True)

        run_search = st.button("Search Similar Shirts ↗", use_container_width=True)
    else:
        st.markdown("""
        <div style="
            border: 1.5px dashed rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 2.5rem 1rem;
            text-align: center;
            color: #556070;
            margin-top: 0.5rem;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👔</div>
            <div style="font-size: 0.85rem; font-weight: 500;">Upload or capture a shirt<br>to start searching</div>
        </div>
        """, unsafe_allow_html=True)
        run_search = False


# ── Right: Results ────────────────────────────────────────────────────────────
with col_right:
    if uploaded_file and run_search:
        with st.spinner("🔍 Searching through the catalog…"):
            dataset = load_dataset()
            results = search_similar(query_img, dataset, top_k=top_k)

        # Filter by category if set
        if category_filter:
            results = [r for r in results if r.get("category") in category_filter]

        # Filter by min score
        results = [r for r in results if r["score"] >= min_score]

        # ── Stats bar ────────────────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        for col, val, lbl, color in [
            (s1, len(results), "Matches found", "#e94560"),
            (s2, f"{results[0]['score']:.3f}" if results else "—", "Best score", "#00c9a7"),
            (s3, f"{np.mean([r['score'] for r in results]):.3f}" if results else "—", "Avg score", "#f5a623"),
            (s4, len(set(r.get('category','') for r in results)), "Categories", "#a78bfa"),
        ]:
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-number" style="color:{color}">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">Top {len(results)} Results</div>', unsafe_allow_html=True)

        # ── Grid of result cards ─────────────────────────────────────────────
        cols_per_row = 3
        for row_start in range(0, len(results), cols_per_row):
            row_results = results[row_start:row_start + cols_per_row]
            grid_cols = st.columns(cols_per_row, gap="medium")

            for gc, res in zip(grid_cols, row_results):
                with gc:
                    score = res["score"]
                    pct = int(score * 100)

                    # Color logic for score
                    if score >= 0.5:
                        bar_color = "#00c9a7"
                        score_color = "#00c9a7"
                    elif score >= 0.4:
                        bar_color = "#f5a623"
                        score_color = "#f5a623"
                    else:
                        bar_color = "#e94560"
                        score_color = "#e94560"

                    cat = res.get("category", "Unknown")
                    name = res.get("name", f"Shirt #{res['id']}")
                    rank = res["rank"]

                    # Category badge color
                    cat_colors = {
                        "Casual": ("#1a3a5c", "#4fc3f7"),
                        "Formal": ("#1a2e1a", "#81c784"),
                        "Smart": ("#2e1a3a", "#ce93d8"),
                        "Traditional": ("#3a2a0a", "#ffb74d"),
                        "Sport": ("#1a2e3a", "#4dd0e1"),
                    }
                    cat_bg, cat_txt = cat_colors.get(cat, ("#2a2a3a", "#aaaacc"))

                    # Try to load actual image, else show placeholder
                    img_path = res.get("path", "")
                    img_html = ""
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    else:
                        # Placeholder image area
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #16213e, #0f3460);
                            height: 220px;
                            border-radius: 12px 12px 0 0;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 3rem;
                            position: relative;
                        ">
                            <div style="
                                position: absolute; top: 10px; left: 10px;
                                background: rgba(13,13,26,0.85);
                                border: 1px solid rgba(255,255,255,0.12);
                                color: #fff;
                                font-family: 'Sora', sans-serif;
                                font-size: 0.62rem;
                                font-weight: 700;
                                letter-spacing: 1px;
                                padding: 3px 8px;
                                border-radius: 8px;
                            ">#{rank}</div>
                            <div style="
                                position: absolute; top: 10px; right: 10px;
                                background: {cat_bg};
                                color: {cat_txt};
                                font-size: 0.62rem;
                                font-weight: 600;
                                letter-spacing: 0.5px;
                                padding: 3px 8px;
                                border-radius: 6px;
                            ">{cat}</div>
                            👔
                        </div>
                        """, unsafe_allow_html=True)

                    # Score bar + info
                    st.markdown(f"""
                    <div style="
                        background: #16213e;
                        border: 1px solid rgba(255,255,255,0.07);
                        border-radius: 0 0 12px 12px;
                        padding: 0.8rem 1rem 0.9rem;
                        margin-top: -4px;
                    ">
                        <div style="
                            font-family: 'Sora', sans-serif;
                            font-size: 0.82rem;
                            font-weight: 600;
                            color: #e8eaf0;
                            margin-bottom: 8px;
                            white-space: nowrap;
                            overflow: hidden;
                            text-overflow: ellipsis;
                        ">{name}</div>
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px;">
                            <span style="font-size:0.65rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#556070;">SIMILARITY</span>
                            <span style="font-family:'Sora',sans-serif; font-size:1rem; font-weight:700; color:{score_color};">{score:.3f}</span>
                        </div>
                        <div style="height:4px; background:rgba(255,255,255,0.07); border-radius:99px; overflow:hidden;">
                            <div style="height:100%; width:{pct}%; background:{bar_color}; border-radius:99px;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between; margin-top:6px;">
                            <span style="font-size:0.65rem; color:#556070;">ID: {res['id']}</span>
                            <span style="font-size:0.65rem; color:#556070;">{pct}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

    elif not uploaded_file:
        # Empty state
        st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            opacity: 0.35;
            text-align: center;
        ">
            <div style="font-size: 5rem; margin-bottom: 1rem;">🔍</div>
            <div style="font-family: 'Sora', sans-serif; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">No query yet</div>
            <div style="font-size: 0.85rem; color: #8892a4;">Upload a shirt image on the left<br>to find visually similar styles</div>
        </div>
        """, unsafe_allow_html=True)

    elif uploaded_file and not run_search:
        st.markdown("""
        <div style="
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 400px;
            opacity: 0.5;
            text-align: center;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">👔</div>
            <div style="font-family: 'Sora', sans-serif; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">Image ready</div>
            <div style="font-size: 0.85rem; color: #8892a4;">Click "Search Similar Shirts" to begin</div>
        </div>
        """, unsafe_allow_html=True)
