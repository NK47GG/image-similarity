# app.py
"""
Streamlit App — Visually Similar Product Recommendation
Dataset  : Fashion Product Images (Shirts only)
Model    : Convolutional Autoencoder (encoder_weights.h5)
Database : Embeddings dari test split (database.pkl)

Run : streamlit run app.py
"""

import os
import numpy as np
import streamlit as st
from PIL import Image
import joblib

import sys
import numpy.core
sys.modules['numpy._core'] = numpy.core

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape,
    Conv2DTranspose, UpSampling2D,
)
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
IMG_SIZE        = (128, 128)
IMG_SHAPE       = (128, 128, 3)
LATENT_DIM      = 256
TOP_K           = 5
ENCODER_WEIGHTS = "encoder_only_weights.weights.h5"
DATABASE_PATH   = "database.pkl"


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "ShirtFinder — Visual Search",
    page_icon  = "🔍",
    layout     = "wide",
)

st.markdown("""
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global reset ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Hide Streamlit chrome ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── App background ──────────────────────────────────────── */
.stApp {
    background: #F7F4EF;
}

/* ── Hero header ─────────────────────────────────────────── */
.hero-wrapper {
    background: #1A1A1A;
    border-radius: 16px;
    padding: 48px 56px 40px;
    margin-bottom: 32px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 24px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F7F4EF;
    line-height: 1.1;
    margin: 0 0 8px;
}
.hero-title span { color: #C8A96E; }
.hero-subtitle {
    font-size: 0.95rem;
    color: #888;
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.02em;
}
.hero-badge {
    background: #C8A96E22;
    border: 1px solid #C8A96E55;
    color: #C8A96E;
    font-size: 0.78rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 100px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Section label ───────────────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}

/* ── Upload card ─────────────────────────────────────────── */
.upload-card {
    background: #FFFFFF;
    border: 1.5px dashed #D4C9B8;
    border-radius: 14px;
    padding: 32px;
    text-align: center;
    transition: border-color 0.2s;
}

/* ── Query image card ────────────────────────────────────── */
.query-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.query-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}

/* ── Result card ─────────────────────────────────────────── */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.10);
}
.rank-badge {
    display: inline-block;
    background: #1A1A1A;
    color: #F7F4EF;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 100px;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}
.score-text {
    font-size: 0.8rem;
    color: #888;
    margin-top: 8px;
}
.score-value {
    font-weight: 600;
    color: #C8A96E;
}
.product-id {
    font-size: 0.75rem;
    color: #AAAAAA;
    margin-top: 2px;
}

/* ── Score bar ───────────────────────────────────────────── */
.score-bar-bg {
    background: #EEEBE4;
    border-radius: 100px;
    height: 4px;
    margin-top: 8px;
    overflow: hidden;
}
.score-bar-fill {
    background: linear-gradient(90deg, #C8A96E, #E8C98E);
    height: 4px;
    border-radius: 100px;
}

/* ── Divider ─────────────────────────────────────────────── */
.custom-divider {
    border: none;
    border-top: 1px solid #E0D9CE;
    margin: 24px 0;
}

/* ── Stats row ───────────────────────────────────────────── */
.stat-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #FFFFFF;
    border: 1px solid #E0D9CE;
    border-radius: 100px;
    padding: 6px 14px;
    font-size: 0.82rem;
    color: #444;
    font-weight: 400;
}
.stat-chip strong { color: #1A1A1A; font-weight: 600; }

/* ── Alert box ───────────────────────────────────────────── */
.error-box {
    background: #FFF2F2;
    border: 1px solid #FFBBBB;
    border-radius: 12px;
    padding: 20px 24px;
    color: #CC0000;
    font-size: 0.9rem;
}

/* ── Radio button styling ────────────────────────────────── */
div[data-testid="stRadio"] label {
    font-size: 0.88rem !important;
}

/* ── Result section heading ──────────────────────────────── */
.results-heading {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #1A1A1A;
    margin: 8px 0 20px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION  (identik dengan arsitektur training)
# ─────────────────────────────────────────────────────────────
def build_autoencoder(input_shape=IMG_SHAPE, latent_dim=LATENT_DIM):
    inputs = Input(shape=input_shape, name="encoder_input")

    x = Conv2D(32,  (3, 3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64,  (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    conv_shape = x.shape[1:]
    x       = Flatten()(x)
    encoded = Dense(latent_dim, activation="relu", name="latent_space")(x)

    x = Dense(conv_shape[0] * conv_shape[1] * conv_shape[2], activation="relu")(encoded)
    x = Reshape(conv_shape)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64,  (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same",
                              name="decoder_output")(x)

    autoencoder = Model(inputs, decoded, name="autoencoder")
    encoder     = Model(inputs, encoded, name="encoder")
    return autoencoder, encoder


# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat encoder model…")
def load_encoder():
    autoencoder, encoder = build_autoencoder()
    encoder.load_weights(ENCODER_WEIGHTS)
    return encoder


@st.cache_data(show_spinner="Memuat embedding database…")
def load_database():
    db         = joblib.load(DATABASE_PATH)
    embeddings = db["embeddings"]
    metadata   = db["metadata"].reset_index(drop=True)
    return embeddings, metadata


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def preprocess_image(pil_image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def find_similar(pil_image, encoder_model, db_embeddings, db_df):
    query_emb = encoder_model.predict(preprocess_image(pil_image), verbose=0)
    scores    = cosine_similarity(query_emb, db_embeddings)[0]
    indices   = np.argsort(scores)[::-1][:TOP_K]
    results   = db_df.iloc[indices].copy().reset_index(drop=True)
    results["similarity_score"] = scores[indices]
    return results


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div>
        <h1 class="hero-title">Shirt<span>Finder</span></h1>
        <p class="hero-subtitle">
            Visual search berbasis Convolutional Autoencoder + Cosine Similarity
        </p>
    </div>
    <div>
        <span class="hero-badge">🧵 Shirts Only · Top-5 Results</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CEK ARTEFAK
# ─────────────────────────────────────────────────────────────
missing = [f for f in [ENCODER_WEIGHTS, DATABASE_PATH] if not os.path.exists(f)]
if missing:
    st.markdown(f"""
    <div class="error-box">
        <strong>⚠️ File tidak ditemukan:</strong> {', '.join(f'<code>{f}</code>' for f in missing)}<br><br>
        Jalankan <code>autoencoder_training_revised.ipynb</code> terlebih dahulu untuk
        men-generate <code>encoder_weights.h5</code> dan <code>database.pkl</code>.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

encoder_model        = load_encoder()
db_embeddings, db_df = load_database()


# ─────────────────────────────────────────────────────────────
# STATS CHIPS
# ─────────────────────────────────────────────────────────────
col_s1, col_s2, col_s3, _ = st.columns([1, 1, 1, 4])
with col_s1:
    st.markdown(f'<div class="stat-chip">🗂 Database <strong>{len(db_df):,} items</strong></div>',
                unsafe_allow_html=True)
with col_s2:
    st.markdown(f'<div class="stat-chip">🧠 Latent dim <strong>{LATENT_DIM}</strong></div>',
                unsafe_allow_html=True)
with col_s3:
    st.markdown(f'<div class="stat-chip">🔎 Top-K <strong>{TOP_K}</strong></div>',
                unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">📥 Input Gambar Query</p>', unsafe_allow_html=True)

    input_method = st.radio(
        "Pilih metode input:",
        options=["📁 Upload dari galeri", "📷 Foto dengan kamera"],
        horizontal=False,
        label_visibility="collapsed",
    )

    query_image = None

    if input_method == "📁 Upload dari galeri":
        uploaded_file = st.file_uploader(
            "Pilih gambar produk (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="visible",
        )
        if uploaded_file:
            query_image = Image.open(uploaded_file)

    else:
        camera_photo = st.camera_input("Arahkan kamera ke produk")
        if camera_photo:
            query_image = Image.open(camera_photo)

    if query_image:
        st.markdown('<div class="query-card">', unsafe_allow_html=True)
        st.markdown('<p class="query-label">Preview Query</p>', unsafe_allow_html=True)
        st.image(query_image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RESULT SECTION
# ─────────────────────────────────────────────────────────────
with right_col:
    if query_image:
        st.markdown('<p class="section-label">🔍 Hasil Rekomendasi</p>', unsafe_allow_html=True)

        with st.spinner("Mencari produk serupa…"):
            results = find_similar(query_image, encoder_model, db_embeddings, db_df)

        st.markdown(
            f'<p class="results-heading">Top {TOP_K} Produk Paling Mirip</p>',
            unsafe_allow_html=True,
        )

        cols = st.columns(TOP_K, gap="small")
        for rank, (col, (_, row)) in enumerate(zip(cols, results.iterrows()), start=1):
            with col:
                score_pct = int(row["similarity_score"] * 100)
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<span class="rank-badge">#{rank}</span>', unsafe_allow_html=True)

                img_path = row["filename"]
                if os.path.exists(img_path):
                    st.image(Image.open(img_path), use_column_width=True)
                else:
                    st.warning("File\ntidak\nditemukan")

                st.markdown(f"""
                    <p class="score-text">
                        Similarity&nbsp;
                        <span class="score-value">{row['similarity_score']:.4f}</span>
                    </p>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{score_pct}%"></div>
                    </div>
                    <p class="product-id">ID: {row['id']}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="
            height: 340px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #FFFFFF;
            border: 1.5px dashed #D4C9B8;
            border-radius: 14px;
            color: #AAAAAA;
            gap: 12px;
        ">
            <span style="font-size: 2.5rem;">👕</span>
            <p style="font-family:'DM Sans',sans-serif; font-size:0.9rem; margin:0;">
                Upload gambar atau ambil foto untuk memulai pencarian
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; font-size:0.78rem; color:#AAAAAA; font-family:'DM Sans',sans-serif;">
    ShirtFinder · Convolutional Autoencoder + Cosine Similarity ·
    Dataset: <em>paramaggarwal/fashion-product-images-dataset</em>
</p>
""", unsafe_allow_html=True)
