# app.py
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import joblib

# --- HACK: Kompatibilitas versi Numpy agar database.pkl terbaca ---
try:
    import numpy.core
    sys.modules['numpy._core'] = numpy.core
except ImportError:
    pass

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
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

.stApp {
    background: #F7F4EF;
}

/* FIX: Warna font label agar hitam pekat dan kontras */
.section-label, 
div[data-testid="stWidgetLabel"] label p, 
div[data-testid="stRadio"] label, 
div[data-testid="stFileUploader"] label p {
    color: #1A1A1A !important;
    font-weight: 600 !important;
}

.hero-wrapper {
    background: #1A1A1A;
    border-radius: 16px;
    padding: 48px 56px 40px;
    margin-bottom: 32px;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #F7F4EF;
    margin: 0;
}
.hero-title span { color: #C8A96E; }

.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    text-align: center;
}
.score-value { font-weight: 700; color: #C8A96E; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL & RESOURCES
# ─────────────────────────────────────────────────────────────
def build_encoder(input_shape=IMG_SHAPE, latent_dim=LATENT_DIM):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3,3), activation="relu", padding="same")(inputs)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(128, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2,2), padding="same")(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim, activation="relu")(x)
    return Model(inputs, encoded)

@st.cache_resource
def load_encoder():
    model = build_encoder()
    model.load_weights(ENCODER_WEIGHTS)
    return model

@st.cache_data
def load_database():
    db = joblib.load(DATABASE_PATH)
    return db["embeddings"], db["metadata"]

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="hero-wrapper"><h1 class="hero-title">Shirt<span>Finder</span></h1></div>', unsafe_allow_html=True)

if not os.path.exists(ENCODER_WEIGHTS) or not os.path.exists(DATABASE_PATH):
    st.error("Artefak model tidak ditemukan di GitHub!")
    st.stop()

encoder_model = load_encoder()
db_embeddings, db_df = load_database()

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">📥 Input Gambar Query</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih gambar kemeja (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Query Preview", use_column_width=True)

with right_col:
    if uploaded_file:
        st.markdown('<p class="section-label">🔍 Hasil Rekomendasi</p>', unsafe_allow_html=True)
        
        # Preprocess & Predict
        img = query_image.convert("RGB").resize(IMG_SIZE)
        arr = img_to_array(img) / 255.0
        query_emb = encoder_model.predict(np.expand_dims(arr, axis=0), verbose=0)
        
        # Search
        scores = cosine_similarity(query_emb, db_embeddings)[0]
        indices = np.argsort(scores)[::-1][:TOP_K]
        
        cols = st.columns(TOP_K)
        for rank, idx in enumerate(indices, start=1):
            row = db_df.iloc[idx]
            with cols[rank-1]:
                # FIX: Ambil gambar dari URL Myntra menggunakan ID (Abaikan path Kaggle)
                product_id = str(row['id'])
                img_url = f"https://assets.myntassets.com/dpr_1.5,q_60,w_400,c_limit,fl_progressive/assets/images/{product_id}.jpg"
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(img_url, use_column_width=True)
                st.markdown(f"""
                    <p style='font-size:0.8rem; color:#888; margin:5px 0 2px;'>Rank #{rank}</p>
                    <p style='font-size:0.85rem;'>Match: <span class="score-value">{scores[idx]:.2f}</span></p>
                    <p style='font-size:0.7rem; color:#AAA;'>ID: {product_id}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
