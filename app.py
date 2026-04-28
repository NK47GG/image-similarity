# app.py
import os
import sys
import numpy as np
import streamlit as st
from PIL import Image
import joblib

# --- HACK: Menangani isu 'numpy._core' untuk kompatibilitas versi ---
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

/* FIX: Warna font label agar tidak putih dan kontras */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1A1A1A !important; 
    margin-bottom: 10px;
}

/* Memaksa teks widget (Radio & Uploader) menjadi hitam */
div[data-testid="stWidgetLabel"] label p, 
div[data-testid="stRadio"] label, 
div[data-testid="stFileUploader"] label p {
    color: #1A1A1A !important;
    font-weight: 500 !important;
}

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
.hero-subtitle { color: #888; font-weight: 300; margin: 0; }

.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    height: 100%;
}
.score-value { font-weight: 600; color: #C8A96E; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODEL DEFINITION
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
    x = Flatten()(x)
    encoded = Dense(latent_dim, activation="relu", name="latent_space")(x)

    # Decoder hanya sebagai dummy untuk loading weights
    x = Dense(conv_shape[0] * conv_shape[1] * conv_shape[2], activation="relu")(encoded)
    x = Reshape(conv_shape)(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(256, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(128, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(64,  (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)

    encoder = Model(inputs, encoded)
    return encoder

# ─────────────────────────────────────────────────────────────
# CACHED RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat encoder model…")
def load_encoder():
    encoder = build_autoencoder()
    encoder.load_weights(ENCODER_WEIGHTS)
    return encoder

@st.cache_data(show_spinner="Memuat database…")
def load_database():
    db = joblib.load(DATABASE_PATH)
    return db["embeddings"], db["metadata"].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def find_similar(pil_image, encoder_model, db_embeddings, db_df):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    query_emb = encoder_model.predict(np.expand_dims(arr, axis=0), verbose=0)
    scores    = cosine_similarity(query_emb, db_embeddings)[0]
    indices   = np.argsort(scores)[::-1][:TOP_K]
    results   = db_df.iloc[indices].copy().reset_index(drop=True)
    results["similarity_score"] = scores[indices]
    return results

# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div>
        <h1 class="hero-title">Shirt<span>Finder</span></h1>
        <p class="hero-subtitle">Visual search berbasis Convolutional Autoencoder</p>
    </div>
</div>
""", unsafe_allow_html=True)

missing = [f for f in [ENCODER_WEIGHTS, DATABASE_PATH] if not os.path.exists(f)]
if missing:
    st.error(f"File missing: {missing}")
    st.stop()

encoder_model = load_encoder()
db_embeddings, db_df = load_database()

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<p class="section-label">📥 Input Gambar Query</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Pilih gambar kemeja", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        query_image = Image.open(uploaded_file)
        st.image(query_image, caption="Preview Query", use_column_width=True)

with right_col:
    if uploaded_file:
        st.markdown('<p class="section-label">🔍 Hasil Rekomendasi</p>', unsafe_allow_html=True)
        results = find_similar(query_image, encoder_model, db_embeddings, db_df)
        
        cols = st.columns(TOP_K)
        for rank, (col, (_, row)) in enumerate(zip(cols, results.iterrows()), start=1):
            with col:
                # FIX: Ambil gambar dari URL CDN Myntra menggunakan ID Produk
                product_id = str(row['id'])
                img_url = f"https://assets.myntassets.com/dpr_1.5,q_60,w_400,c_limit,fl_progressive/assets/images/{product_id}.jpg"
                
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.image(img_url, use_column_width=True)
                st.markdown(f"""
                    <p style='font-size:0.8rem; color:#888; margin-bottom:2px;'>#{rank}</p>
                    <p style='font-size:0.85rem;'>Match: <span class="score-value">{row['similarity_score']:.2f}</span></p>
                    <p style='font-size:0.7rem; color:#AAA;'>ID: {product_id}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Silakan upload gambar kemeja untuk memulai.")
