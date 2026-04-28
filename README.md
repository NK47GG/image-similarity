import os

readme_content = """# 🔍 ShirtFinder — Visual Search Engine

**ShirtFinder** adalah aplikasi pencarian produk berbasis visual yang menggunakan teknologi Deep Learning untuk menemukan kemeja yang serupa berdasarkan fitur visual (bentuk, pola, dan warna). Proyek ini memanfaatkan arsitektur **Convolutional Autoencoder** untuk mengestraksi representasi fitur (*embeddings*) dan **Cosine Similarity** untuk melakukan pencarian kemiripan.

### 🌐 Live Demo
Aplikasi ini dapat diakses langsung melalui tautan berikut:
👉 [**https://image-similarity-mawmbxnui3vmbwfn6happyz.streamlit.app/**](https://image-similarity-mawmbxnui3vmbwfn6happyz.streamlit.app/)

---

## 🚀 Fitur Utama
- **Visual Recommendation**: Menampilkan Top-5 produk yang paling mirip dengan tingkat akurasi tinggi.
- **Dual Input Method**:
  - 📁 **Upload Gallery**: Unggah foto kemeja langsung dari perangkat.
  - 📷 **Camera Input**: Ambil foto secara *real-time* menggunakan kamera perangkat.
- **Optimized Image Loading**: Menggunakan sistem *lookup table* untuk memanggil gambar secara efisien dari CDN Myntra tanpa membebani penyimpanan lokal.
- **Modern UI/UX**: Antarmuka gelap yang futuristik dan responsif menggunakan Streamlit.

## 🛠️ Tech Stack
- **Framework**: Streamlit
- **Deep Learning**: TensorFlow / Keras (Convolutional Autoencoder)
- **Data Processing**: Pandas, NumPy
- **Similarity Metric**: Scikit-learn (Cosine Similarity)
- **Model Persistence**: Joblib, H5 Weights

## 🧠 Cara Kerja Model
1. **Encoder**: Gambar kemeja (128x128) diproses melalui lapisan konvolusi untuk diekstraksi menjadi vektor dimensi rendah (*Latent Space* 256d).
2. **Database Embedding**: Seluruh gambar dalam dataset telah diproses sebelumnya menjadi vektor *embeddings* dan disimpan dalam file `database.pkl`.
3. **Inference**: Saat pengguna memasukkan gambar baru, sistem mengekstraksi vektor fiturnya, lalu membandingkannya dengan database menggunakan perhitungan jarak *Cosine Similarity*.

## 📦 Instalasi & Penggunaan Lokal

Jika ingin menjalankan proyek ini di lingkungan lokal, ikuti langkah-langkah berikut:

1. **Clone Repository**
   ```bash
   git clone [https://github.com/username-kamu/nama-repo.git](https://github.com/username-kamu/nama-repo.git)
   cd nama-repo
