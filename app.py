import streamlit as st
import pandas as pd
import pickle
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Set Konfigurasi Halaman
st.set_page_config(page_title="Analisis Sentimen Pelabuhan Ferry Batam", layout="centered")

# 2. Load Model dan Vectorizer dengan Caching agar Cepat
@st.cache_resource
def load_nlp_objects():
    with open('svm_model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('tfidf_vectorizer.pkl', 'rb') as v_file:
        vec = pickle.load(v_file)
    return model, vec

try:
    model, vectorizer = load_nlp_objects()
except FileNotFoundError:
    st.error("File model atau vectorizer tidak ditemukan. Pastikan svm_model.pkl dan tfidf_vectorizer.pkl sudah diunggah ke GitHub.")

# 3. Inisialisasi Preprocessing Sastrawi
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# 4. Antarmuka Pengguna (UI Streamlit)
st.title("🚢 Analisis Sentimen Ulasannya Pelabuhan Ferry di Batam")
st.write("Aplikasi ini memprediksi sentimen ulasan penumpang menggunakan model Machine Learning SVM yang dikombinasikan dengan pembobotan TF-IDF.")

# Pilihan Pelabuhan
pelabuhan_list = [
    "Batam Center International Ferry Terminal",
    "Pelabuhan Ferry Harbour Bay",
    "Terminal Ferry Telaga Punggur Batam",
    "Nongsapura Ferry Terminal",
    "Sekupang Ferry Terminal"
]
selected_pelabuhan = st.selectbox("Pilih Pelabuhan Ferry:", pelabuhan_list)

# Input Ulasan
user_review = st.text_area("Masukkan Ulasan Penumpang di Sini:", placeholder="Contoh: Pelayanan di pelabuhan ini sangat cepat dan ruang tunggunya bersih...")

if st.button("Analisis Sentimen"):
    if user_review.strip() == "":
        st.warning("Silakan masukkan teks ulasan terlebih dahulu.")
    else:
        # Proses NLP
        with st.spinner("Sedang memproses teks dan melakukan klasifikasi..."):
            cleaned_text = preprocess_text(user_review)
            
            # Tampilkan tahapan pra-proses untuk transparansi akademis
            st.subheader("Hasil Preprocessing Teks")
            st.info(f"**Teks Asli:** {user_review}\n\n**Setelah Stemming & Stopword:** {cleaned_text}")
            
            # Transformasi TF-IDF & Prediksi
            text_tfidf = vectorizer.transform([cleaned_text])
            prediction = model.predict(text_tfidf)[0]
            probabilities = model.predict_proba(text_tfidf)[0]
            class_labels = model.classes_
            
            # Menampilkan Hasil Akhir
            st.subheader("Hasil Prediksi Sentimen")
            if prediction == 'positif':
                st.success(f"Sentimen Terdeteksi: **POSITIF** 🟢")
            elif prediction == 'negatif':
                st.error(f"Sentimen Terdeteksi: **NEGATIF** 🔴")
            else:
                st.warning(f"Sentimen Terdeteksi: **NETRAL** 🟡")
                
            # Menampilkan Probabilitas Tiap Kelas
            st.write("**Probabilitas Distribusi Kelas:**")
            prob_df = pd.DataFrame({
                'Sentimen': class_labels,
                'Keyakinan Model (%)': [round(p * 100, 2) for p in probabilities]
            })
            st.dataframe(prob_df, use_container_width=True)