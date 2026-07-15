import streamlit as st
import joblib
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load Model dan Vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Setup Sastrawi
factory_stopword = StopWordRemoverFactory()
stopword = factory_stopword.create_stop_word_remover()
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()

def preprocess_input(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stopword.remove(text)
    text = stemmer.stem(text)
    return text

st.title("Analisis Sentimen Ulasan Google Maps")
st.write("Masukkan ulasan untuk memprediksi apakah sentimennya Positif, Negatif, atau Netral.")

user_input = st.text_area("Tulis ulasan di sini:")

if st.button("Analisis Sentimen"):
    if user_input:
        clean_text = preprocess_input(user_input)
        vectorized_text = vectorizer.transform([clean_text])
        prediction = svm_model.predict(vectorized_text)
        
        st.subheader("Hasil Prediksi:")
        if prediction[0] == 'Positif':
            st.success("Sentimen: Positif 😊")
        elif prediction[0] == 'Negatif':
            st.error("Sentimen: Negatif 😠")
        else:
            st.warning("Sentimen: Netral 😐")
    else:
        st.write("Tolong masukkan teks terlebih dahulu.")
