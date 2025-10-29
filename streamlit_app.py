import streamlit as st
import pandas as pd

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Loan Default Prediction Report", layout="wide")

st.title("📊 Final Project: Loan Default Prediction using XGBoost")
st.markdown("""
Dashboard ini menampilkan hasil analisis data dan model machine learning untuk memprediksi kemungkinan **gagal bayar pinjaman**.  
Seluruh visualisasi dan metrik berasal dari hasil eksperimen di notebook `FinalProjectDS.ipynb`.
""")

# -----------------------------
# 1️⃣ Hasil Evaluasi Model
# -----------------------------
st.header("🧠 Hasil Evaluasi Model")

st.markdown("""
Berikut perbandingan performa model **XGBoost** pada berbagai tahap pelatihan:
""")

# Tampilkan tabel hasil evaluasi
df_metrics = pd.DataFrame({
    "Model": [
        "Base XGBoost (Train)",
        "Base XGBoost (Test)",
        "XGBoost Tuning (Train)",
        "XGBoost Tuning (Test)",
        "XGBoost Tuning + SMOTE (Train)",
        "XGBoost Tuning + SMOTE (Test)",
    ],
    "Precision": [0.990245, 0.966574, 0.990269, 0.984496, 0.992906, 0.963585],
    "Recall": [0.802812, 0.744635, 0.751142, 0.726753, 0.811775, 0.738197],
    "F1-Score": [0.886732, 0.841212, 0.854287, 0.836214, 0.893251, 0.835966],
    "ROC AUC": [0.987407, 0.947291, 0.976010, 0.947167, 0.991971, 0.946877]
})

st.dataframe(df_metrics, use_container_width=True)

# -----------------------------
# 2️⃣ Visualisasi Hasil Model
# -----------------------------
st.header("📉 Visualisasi Model XGBoost")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    st.image("cm.png", use_column_width=True)

with col2:
    st.subheader("ROC Curve")
    st.image("roc.png", use_column_width=True)

# -----------------------------
# 3️⃣ Feature Importance
# -----------------------------
st.header("🔍 Feature Importance")

st.markdown("""
Visualisasi berikut menunjukkan fitur paling berpengaruh terhadap prediksi model XGBoost.
""")

st.image("fi.png", use_column_width=True)

st.info("""
**Top 3 Feature Penting:**
1. `loan_grade` – berhubungan langsung dengan risiko gagal bayar  
2. `person_income` – semakin tinggi pendapatan, semakin kecil kemungkinan gagal bayar  
3. `loan_percent_income` – rasio besar pinjaman terhadap pendapatan nasabah
""")

# -----------------------------
# 4️⃣ Kesimpulan Akhir
# -----------------------------
st.header("📝 Kesimpulan")

st.markdown("""
- Base XGBoost Model menunjukkan performa terbaik tanpa perlu hyperparameter tuning atau SMOTE. Upaya tuning dan handling class imbalance justru menurunkan keseimbangan overall model, khususnya dalam hal recall dan F1-Score yang krusial untuk use case prediksi gagal bayar.  
- Fitur seperti `loan_grade`, `person_income`, dan `loan_percent_income` memiliki kontribusi terbesar terhadap hasil prediksi.  
- Model ini dapat digunakan sebagai sistem pendukung keputusan untuk menilai risiko pinjaman baru.
""")

st.success("Dashboard selesai dimuat ✅")
