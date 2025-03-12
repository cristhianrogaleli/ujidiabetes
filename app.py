import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 1. Load Dataset
data = pd.read_csv("diabetes.csv")

# 2. Standarisasi Data
X = data.drop(columns=['Outcome'])
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Memisahkan Data Training dan Data Testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Membuat Model Naïve Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 5. Evaluasi Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

# 6. Simpan Model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Streamlit App UI Styling
st.set_page_config(page_title="Deteksi Diabetes", page_icon="🩺", layout="centered")
st.markdown("""
    <style>
        .main {background-color: #f4f4f4;}
        h1 {color: #2E8B57; text-align: center;}
        .stButton>button {background-color: #2E8B57; color: white; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Deteksi Diabetes dengan Naïve Bayes")
st.subheader(f"📊 Akurasi Model: {accuracy:.2f}")
st.markdown("---")

st.markdown("**Masukkan Data Pasien:**")

# Input Data untuk Prediksi dengan Kolom yang Lebih Rapi
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=200, value=100)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=80)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=30)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

st.markdown("---")

# Prediksi
if st.button("🔍 Prediksi", key="predict_button"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    result = "✅ Tidak Diabetes" if prediction[0] == 0 else "⚠️ Diabetes"
    
    st.success(f"**Hasil Prediksi: {result}**")
