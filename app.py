# ============================================================
# 🚖 UBER FARE PREDICTION APP
# ============================================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

# ============================================================
# 1️⃣ Load Best Model (langsung dari root folder)
# ============================================================
@st.cache_resource
def load_model():
    """
    Memuat pipeline model terbaik dari file joblib di root folder.
    """
    model_path = "random_forest_pipeline.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ File model '{model_path}' tidak ditemukan. "
            "Pastikan file sudah di-upload ke root repo GitHub."
        )
    return joblib.load(model_path)

# Load model
model = load_model()

# ============================================================
# 2️⃣ App Configuration
# ============================================================
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

st.title("🚖 Uber Fare Prediction App")
st.markdown("Masukkan detail perjalanan untuk memprediksi tarif Uber.")

# ============================================================
# 3️⃣ Input Form
# ============================================================
with st.form("fare_form"):
    st.subheader("📝 Trip Details Input")

    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=10, value=1)
    pickup_latitude = st.number_input("Pickup Latitude", format="%.6f")
    pickup_longitude = st.number_input("Pickup Longitude", format="%.6f")
    dropoff_latitude = st.number_input("Dropoff Latitude", format="%.6f")
    dropoff_longitude = st.number_input("Dropoff Longitude", format="%.6f")
    pickup_hour = st.slider("Pickup Hour", 0, 23, 12)
    pickup_day = st.slider("Pickup Day", 1, 31, 15)
    pickup_month = st.slider("Pickup Month", 1, 12, 6)
    pickup_year = st.selectbox("Pickup Year", [2009, 2010, 2011, 2012, 2013, 2014, 2015])

    submitted = st.form_submit_button("🔮 Predict Fare")

# ============================================================
# 4️⃣ Prediction
# ============================================================
if submitted:
    try:
        # Buat DataFrame sesuai urutan fitur model
        input_data = pd.DataFrame([[
            passenger_count,
            pickup_latitude,
            pickup_longitude,
            dropoff_latitude,
            dropoff_longitude,
            pickup_hour,
            pickup_day,
            pickup_month,
            pickup_year
        ]], columns=[
            'passenger_count',
            'pickup_latitude',
            'pickup_longitude',
            'dropoff_latitude',
            'dropoff_longitude',
            'pickup_hour',
            'pickup_day',
            'pickup_month',
            'pickup_year'
        ])

        # Prediksi tarif
        fare_pred = model.predict(input_data)[0]
        st.success(f"💰 Predicted Fare: **${fare_pred:.2f}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
