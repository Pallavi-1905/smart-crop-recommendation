import streamlit as st
import pickle
import numpy as np

# Load saved files
model = pickle.load(open("crop_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

st.set_page_config(page_title="Smart Crop Recommendation", page_icon="🌾")

st.title("🌾 AI-Powered Smart Crop Recommendation System")

st.write("Enter soil and climate details to get recommended crop.")

# Input fields
N = st.number_input("Nitrogen (N)", min_value=0.0)
P = st.number_input("Phosphorus (P)", min_value=0.0)
K = st.number_input("Potassium (K)", min_value=0.0)
temperature = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Recommend Crop"):

    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    crop_name = le.inverse_transform(prediction)

    st.success(f"🌱 Recommended Crop: {crop_name[0]}")