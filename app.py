import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model & preprocessor
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# App title
st.title("💰 Medical Insurance Cost Predictor")

st.write("Enter your details to predict insurance charges")

# ---- User Inputs ----
age = st.slider("Age", 18, 100, 30)
bmi = st.slider("BMI", 15.0, 50.0, 25.0)
children = st.selectbox("Number of Children", [0,1,2,3,4,5])

sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Create input dataframe
input_data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex": [sex],
    "smoker": [smoker],
    "region": [region]
})

# ---- Prediction ----
if st.button("Predict Insurance Cost"):
    input_processed = preprocessor.transform(input_data)
    prediction = model.predict(input_processed)

    st.success(f"💵 Predicted Insurance Charge: ${prediction[0]:,.2f}")



# Running Command - python -m streamlit run app.py
