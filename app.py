import streamlit as st
import numpy as np
import pickle
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("models/xgboost_model.json")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Diabetes Prediction App")

# User input
Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose = st.number_input("Glucose", 0, 200, 120)
BloodPressure = st.number_input("BloodPressure", 0, 150, 70)
SkinThickness = st.number_input("SkinThickness", 0, 100, 20)
Insulin = st.number_input("Insulin", 0, 900, 79)
BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.number_input("DiabetesPedigreeFunction", 0.0, 2.5, 0.5)
Age = st.number_input("Age", 1, 120, 33)

if st.button("Predict"):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                          Insulin, BMI, DiabetesPedigreeFunction, Age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    st.write("Prediction (0 = No, 1 = Yes):", prediction)
    st.write("Probability of Diabetes:", round(probability, 2))