import numpy as np
import pickle
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model("models/xgboost_model.json")

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Example input: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
sample_input = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

# Scale input
sample_input = scaler.transform(sample_input)

# Predict
prediction = model.predict(sample_input)[0]
probability = model.predict_proba(sample_input)[0][1]

print("Prediction (0=No, 1=Yes):", prediction)
print("Probability of diabetes:", probability)