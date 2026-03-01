from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from xgboost import XGBClassifier

app = FastAPI()

# Load model and scaler
model = XGBClassifier()
model.load_model("models/xgboost_model.json")
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API running"}

@app.post("/predict")
def predict(data: DiabetesInput):
    features = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure,
                          data.SkinThickness, data.Insulin, data.BMI,
                          data.DiabetesPedigreeFunction, data.Age]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    return {"prediction": int(prediction), "probability": float(probability)}