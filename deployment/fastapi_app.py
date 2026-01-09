from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = os.getenv("MODEL_PATH", "../models/final_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "../models/scaler.pkl")

scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection API")

class PredictionRequest(BaseModel):
    gender: int
    age: float
    previously_insured: int
    vehicle_age: int
    vehicle_damage: int
    policy_sales_channel: float

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict(input_data: PredictionRequest):
    df = pd.DataFrame([input_data.dict()])
    try:
        X_scaled = scaler.transform(df)
        proba = model.predict_proba(X_scaled)[:, 1][0]
        return {"fraud_probability": float(proba)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
