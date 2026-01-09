from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

def scale_features(X, scaler_path = "../models/scaler.pkl"):
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)
    return x_scaled

def load_scaler(scaler_path="../models/scaler.pkl"):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)