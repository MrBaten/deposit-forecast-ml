"""
REST API for Customer Deposit Forecasting.
Run with: uvicorn src.api:app --reload
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants
MODEL_PATH = "../models/random_forest.pkl"

# Initialize FastAPI
app = FastAPI(
    title="Deposit Forecasting API",
    description="API for predicting next-day customer deposits.",
    version="1.0.0"
)

# Global model variable
model = None

@app.on_event("startup")
def load_model():
    """Load the model on startup."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    deposit_lag_1d: float
    deposit_lag_7d: float
    days_since_last_deposit: int
    rolling_mean_30d: float
    rolling_std_30d: float
    rolling_sum_30d: float
    wow_change: float
    is_weekend: int
    # Add other features as optional or required based on model needs
    # For simplicity, we'll allow extra fields to be passed in a dict if needed
    # but explicit fields are better for documentation.
    
    class Config:
        schema_extra = {
            "example": {
                "deposit_lag_1d": 50.0,
                "deposit_lag_7d": 45.0,
                "days_since_last_deposit": 1,
                "rolling_mean_30d": 48.5,
                "rolling_std_30d": 12.0,
                "rolling_sum_30d": 1455.0,
                "wow_change": 0.1,
                "is_weekend": 0
            }
        }

class PredictionResponse(BaseModel):
    predicted_deposit: float
    is_high_value: bool

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict deposit for a single customer."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        # Note: In a real scenario, we need to ensure ALL features expected by the model are present.
        # Here we assume the model can handle missing features or we fill them.
        # For robustness, we should match the exact feature set.
        
        # Get expected features from model if available
        try:
            expected_features = model.feature_names_in_
        except:
            # Fallback list if attribute missing
            expected_features = [
                'deposit_lag_1d', 'deposit_lag_7d', 'days_since_last_deposit',
                'rolling_mean_30d', 'rolling_std_30d', 'rolling_sum_30d',
                'wow_change', 'is_weekend'
            ]
            
        input_data = request.dict()
        
        # Fill missing expected features with 0 (or some default)
        for feat in expected_features:
            if feat not in input_data:
                input_data[feat] = 0.0
                
        df = pd.DataFrame([input_data])
        
        # Reorder columns
        df = df[expected_features]
        
        # Predict
        prediction = model.predict(df)[0]
        prediction = max(0, prediction)
        
        return {
            "predicted_deposit": round(prediction, 2),
            "is_high_value": prediction > 100.0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        for req in requests:
            # Reuse logic (could be optimized by vectorizing)
            # For simplicity, calling loop here
            # In production, convert list of dicts to DataFrame once and predict
            
            try:
                expected_features = model.feature_names_in_
            except:
                 expected_features = [
                    'deposit_lag_1d', 'deposit_lag_7d', 'days_since_last_deposit',
                    'rolling_mean_30d', 'rolling_std_30d', 'rolling_sum_30d',
                    'wow_change', 'is_weekend'
                ]
            
            input_data = req.dict()
            for feat in expected_features:
                if feat not in input_data:
                    input_data[feat] = 0.0
            
            df = pd.DataFrame([input_data])
            df = df[expected_features]
            
            pred = model.predict(df)[0]
            pred = max(0, pred)
            
            results.append({
                "predicted_deposit": round(pred, 2),
                "is_high_value": pred > 100.0
            })
            
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
