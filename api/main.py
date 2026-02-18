from fastapi import FastAPI
from typing import Dict, Any
import pandas as pd
import joblib
import os

# Initialize the API
app = FastAPI(title="Telco Churn Prediction API", version="1.0")

# Load your trained AI brain!
model_path = '../models/churn_model.pkl' if os.path.exists('../models/churn_model.pkl') else 'models/churn_model.pkl'
pipeline = joblib.load(model_path)

# Create a simple endpoint to make sure the API is awake
@app.get("/")
def home():
    return {"message": "Churn Prediction API is up and running!"}

# Create the prediction endpoint
@app.post("/predict")
def predict_churn(customer_data: Dict[str, Any]):
    # 1. Convert the incoming JSON data into a Pandas DataFrame
    df = pd.DataFrame([customer_data])
    
    # 2. Ask the AI to predict
    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)
    
    # 3. Send the result back!
    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": float(probability[0][1]),
        "risk_level": "High" if probability[0][1] > 0.5 else "Low"
    }