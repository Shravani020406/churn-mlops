from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Find the model relative to this file, so it also works inside Docker later
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Churn Prediction API", version="1.0")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = None


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running. Open /docs to try it."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.model_dump()])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    probability = float(model.predict_proba(df)[0, 1])

    if probability >= 0.7:
        risk = "high"
    elif probability >= 0.4:
        risk = "medium"
    else:
        risk = "low"

    return {
        "will_churn": probability >= 0.5,
        "probability": round(probability, 3),
        "risk_level": risk,
    }