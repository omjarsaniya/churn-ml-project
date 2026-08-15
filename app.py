import sys
from pathlib import Path

# allow importing from src/ regardless of where uvicorn is launched from
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from paths import PIPELINE_PATH

# Chosen by maximizing F2-score on the precision-recall curve in src/train.py
# (F2 weights recall 2x precision, matching the business cost of a missed churner).
# Re-run `python src/train.py` and check reports/precision_recall_curve.png if the
# model is retrained on new data — this value may shift.
CLASSIFICATION_THRESHOLD = 0.26


class CustomerData(BaseModel):
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
    TotalCharges: float


app = FastAPI(title="Customer Churn Prediction API")

pipeline = joblib.load(PIPELINE_PATH)


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.model_dump()])

    prob = float(pipeline.predict_proba(df)[0][1])
    prediction = int(prob >= CLASSIFICATION_THRESHOLD)

    return {
        "prediction": prediction,
        "churn_probability": round(prob, 4),
        "threshold_used": CLASSIFICATION_THRESHOLD,
    }
