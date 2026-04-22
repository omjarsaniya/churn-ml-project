from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

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

app = FastAPI()

# load trained pipeline
pipeline = joblib.load("models/pipeline.pkl")


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])

    prob = pipeline.predict_proba(df)[0][1]

    threshold = 0.3
    prediction = int(prob > threshold)

    return {
        "prediction": int(prediction),
        "churn_probability": float(prob), 
        "threshold_used": float(threshold)
    }
