"""
Quick manual sanity-check for the trained pipeline.

Run from anywhere:
    python src/predict.py
"""

import joblib
import pandas as pd

from paths import PIPELINE_PATH

pipeline = joblib.load(PIPELINE_PATH)

sample = {
    "gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "tenure": [12],
    "PhoneService": ["Yes"],
    "MultipleLines": ["No"],
    "InternetService": ["Fiber optic"],
    "OnlineSecurity": ["No"],
    "OnlineBackup": ["Yes"],
    "DeviceProtection": ["No"],
    "TechSupport": ["No"],
    "StreamingTV": ["Yes"],
    "StreamingMovies": ["Yes"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [70],
    "TotalCharges": [840],
}

if __name__ == "__main__":
    df = pd.DataFrame(sample)
    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)[0][1]
    print("Prediction:", prediction[0], "| Churn probability:", round(probability, 3))
