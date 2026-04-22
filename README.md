# 📊 Customer Churn Prediction System

## 🚀 Overview

This project predicts whether a telecom customer will churn using machine learning. It includes a full pipeline from data preprocessing to model deployment via API.

---

## 🎯 Problem Statement

Customer churn is a major business challenge. The goal is to identify customers likely to leave so that retention strategies can be applied.

---

## 🧠 Approach

* Data Cleaning & Preprocessing
* Feature Engineering
* Handling Imbalanced Data
* Model Training (XGBoost)
* Evaluation using Recall & ROC-AUC
* Threshold Tuning
* Deployment using FastAPI

---

## ⚙️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* FastAPI

---

## 📈 Model Performance

* Recall (Churn): ~81%
* ROC-AUC: ~0.85
* Threshold tuned to 0.3 for better churn detection

---

## 🔥 Key Insights

* Customers with month-to-month contracts are more likely to churn
* Short tenure customers have higher churn probability
* Electronic payment users show higher churn tendency

---

## 🌐 API Usage

### Endpoint

POST `/predict`

### Sample Input

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70,
  "TotalCharges": 840
}
```

### Sample Output

```json
{
  "prediction": 1,
  "churn_probability": 0.79,
  "threshold_used": 0.3
}
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run API

```bash
uvicorn app:app --reload
```

### 3. Open Swagger UI

```bash
http://127.0.0.1:8000/docs
```

---

## 🧩 Project Structure

```bash
src/        → training + preprocessing
app.py      → API
models/     → saved model (ignored)
data/       → dataset
```

---

## 🎤 Author

Om Jarsaniya
