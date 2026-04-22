# 📊 Customer Churn Prediction System

## 🚀 Overview

This project predicts whether a telecom customer will churn using machine learning. It implements a complete end-to-end pipeline including data preprocessing, model training, evaluation, and deployment using a FastAPI-based REST API.

---

## 🎯 Problem Statement

Customer churn is a major challenge for telecom companies. The objective of this project is to identify customers who are likely to leave, enabling businesses to take proactive retention actions.

---

## 💼 Business Impact

* Retaining existing customers is significantly cheaper than acquiring new ones
* Early identification of churn allows targeted retention strategies
* Even a small reduction in churn can lead to substantial revenue gains

---

## 🧠 Approach

* Data Cleaning & Preprocessing
* Feature Engineering
* Handling Imbalanced Data
* Model Training using XGBoost
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

* **Recall (Churn Class):** ~81%
* **ROC-AUC Score:** ~0.85
* **Threshold:** 0.3 (tuned for better recall)

---

## ⚖️ Threshold Strategy

The default classification threshold of 0.5 was reduced to 0.3 to improve recall.

This ensures that more potential churn customers are identified, even if it increases false positives.

> We prioritize recall because missing a churn customer is more costly than incorrectly flagging a non-churn customer.

---

## 📉 Model Tradeoff

* Higher recall → more churn cases detected
* Lower precision → more false positives

This tradeoff is acceptable in churn prediction problems.

---

## 🔥 Key Insights

* Customers with month-to-month contracts are more likely to churn
* Customers with shorter tenure have higher churn probability
* Electronic payment users show higher churn tendency

---

## 🌐 API Usage

### Endpoint

`POST /predict`

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

### cURL Example

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{
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
}'
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/omjarsaniya/churn-ml-project
cd churn-ml-project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app:app --reload
```

### 5. Open Swagger UI

```bash
http://127.0.0.1:8000/docs
```

---

## 🧩 Project Structure

```text
src/        → training + preprocessing  
app.py      → FastAPI application  
models/     → saved model (ignored in git)  
data/       → dataset  
notebooks/  → EDA notebooks  
```

---

## 🎯 Key Learnings

* Built an end-to-end machine learning pipeline
* Handled class imbalance using weighting and threshold tuning
* Focused on business-relevant metrics (Recall, ROC-AUC)
* Deployed ML model using FastAPI
* Ensured consistent preprocessing using pipelines

---

## 🔮 Future Improvements

* Add Streamlit UI for frontend
* Integrate MLflow for model tracking
* Deploy on cloud (AWS/GCP)
* Add logging and monitoring

---

## 📄 License

This project is for educational purposes.

---

## 🎤 Author

* **Om Jarsaniya**
