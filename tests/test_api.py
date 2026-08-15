import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient
from app import app, CLASSIFICATION_THRESHOLD  # noqa: E402

client = TestClient(app)

SAMPLE_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 70, "TotalCharges": 840,
}


def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_returns_expected_shape():
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"prediction", "churn_probability", "threshold_used"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["threshold_used"] == CLASSIFICATION_THRESHOLD


def test_predict_rejects_missing_field():
    bad_payload = {k: v for k, v in SAMPLE_CUSTOMER.items() if k != "tenure"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # FastAPI/Pydantic validation error


def test_prediction_is_consistent_with_threshold():
    """The binary prediction should always agree with (probability >= threshold) —
    catches bugs where the two get out of sync after a refactor."""
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    body = response.json()
    expected = int(body["churn_probability"] >= body["threshold_used"])
    assert body["prediction"] == expected
