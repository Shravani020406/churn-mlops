from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 90.0, "TotalCharges": 90.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
    "MonthlyCharges": 65.0, "TotalCharges": 3900.0,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_response():
    response = client.post("/predict", json=HIGH_RISK)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_level"] in ["low", "medium", "high"]
    assert isinstance(body["will_churn"], bool)


def test_high_risk_customer_scores_higher_than_low_risk():
    high = client.post("/predict", json=HIGH_RISK).json()["probability"]
    low = client.post("/predict", json=LOW_RISK).json()["probability"]
    assert high > low


def test_missing_field_is_rejected():
    bad = dict(HIGH_RISK)
    del bad["Contract"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422