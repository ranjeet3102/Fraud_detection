from fastapi.testclient import TestClient
from deployment.fastapi_app import app  # Assuming FastAPI app in src/api.py

client = TestClient(app)

def test_health_check():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}

def test_predict_route():
    sample_input = {
        "gender": 1,
        "age": 35,
        "previously_insured": 0,
        "vehicle_age": 1,
        "vehicle_damage": 1,
        "policy_sales_channel": 26
    }
    res = client.post("/predict", json=sample_input)
    assert res.status_code == 200
    assert "fraud_probability" in res.json()
