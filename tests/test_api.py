# =============================================================================
# Basic endpoint tests using Flask's test client
# =============================================================================

import json
import os
import joblib
from app import app, FEATURE_NAMES

def test_healthcheck():
    client = app.test_client()
    res = client.get("/")
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "ok"

def test_predict_missing_features():
    client = app.test_client()
    res = client.post("/predict", json={"features": {"foo": 1}})
    assert res.status_code == 400
    data = res.get_json()
    assert "Missing feature(s)" in data.get("error", "")

def test_predict_valid():
    # Load model for building a minimal valid payload
    model_path = os.path.join("models", "breast_cancer_pipeline.joblib")
    assert os.path.exists(model_path)
    _ = joblib.load(model_path)

    # Create a dummy feature vector (zeros) just to test validation path
    payload_features = {name: 0.0 for name in FEATURE_NAMES}
    client = app.test_client()
    res = client.post("/predict", json={"features": payload_features})
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("prediction") in ("Benign", "Malignant")
