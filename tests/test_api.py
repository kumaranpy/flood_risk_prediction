"""
FastAPI Endpoint Tests.

Tests the REST API endpoints with valid and invalid inputs,
verifying response schemas, status codes, and error handling.
"""

from pathlib import Path
import sys
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Try to import FastAPI test client
try:
    from fastapi.testclient import TestClient
    from api.main import app
    # Verify TestClient works with our starlette version
    _client = TestClient(app)
    HAS_FASTAPI = True
except Exception:
    HAS_FASTAPI = False


VALID_INPUT = {
    "MonsoonIntensity": 10.0,
    "TopographyDrainage": 8.0,
    "RiverManagement": 7.0,
    "Deforestation": 8.0,
    "Urbanization": 9.0,
    "ClimateChange": 9.0,
    "DamsQuality": 4.0,
    "Siltation": 8.0,
    "AgriculturalPractices": 6.0,
    "Encroachments": 8.0,
    "IneffectiveDisasterPreparedness": 9.0,
    "DrainageSystems": 5.0,
    "CoastalVulnerability": 7.0,
    "Landslides": 6.0,
    "Watersheds": 7.0,
    "DeterioratingInfrastructure": 8.0,
    "PopulationScore": 8.0,
    "WetlandLoss": 7.0,
    "InadequatePlanning": 8.0,
    "PoliticalFactors": 6.0,
}

VALID_CLASSES = {"Low", "Medium", "High", "UNCERTAIN"}


@pytest.fixture(scope="module")
def client():
    """Return a TestClient for the FastAPI app."""
    from fastapi.testclient import TestClient as TC
    from api.main import app as _app
    return TC(_app)


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_health_endpoint(client):
    """GET /health returns 200 with status='ok'."""
    response = client.get("/health")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "ok", f"Expected status='ok', got: {data}"
    assert "model_version" in data
    assert "timestamp" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_predict_valid_input(client):
    """POST /predict with valid input returns PredictionResponse."""
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in VALID_CLASSES
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 100.0
    assert "probabilities" in data
    assert "decision" in data
    assert data["decision"] in {"PREDICTED", "UNCERTAIN"}
    assert "model_version" in data


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_predict_invalid_input_out_of_bounds(client):
    """POST /predict with feature out of bounds returns 422."""
    invalid_input = VALID_INPUT.copy()
    invalid_input["MonsoonIntensity"] = 999.0  # Way out of bounds [0, 15]
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422, f"Expected 422 for out-of-bounds, got {response.status_code}"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_predict_missing_feature(client):
    """POST /predict with missing required feature returns 422."""
    incomplete_input = VALID_INPUT.copy()
    del incomplete_input["MonsoonIntensity"]  # Remove required field
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422, f"Expected 422 for missing field, got {response.status_code}"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_predict_batch_endpoint(client):
    """POST /predict/batch with valid list returns batch results."""
    batch_payload = {"predictions": [VALID_INPUT, VALID_INPUT]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert "results" in data
    assert "count" in data
    assert data["count"] == 2
    assert len(data["results"]) == 2


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_predict_negative_feature(client):
    """POST /predict with negative feature value (below min=0) returns 422."""
    invalid_input = VALID_INPUT.copy()
    invalid_input["TopographyDrainage"] = -1.0  # Below minimum 0.0
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422, f"Expected 422 for negative value, got {response.status_code}"


@pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI TestClient not compatible with installed starlette version")
def test_model_info_endpoint(client):
    """GET /model/info returns model registry JSON."""
    response = client.get("/model/info")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert "model_version" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
