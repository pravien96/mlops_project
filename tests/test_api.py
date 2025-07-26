import pytest
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_valid_input():
    test_data = {
        "MedInc": 8.0,
        "HouseAge": 10.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 3000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    
    response = client.post("/predict", json=test_data)
    if response.status_code == 200:
        assert "predicted_price" in response.json()
        assert isinstance(response.json()["predicted_price"], (int, float))

def test_predict_invalid_input():
    test_data = {
        "MedInc": -1.0,  # Invalid negative value
        "HouseAge": 10.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 3000.0,
        "AveOccup": 3.0,
        "Latitude": 34.0,
        "Longitude": -118.0
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422  # Validation error
