"""
Unit tests for Voyage Analytics Flask API and ML Models
"""

import os
import json
import pytest
import joblib
import pandas as pd
from api.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_api_index(client):
    """Test API root status endpoint"""
    response = client.get('/')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'online'
    assert 'version' in data

def test_api_health(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'
    assert data['model_loaded'] is True

def test_predict_single_flight(client):
    """Test prediction for single flight record"""
    payload = {
        "from": "Recife (PE)",
        "to": "Florianopolis (SC)",
        "flightType": "economic",
        "agency": "FlyingDrops",
        "time": 1.7,
        "distance": 676.53,
        "date": "2026-10-25"
    }
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert isinstance(data['predictions'], (int, float))
    assert data['predictions'] > 0

def test_predict_batch_flights(client):
    """Test prediction for batch flights"""
    payload = [
        {
            "from": "Recife (PE)",
            "to": "Florianopolis (SC)",
            "flightType": "economic",
            "agency": "FlyingDrops",
            "time": 1.7,
            "distance": 676.53,
            "date": "2026-10-25"
        },
        {
            "from": "Brasilia (DF)",
            "to": "Florianopolis (SC)",
            "flightType": "firstClass",
            "agency": "CloudFy",
            "time": 1.6,
            "distance": 637.56,
            "date": "2026-10-26"
        }
    ]
    response = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'success'
    assert data['count'] == 2
    assert len(data['predictions']) == 2
    assert data['predictions'][1] > data['predictions'][0]  # first class is more expensive than economic

def test_model_artifact_loading():
    """Verify saved models exist and can be loaded"""
    assert os.path.exists("models/flight_price_model.joblib")
    assert os.path.exists("models/gender_classifier.joblib")
    assert os.path.exists("models/hotel_recommender.joblib")

    model_reg = joblib.load("models/flight_price_model.joblib")
    assert hasattr(model_reg, "predict")
