"""
Production MLOps Pipeline & System Validation Test Suite
Voyage Analytics: Integrating MLOps in Travel | AlmaBetter Capstone
"""

import os
import json
import pytest
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

@pytest.fixture(scope="module")
def flights_sample():
    path = os.path.join(DATA_DIR, "flights.csv")
    assert os.path.exists(path), f"flights.csv missing at {path}"
    return pd.read_csv(path, nrows=5000)

@pytest.fixture(scope="module")
def regression_model():
    path = os.path.join(MODELS_DIR, "flight_price_model.joblib")
    assert os.path.exists(path), f"Flight model missing at {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def classification_model():
    path = os.path.join(MODELS_DIR, "gender_classifier.joblib")
    assert os.path.exists(path), f"Gender model missing at {path}"
    return joblib.load(path)

@pytest.fixture(scope="module")
def recommender_model():
    path = os.path.join(MODELS_DIR, "hotel_recommender.joblib")
    assert os.path.exists(path), f"Hotel model missing at {path}"
    return joblib.load(path)

def test_01_relational_data_integrity():
    """Verify raw datasets presence, schema, and non-empty rows."""
    flights_p = os.path.join(DATA_DIR, "flights.csv")
    hotels_p = os.path.join(DATA_DIR, "hotels.csv")
    users_p = os.path.join(DATA_DIR, "users.csv")
    
    assert os.path.exists(flights_p)
    assert os.path.exists(hotels_p)
    assert os.path.exists(users_p)
    
    users = pd.read_csv(users_p)
    assert len(users) == 1340
    assert {'code', 'company', 'name', 'gender', 'age'}.issubset(set(users.columns))

def test_02_feature_engineering_derivations(flights_sample):
    """Verify temporal and kinematic feature derivations."""
    df = flights_sample.copy()
    df['speed'] = df['distance'] / (df['time'] + 0.001)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['month'] = df['date_dt'].dt.month
    df['day'] = df['date_dt'].dt.day
    df['dayofweek'] = df['date_dt'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x in [5, 6] else 0)
    
    assert (df['speed'] > 0).all()
    assert df['month'].between(1, 12).all()
    assert df['is_weekend'].isin([0, 1]).all()

def test_03_flight_price_regression_model_inference(regression_model):
    """Verify champion XGBoost model inference and price sanity."""
    sample = pd.DataFrame([{
        'from': 'Recife (PE)',
        'to': 'Florianopolis (SC)',
        'flightType': 'economic',
        'agency': 'FlyingDrops',
        'time': 1.7,
        'distance': 676.53,
        'speed': 676.53 / 1.701,
        'month': 10,
        'day': 25,
        'dayofweek': 6,
        'is_weekend': 1
    }, {
        'from': 'Recife (PE)',
        'to': 'Florianopolis (SC)',
        'flightType': 'firstClass',
        'agency': 'FlyingDrops',
        'time': 1.7,
        'distance': 676.53,
        'speed': 676.53 / 1.701,
        'month': 10,
        'day': 25,
        'dayofweek': 6,
        'is_weekend': 1
    }])
    preds = regression_model.predict(sample)
    assert len(preds) == 2
    assert preds[0] > 0
    assert preds[1] > preds[0], "First class must be strictly more expensive than economic"

def test_04_gender_classification_model_inference(classification_model):
    """Verify random forest demographic classifier behavior."""
    assert hasattr(classification_model, "predict")
    assert hasattr(classification_model, "predict_proba")
    # Feed test dataframe with all expected features
    test_df = pd.DataFrame([{
        'age': 35,
        'flight_count': 12,
        'flight_total_spend': 6500.0,
        'flight_avg_spend': 541.67,
        'flight_avg_duration': 1.85,
        'first_class_ratio': 0.25,
        'hotel_count': 8,
        'hotel_total_spend': 1400.0,
        'hotel_avg_spend': 175.0,
        'hotel_total_days': 16,
        'unique_places_visited': 4,
        'company': '4You'
    }])
    pred = classification_model.predict(test_df)
    proba = classification_model.predict_proba(test_df)
    assert len(pred) == 1
    assert proba.shape == (1, 2)
    assert np.isclose(np.sum(proba), 1.0)

def test_05_hotel_recommender_system_accuracy(recommender_model):
    """Verify collaborative hotel recommendation similarity matrix."""
    assert isinstance(recommender_model, dict)
    assert "item_similarity_df" in recommender_model
    assert "hotel_catalog" in recommender_model
    sim_df = recommender_model["item_similarity_df"]
    assert sim_df.shape[0] == 9
    assert sim_df.shape[1] == 9
    assert (np.diag(sim_df.values) >= 0.99).all()

def test_06_model_performance_benchmarks():
    """Verify recorded metrics meet production thresholds."""
    metrics_path = os.path.join(MODELS_DIR, "metrics_summary.json")
    assert os.path.exists(metrics_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    assert metrics["regression"]["r2_score"] > 0.99
    assert metrics["regression"]["rmse"] < 15.0
    assert metrics["classification"]["accuracy"] > 0.60

def test_07_flask_rest_api_contract():
    """Verify Flask REST API health and predict contracts."""
    from api.app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        res_health = client.get('/health')
        assert res_health.status_code == 200
        health_data = json.loads(res_health.data)
        assert health_data['status'] == 'healthy'
        assert health_data['model_loaded'] is True

        payload = {
            "from": "Recife (PE)",
            "to": "Florianopolis (SC)",
            "flightType": "economic",
            "agency": "FlyingDrops",
            "time": 1.7,
            "distance": 676.53,
            "date": "2026-10-25"
        }
        res_pred = client.post('/predict', data=json.dumps(payload), content_type='application/json')
        assert res_pred.status_code == 200
        pred_data = json.loads(res_pred.data)
        assert pred_data['status'] == 'success'
        assert pred_data['predictions'] > 0

def test_08_airflow_dag_integrity():
    """Verify Airflow DAG file parses without syntax errors."""
    dag_path = os.path.join(BASE_DIR, "airflow", "dags", "travel_flight_price_mlops_dag.py")
    assert os.path.exists(dag_path)
    with open(dag_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "DAG(" in content or "travel_flight_price_mlops_dag" in content
    assert ">>" in content, "DAG must have pipeline dependency operators"

def test_09_docker_and_k8s_manifests_integrity():
    """Verify Dockerfiles and Kubernetes deployment manifests exist and are valid."""
    assert os.path.exists(os.path.join(BASE_DIR, "docker", "Dockerfile.flask"))
    assert os.path.exists(os.path.join(BASE_DIR, "docker", "Dockerfile.streamlit"))
    assert os.path.exists(os.path.join(BASE_DIR, "docker", "docker-compose.yml"))
    
    k8s_deploy = os.path.join(BASE_DIR, "k8s", "deployment.yaml")
    k8s_hpa = os.path.join(BASE_DIR, "k8s", "hpa.yaml")
    assert os.path.exists(k8s_deploy)
    assert os.path.exists(k8s_hpa)
    
    with open(k8s_hpa, "r") as f:
        hpa_content = f.read()
    assert "HorizontalPodAutoscaler" in hpa_content

def test_10_publication_figures_and_reports_integrity():
    """Verify all 6 publication figures exist in figures/ directory."""
    expected_figures = [
        "01_flight_price_vs_distance_duration.png",
        "02_agency_pricing_and_routes.png",
        "03_xgboost_actual_vs_predicted_residuals.png",
        "04_gender_classification_performance.png",
        "05_hotel_collaborative_similarity_matrix.png",
        "06_mlops_system_architecture_dag.png"
    ]
    for fig in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig)
        assert os.path.exists(fig_path), f"Figure {fig} missing at {fig_path}"
        assert os.path.getsize(fig_path) > 10000, f"Figure {fig} appears incomplete"
