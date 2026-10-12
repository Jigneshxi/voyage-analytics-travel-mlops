"""
Voyage Analytics - Production Flask REST API
Serves real-time Flight Price Predictions
"""

import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "flight_price_model.joblib")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("models", "flight_price_model.joblib")

model_pipeline = joblib.load(MODEL_PATH)

def preprocess_flight_input(data_dict):
    """Derive required temporal and route features for inference"""
    df = pd.DataFrame([data_dict]) if isinstance(data_dict, dict) else pd.DataFrame(data_dict)
    
    # Process date
    if 'date' in df.columns:
        dt_series = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = dt_series.dt.month.fillna(10).astype(int)
        df['day'] = dt_series.dt.day.fillna(15).astype(int)
        df['dayofweek'] = dt_series.dt.dayofweek.fillna(2).astype(int)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    else:
        df['month'] = 10
        df['day'] = 15
        df['dayofweek'] = 2
        df['is_weekend'] = 0

    if 'speed' not in df.columns:
        df['speed'] = df['distance'] / (df['time'] + 0.001)

    required_cols = ['from', 'to', 'flightType', 'agency', 'time', 'distance', 'speed', 'month', 'day', 'dayofweek', 'is_weekend']
    return df[required_cols]


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Voyage Analytics Flight Price Prediction API",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "metrics": "/metrics"
        }
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model_pipeline is not None,
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "models", "metrics_summary.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r") as f:
            data = json.load(f)
        return jsonify(data.get("regression", {})), 200
    return jsonify({"message": "Metrics not found"}), 404


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if not payload:
            return jsonify({"error": "Missing JSON request body"}), 400

        # Support single dictionary or list of dictionaries
        is_batch = isinstance(payload, list)
        processed_df = preprocess_flight_input(payload)
        predictions = model_pipeline.predict(processed_df)

        results = [round(float(p), 2) for p in predictions]

        return jsonify({
            "status": "success",
            "count": len(results),
            "predictions": results if is_batch else results[0],
            "currency": "USD",
            "model_version": "v1.0-xgb"
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
