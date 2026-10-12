"""
Apache Airflow DAG: Voyage Analytics Automated MLOps Retraining Pipeline
Orchestrates End-to-End Flight Price Regression Pipeline:
1. Ingestion & Schema Validation
2. Feature Engineering & Quality Checks
3. Model Retraining with MLflow Logging
4. Performance Gate & Model Registry Promotion
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import json

default_args = {
    'owner': 'voyage_mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email': ['evaluator@almabetter.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def task_validate_ingestion(**kwargs):
    """Verify presence, integrity, and non-empty schema of dataset"""
    import pandas as pd
    data_path = kwargs.get('data_path', 'data/flights.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Flight dataset missing at: {data_path}")
    
    df = pd.read_csv(data_path, nrows=100)
    required_cols = {'travelCode', 'userCode', 'from', 'to', 'flightType', 'price', 'time', 'distance', 'agency', 'date'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing mandatory columns: {missing}")
    
    print(f"Validation successful. Verified {len(df.columns)} columns across incoming flight stream.")
    return {"status": "validated", "sample_rows": len(df)}

def task_data_drift_check(**kwargs):
    """Detect potential covariate drift or abnormal fare fluctuations"""
    import pandas as pd
    import numpy as np
    
    df = pd.read_csv('data/flights.csv', nrows=5000)
    mean_price = df['price'].mean()
    std_price = df['price'].std()
    
    print(f"Drift Analysis -> Mean Price: ${mean_price:.2f}, Std Dev: ${std_price:.2f}")
    # Threshold checks for operational safety
    if mean_price < 50 or mean_price > 5000:
        raise ValueError(f"Extreme data drift detected! Mean fare of ${mean_price:.2f} is outside normal bounds.")
    
    return {"status": "drift_normal", "mean_fare": float(mean_price)}

def task_feature_engineering_etl(**kwargs):
    """Compute temporal and kinematic features for regression modeling"""
    import pandas as pd
    df = pd.read_csv('data/flights.csv', nrows=10000)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['speed'] = df['distance'] / (df['time'] + 0.001)
    
    os.makedirs('data/processed', exist_ok=True)
    processed_path = 'data/processed/flights_featured.parquet'
    df.to_parquet(processed_path)
    print(f"Features engineered and written to {processed_path}")
    return {"processed_path": processed_path}

def task_retrain_and_track_mlflow(**kwargs):
    """Train XGBoost regression model and record run telemetry to MLflow"""
    import pandas as pd
    import numpy as np
    import joblib
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score
    import mlflow

    df = pd.read_parquet('data/processed/flights_featured.parquet')
    features = ['from', 'to', 'flightType', 'agency', 'time', 'distance', 'speed', 'month', 'day', 'dayofweek', 'is_weekend']
    X = df[features]
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['time', 'distance', 'speed', 'month', 'day', 'dayofweek', 'is_weekend']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['from', 'to', 'flightType', 'agency'])
        ]
    )

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42))
    ])

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Voyage_Analytics_MLOps")

    with mlflow.start_run(run_name="Airflow_Scheduled_Retrain"):
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))

        mlflow.log_metrics({"airflow_rmse": rmse, "airflow_r2": r2})
        print(f"Model Retrained -> RMSE: {rmse:.2f}, R2: {r2:.4f}")

        # Quality Gate
        if r2 < 0.90:
            raise ValueError(f"Model failed quality gate. R2={r2:.4f} is below 0.90 threshold!")

        joblib.dump(pipe, "models/flight_price_model.joblib")
        print("Model promoted to production directory.")

    return {"rmse": rmse, "r2": r2, "status": "promoted"}

# Define the DAG
with DAG(
    dag_id='travel_flight_price_mlops_dag',
    default_args=default_args,
    description='Automated ETL, drift detection, and retraining for flight price regression',
    schedule_interval='@daily',
    catchup=False,
    tags=['mlops', 'travel', 'voyage', 'regression', 'xgboost'],
) as dag:

    validate_task = PythonOperator(
        task_id='validate_data_ingestion',
        python_callable=task_validate_ingestion,
    )

    drift_task = PythonOperator(
        task_id='detect_covariate_drift',
        python_callable=task_data_drift_check,
    )

    etl_task = PythonOperator(
        task_id='feature_engineering_etl',
        python_callable=task_feature_engineering_etl,
    )

    retrain_task = PythonOperator(
        task_id='retrain_and_register_model',
        python_callable=task_retrain_and_track_mlflow,
    )

    # DAG Dependency Pipeline
    validate_task >> drift_task >> etl_task >> retrain_task
