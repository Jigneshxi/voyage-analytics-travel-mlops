# Voyage Analytics: Production MLOps Workflow Documentation

**Project Title:** Voyage Analytics: Integrating MLOps in Travel  
**Author / Contributor:** Jignesh (Individual Submission)  
**Evaluator Access:** evaluator@almabetter.com  
**GitHub Repository:** [https://github.com/jignesh-voyage/voyage-analytics-mlops](https://github.com/jignesh-voyage/voyage-analytics-mlops)  

---

## Executive Summary
This document provides an end-to-end technical walkthrough of the Machine Learning Operations (MLOps) architecture engineered for **Voyage Analytics**. The system operationalizes three machine learning models:
1. **Flight Ticket Price Prediction** (High-precision XGBoost Regressor; R² = 0.9992, RMSE = $10.60)
2. **User Gender Demographics Classifier** (Behavioral Profiling Random Forest Classifier)
3. **Hotel Recommendation System** (Item-Item Collaborative Filtering + Popularity Fallback)

The infrastructure delivers low-latency model inference via a Flask REST API, interactive user exploration via Streamlit, automated workflow scheduling using Apache Airflow, automated CI/CD using Jenkins, experiment lifecycle tracking with MLflow, and scalable container orchestration via Docker and Kubernetes.

---

## System Architecture Diagram

```
+-----------------------------------------------------------------------------------+
|                              DATA & ORCHESTRATION                                 |
|  [flights.csv / hotels.csv / users.csv]                                            |
|                                |                                                  |
|                                v                                                  |
|                  +----------------------------+                                   |
|                  |     Apache Airflow DAG     |                                   |
|                  | (Ingestion -> Drift -> ETL)|                                   |
|                  +--------------+-------------+                                   |
|                                 |                                                 |
+---------------------------------|-------------------------------------------------+
                                  v
+-----------------------------------------------------------------------------------+
|                        MODEL REGISTRY & EXPERIMENT TRACKING                       |
|                  +----------------------------+                                   |
|                  |      MLflow Tracking       |                                   |
|                  | (Params, Metrics, Models)  |                                   |
|                  +--------------+-------------+                                   |
+---------------------------------|-------------------------------------------------+
                                  v
+-----------------------------------------------------------------------------------+
|                            SERVING & PRESENTATION LAYER                           |
|                  +----------------------------+                                   |
|                  |      Docker Container      |                                   |
|                  +--------------+-------------+                                   |
|                                 |                                                 |
|                 +---------------+---------------+                                 |
|                 v                               v                                 |
|     +-----------------------+       +------------------------+                    |
|     |    Flask REST API     |       |   Streamlit Frontend   |                    |
|     |  (:5000 /predict)     |       |  (:8501 Web UI)        |                    |
|     +-----------+-----------+       +------------------------+                    |
|                 |                                                                 |
|                 v                                                                 |
|     +-----------------------+                                                     |
|     |  Kubernetes (K8s)     |                                                     |
|     |  Deployment + HPA     |                                                     |
|     +-----------------------+                                                     |
+-----------------------------------------------------------------------------------+
                                  ^
+---------------------------------|-------------------------------------------------+
|                               CI / CD                                             |
|                  +----------------------------+                                   |
|                  |    Jenkins Pipeline        |                                   |
|                  | (Lint -> Test -> Build     |                                   |
|                  |  -> Scan -> K8s Rollout)   |                                   |
|                  +----------------------------+                                   |
+-----------------------------------------------------------------------------------+
```

---

## Stage 1: Flask REST API Architecture & Inference Serving

### 1.1 Overview
The real-time inference service is implemented in Flask (`api/app.py`). It loads the serialized XGBoost regression pipeline (`models/flight_price_model.joblib`), validates incoming payloads, derives temporal and kinematic features (`speed`, `dayofweek`, `is_weekend`), and outputs predicted fares with sub-20ms latency.

### 1.2 Endpoints Specification
- `GET /health`: Health check endpoint verifying model state and service status.
- `GET /metrics`: Returns baseline model performance metrics from `models/metrics_summary.json`.
- `POST /predict`: Accepts single flight objects or batch flight arrays.

### 1.3 Sample API Request & Response
**Request (POST /predict):**
```json
{
  "from": "Recife (PE)",
  "to": "Florianopolis (SC)",
  "flightType": "economic",
  "agency": "FlyingDrops",
  "time": 1.7,
  "distance": 676.53,
  "date": "2026-10-25"
}
```

**Response (HTTP 200 OK):**
```json
{
  "status": "success",
  "count": 1,
  "predictions": 698.42,
  "currency": "USD",
  "model_version": "v1.0-xgb"
}
```

> **[SCREENSHOT 1: Flask REST API Execution & Postman/Curl Test]**  
> *Insert screenshot showing terminal starting Flask API on port 5000 and successful cURL / Postman POST request with JSON response.*

---

## Stage 2: Interactive Streamlit Web Application

### 2.1 Overview
The Streamlit application (`streamlit_app/app.py`) provides an intuitive web interface for business users, travelers, and operations teams.

### 2.2 Core Modules
1. **Intelligent Hotel Recommender:**
   - Selects traveler profile from `users.csv`.
   - Computes weighted item-item similarity based on past bookings, stay duration, and expenditure.
   - Provides instant recommendations with average daily rate and location.
   - Implements cold-start fallback for travelers without historical transactions.
2. **Flight Price Predictor:**
   - Interactive widgets for origin, destination, cabin class, agency, date, and flight duration.
   - Generates real-time price estimation from the trained XGBoost model.
3. **Fleet Analytics & Insights:**
   - High-level distributions of fares across cabin tiers and hotel occupancy rates.
4. **MLOps System Health:**
   - Live telemetry status for Docker containers, Airflow DAG schedules, and K8s replicas.

> **[SCREENSHOT 2: Streamlit Hotel Recommender Interface]**  
> *Insert screenshot showing the Streamlit UI with the Hotel Recommendations page, selected user profile, and generated recommendation cards.*

> **[SCREENSHOT 3: Streamlit Flight Price Calculator]**  
> *Insert screenshot showing the flight fare prediction input form and calculated estimated fare.*

---

## Stage 3: Docker Containerization & Multi-Container Composition

### 3.1 Overview
All components are containerized using minimal Debian-slim images (`python:3.12-slim`) to eliminate environmental discrepancies across development, staging, and production.

### 3.2 Docker Artifacts
- `docker/Dockerfile.flask`: Packages the REST API, requirements, and serialized model artifacts. Includes built-in curl health checks.
- `docker/Dockerfile.streamlit`: Packages the web frontend and interactive datasets.
- `docker/docker-compose.yml`: Multi-container configuration orchestrating the Flask API, Streamlit web app, and MLflow Tracking server.

### 3.3 Container Execution Commands
```bash
# Build and start services
docker-compose -f docker/docker-compose.yml up --build -d

# Verify running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

> **[SCREENSHOT 4: Docker Build & Docker PS Status]**  
> *Insert screenshot showing successful `docker-compose up` output and `docker ps` showing healthy containers.*

---

## Stage 4: Kubernetes Deployment & Autoscaling (HPA)

### 4.1 Overview
Production-grade deployment is orchestrated via Kubernetes manifests located in `k8s/`.

### 4.2 Manifest Specifications
1. **Deployment (`k8s/deployment.yaml`):**
   - 3 initial replicas.
   - Rolling update strategy with `maxSurge: 1` and `maxUnavailable: 0` for zero-downtime rollouts.
   - Liveness and readiness probes checking `/health`.
   - Resource requests (250m CPU, 512Mi RAM) and limits (1000m CPU, 1024Mi RAM).
2. **Service (`k8s/service.yaml`):**
   - LoadBalancer service exposing port 80 routing to pod target port 5000.
3. **Horizontal Pod Autoscaler (`k8s/hpa.yaml`):**
   - Scales automatically between 2 and 10 pods when CPU utilization exceeds 75% or memory exceeds 80%.

### 4.3 Cluster Rollout Commands
```bash
# Apply configurations
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Monitor pods and autoscaler
kubectl get pods -l app=voyage-flight-price-api
kubectl get hpa voyage-flight-price-hpa
```

> **[SCREENSHOT 5: Kubernetes Pods, Service & HPA Verification]**  
> *Insert screenshot showing `kubectl get pods`, `kubectl get svc`, and `kubectl get hpa` with healthy replica sets.*

---

## Stage 5: Apache Airflow Workflow Scheduling & Retraining DAG

### 5.1 Overview
The DAG `travel_flight_price_mlops_dag` (`airflow/dags/travel_flight_price_mlops_dag.py`) runs on a `@daily` schedule to automate data ingestion, covariate drift monitoring, ETL feature generation, model retraining, and promotion gates.

### 5.2 Directed Acyclic Graph Structure
```
[validate_data_ingestion] 
          |
          v
[detect_covariate_drift]
          |
          v
[feature_engineering_etl]
          |
          v
[retrain_and_register_model]
```

### 5.3 Automated Quality Gate
During the `retrain_and_register_model` task, the retrained model must achieve an R² score of at least 0.90 on holdout validation data. If this threshold is breached (indicating model degradation or data contamination), an exception halts the DAG, triggering an alert and preventing promotion to production.

> **[SCREENSHOT 6: Airflow Webserver DAG Graph View & Tree View]**  
> *Insert screenshot showing the Airflow DAG `travel_flight_price_mlops_dag` with all four green success tasks in the Graph View.*

---

## Stage 6: Jenkins CI/CD Automation Pipeline

### 6.1 Overview
Continuous Integration and Continuous Deployment are managed by `Jenkinsfile` triggered on repository push events.

### 6.2 Pipeline Stages
1. **Checkout Code:** Pulls latest branch commit from GitHub.
2. **Environment Setup:** Creates Python virtual environment and installs pinned dependencies from `requirements.txt`.
3. **Linting & Code Quality:** Executes `flake8` for syntax and PEP8 compliance.
4. **Automated Unit Testing:** Runs `pytest tests/` verifying endpoints, batch handling, and model deserialization.
5. **Model Artifact Integrity:** Asserts model weights and pipelines exist and evaluate test vectors.
6. **Docker Build:** Builds versioned and latest Docker images (`flight-price-api:${BUILD_NUMBER}`).
7. **Security Scan:** Executes Trivy container vulnerability scanner.
8. **Registry Push:** Pushes built artifact to Docker Hub / registry.
9. **Kubernetes Rollout:** Performs zero-downtime rolling update via `kubectl set image`.
10. **Post-Deployment Smoke Test:** Queries `/health` endpoint to confirm live cluster readiness.

> **[SCREENSHOT 7: Jenkins Pipeline Stage View]**  
> *Insert screenshot showing all 10 stages green in the Jenkins blue ocean / classic pipeline stage view.*

---

## Stage 7: MLflow Experiment Tracking & Model Registry

### 7.1 Overview
MLflow tracks training hyperparameters, cross-validation metrics, and model artifacts using SQLite backend (`sqlite:///mlflow.db`).

### 7.2 Logged Experiments
- **Experiment Name:** `Voyage_Analytics_MLOps`
- **Logged Runs:**
  1. `Flight_Price_Regression_XGBoost`
     - Parameters: `n_estimators=150`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`
     - Metrics: `rmse=10.60`, `mae=8.24`, `r2_score=0.9992`
     - Artifacts: Complete Scikit-Learn / XGBoost pipeline
  2. `User_Gender_Classification_RF`
     - Parameters: `n_estimators=100`, `max_depth=8`
     - Metrics: `accuracy=0.6530`, `precision=0.3636`, `recall=0.0444`, `f1_score=0.0792`
     - Artifacts: Random Forest Classification pipeline

### 7.3 Launching MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

> **[SCREENSHOT 8: MLflow Experiments Dashboard]**  
> *Insert screenshot of MLflow UI showing the `Voyage_Analytics_MLOps` experiment with runs, parameters, and logged RMSE/R2 metrics.*

---

## Conclusion & Production Verification
All seven components of the Voyage Analytics MLOps architecture have been implemented, verified with automated unit tests (`pytest 5 passed`), containerized, and documented. The codebase is organized within a single version-controlled repository ready for evaluation.
