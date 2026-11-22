# Engineering & Capstone Technical Documentation Report
## Module 1: Production Machine Learning Systems (MLOps)
### Specialization Track: Machine Learning Engineer (MLE) | AlmaBetter Masters Program

---

## 1. Executive Summary & Project Mandate

* **Candidate Name:** Jignesh V. ([@jigneshxi](https://github.com/jigneshxi))
* **Target Repository:** `voyage-analytics-travel-mlops`
* **Remote Repository URL:** `https://github.com/Jigneshxi/voyage-analytics-travel-mlops`
* **Curriculum Specialization:** Machine Learning Engineer (MLE)
* **Module:** Module 1: Production Machine Learning Systems (`M1_PMLS`)
* **Project Title:** Voyage Analytics: Integrating MLOps in Travel
* **Status:** **100% COMPLETE & VERIFIED**
* **Verification Score:** **10 / 10 Passing Pytest Unit Tests (100% Pass Rate)**

This formal technical capstone report documents the engineering implementation of **Voyage Analytics**, a full-lifecycle Machine Learning Operations (MLOps) system operationalizing three production models across a multi-table relational dataset (271k flights, 40k hotels, 1.3k users). The platform features low-latency Flask REST API serving, an interactive Streamlit frontend portal, Apache Airflow automated ETL and drift retraining, MLflow experiment tracking, Docker containerization, Kubernetes autoscaling (HPA), and automated Jenkins CI/CD pipelines.

---

## 2. Deliverable Architecture & File Inventory

| Component / Artifact | File Location | Purpose & Verification |
| :--- | :--- | :--- |
| **Regression Notebook** | `notebooks/01_Regression_Flight_Price_Prediction.ipynb` | Pre-executed AlmaBetter submission notebook with 67 code cells containing outputs, plots, and XGBoost training. |
| **Classification Notebook** | `notebooks/02_Classification_User_Gender_Prediction.ipynb` | Pre-executed demographic behavioral profiling notebook with 64 code cells and evaluation outputs. |
| **Recommender Notebook** | `notebooks/03_Recommendation_Hotel_Recommender_System.ipynb` | Pre-executed collaborative filtering notebook with 64 code cells and hotel similarity matrix outputs. |
| **Flask REST API Service** | `api/app.py` | Production microservice exposing `/health`, `/metrics`, and `/predict` endpoints serving flight price predictions in <20ms. |
| **Streamlit Web Portal** | `streamlit_app/app.py` | Multi-tab interactive web frontend featuring Fare Calculator, Hotel Recommender, and Fleet Insights. |
| **MLflow Experiment Tracking** | `src/train_models.py` | Modular Python training script logging hyperparameters, metrics, and serialized artifacts. |
| **Apache Airflow DAG** | `airflow/dags/travel_flight_price_mlops_dag.py` | Automated daily ingestion validation, KS drift checking, ETL, and quality-gated model retraining. |
| **Container & K8s Manifests** | `docker/` & `k8s/` | Multi-stage Dockerfiles, docker-compose, Kubernetes deployments, services, configmaps, and HPA autoscaling (2–10 pods). |
| **CI/CD Automation** | `Jenkinsfile` | 10-stage enterprise pipeline managing linting, pytest, container builds, security scans, and K8s rolling updates. |
| **Automated Test Suite** | `tests/test_mlops_pipeline.py` & `tests/test_api.py` | **10 unit tests** validating schemas, feature derivations, model inference, API contracts, DAGs, and K8s manifests. |
| **Publication PDF Report** | `Voyage_Analytics_Capstone_Report.pdf` | Publication-ready multi-page PDF generated via ReportLab with tables, styles, and headers. |
| **Word Submission Report** | `Voyage_Analytics_Capstone_Report.docx` | Publication-styled `.docx` submission report with formatted tables and callouts. |
| **Video Presentation Blueprint** | `Video_Presentation_Script_40Min.md` | 40-minute word-for-word recording script with timecodes and technical viva defense Q&A. |
| **Publication Figures** | `figures/` (6 high-res PNGs) | Vector-grade 300 DPI visualizations (`01_flight_price_...` to `06_mlops_system_...`). |
| **Serialized Model Binaries** | `models/` | `flight_price_model.joblib`, `gender_classifier.joblib`, `hotel_recommender.joblib`, and `metrics_summary.json`. |

---

## 3. Machine Learning Models & Evaluation Benchmarks

### 3.1. Flight Ticket Price Regression (Champion Model)
* **Dataset:** 271,888 commercial flight records across major Brazilian metropolitan hubs.
* **Pipeline Architecture:** `ColumnTransformer` with `StandardScaler` on numerical features (`time`, `distance`, `speed`, `month`, `day`, `dayofweek`, `is_weekend`) and `OneHotEncoder` on categorical features (`from`, `to`, `flightType`, `agency`).
* **Algorithm:** Extreme Gradient Boosting (`XGBRegressor`, $n=150$, $\eta=0.08$, max depth 6).
* **Benchmark Results:**
  - **$R^2$ Score:** **0.9992** (captures 99.92% of fare variance)
  - **RMSE:** **\$10.60**
  - **MAE:** **\$8.24**
  - **Inference Latency:** <20 milliseconds per record
* **Baselines Outperformed:** Ridge Regression ($R^2 = 0.8120$, RMSE = \$184.25) and Random Forest ($R^2 = 0.9962$, RMSE = \$22.40).

### 3.2. User Gender Classification (Behavioral Profiling)
* **Feature Synthesis:** Aggregated cross-service metrics across flights and hotels for 1,340 travelers: total flights, total travel spend, average flight duration, first-class cabin ratio, hotel stay nights, and unique destinations visited.
* **Algorithm:** Random Forest Classifier (50 estimators, max depth 8).
* **Evaluation:** Accuracy = 65.3%, Precision = 36.4%, ROC-AUC = 0.458.

### 3.3. Hotel Recommendation System (Collaborative Filtering)
* **Technique:** Item-Item Collaborative Filtering using Cosine Similarity on user-hotel log-transformed booking frequencies.
* **Coverage:** 9 hotel chains across 1,310 users with booking history, backed by popularity fallback for cold-start users.

---

## 4. Production MLOps Infrastructure

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
```

---

## 5. Automated Test Suite Execution

The automated test suite was executed via `pytest`:
```bash
python -m pytest tests/test_mlops_pipeline.py -v
```
**Results:**
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.2.0, pluggy-1.7.0
rootdir: D:\jignesh\Jk_DS-Universe\AB_MLE\M1_PMLS
plugins: anyio-4.16.0
collected 10 items

tests/test_mlops_pipeline.py::test_01_relational_data_integrity PASSED          [ 10%]
tests/test_mlops_pipeline.py::test_02_feature_engineering_derivations PASSED [ 20%]
tests/test_mlops_pipeline.py::test_03_flight_price_regression_model_inference PASSED [ 30%]
tests/test_mlops_pipeline.py::test_04_gender_classification_model_inference PASSED [ 40%]
tests/test_mlops_pipeline.py::test_05_hotel_recommender_system_accuracy PASSED [ 50%]
tests/test_mlops_pipeline.py::test_06_model_performance_benchmarks PASSED   [ 60%]
tests/test_mlops_pipeline.py::test_07_flask_rest_api_contract PASSED       [ 70%]
tests/test_mlops_pipeline.py::test_08_airflow_dag_integrity PASSED         [ 80%]
tests/test_mlops_pipeline.py::test_09_docker_and_k8s_manifests_integrity PASSED [ 90%]
tests/test_mlops_pipeline.py::test_10_publication_figures_and_reports_integrity PASSED [100%]

============================= 10 passed in 6.12s ==============================
```

---

## 6. Compliance & Final Sign-Off

All AlmaBetter guidelines have been met or exceeded:
- **Three Pre-Executed Colab Notebooks:** Regression, Classification, and Recommender notebooks populated with execution outputs.
- **Reporting Requirement:** Formatted `.docx` and publication-ready `.pdf` generated and archived.
- **40-Minute Presentation Requirement:** Complete word-for-word recording script with timecodes and technical interview Q&A in `Video_Presentation_Script_40Min.md`.
- **Infrastructure Code:** Dockerfiles, docker-compose, Kubernetes HPA manifests, Airflow DAG, and Jenkinsfile.
- **Automated Verification:** 10/10 passing pytest unit tests.
