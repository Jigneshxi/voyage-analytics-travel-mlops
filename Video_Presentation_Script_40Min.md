# 🎬 Voyage Analytics: 40-Minute Comprehensive Video Presentation Script

**Project Title:** Voyage Analytics: Integrating MLOps in Travel  
**Presenter:** Jignesh (Individual Capstone Presentation)  
**Primary Model Focus:** Flight Price Prediction (Regression Model) & Full MLOps Lifecycle  
**Total Target Duration:** 40 to 45 Minutes  

---

## ⏱️ Video Time Allocation & Structure Overview

| Part | Topic | Target Time | Slide / Screen Display |
| :---: | :--- | :---: | :--- |
| **Part 1** | **Executive Introduction & System Architecture** | 00:00 – 03:00 (3 min) | Architecture Diagram & Tech Stack Slide |
| **Part 2** | **Business Context & Problem Formulation** | 03:00 – 06:30 (3.5 min) | Business Context & Production Value Slide |
| **Part 3** | **Data Understanding & Multi-Dataset Relationships** | 06:30 – 10:30 (4 min) | Schema mapping (`flights`, `hotels`, `users`) |
| **Part 4** | **Exploratory Data Analysis & Feature Engineering** | 10:30 – 17:00 (6.5 min) | Regression Colab Notebook & Seaborn charts |
| **Part 5** | **Regression Model Training, Tuning & MLflow Tracking** | 17:00 – 24:00 (7 min) | XGBoost Training, Hyperparameters & MLflow UI |
| **Part 6** | **Real-Time Serving Architecture: Flask REST API** | 24:00 – 29:00 (5 min) | `api/app.py` walkthrough, cURL & Postman demo |
| **Part 7** | **Containerization & Kubernetes Scaling (Docker + K8s HPA)** | 29:00 – 34:00 (5 min) | `Dockerfile`, `deployment.yaml`, `kubectl` commands |
| **Part 8** | **Workflow Automation with Apache Airflow & Jenkins CI/CD** | 34:00 – 39:00 (5 min) | Airflow DAG Graph view & Jenkins Stage view |
| **Part 9** | **Streamlit App, Monitoring, Reliability & Q&A Defenses** | 39:00 – 43:30 (4.5 min) | Live Streamlit portal & Follow-Up Q&A |

---

## 🎙️ Spoken Presentation Script (Verbatim Talking Points)

### Part 1: Executive Introduction & System Architecture (00:00 – 03:00)

**[Screen: Slide 1 - Title Slide & High-Level System Architecture Mermaid Diagram]**

> "Hello everyone, respected evaluators, and mentors at AlmaBetter. My name is Jignesh, and today I am excited to present my End Course Summative Capstone Project: **'Voyage Analytics: Integrating MLOps in Travel'**.
>
> In today’s presentation, I will walk you through an enterprise-grade, end-to-end Machine Learning Operations system designed to tackle a fundamental challenge in the travel and tourism industry: dynamic pricing and personalized travel curation.
>
> While our solution develops three distinct machine learning systems—namely, a flight price regression model, a user gender classification model, and an intelligent hotel recommender system—in strict accordance with the capstone guidelines, my presentation will focus deeply on our core engine: **The Flight Price Regression Model and its comprehensive MLOps lifecycle**.
>
> Our production technology stack spans:
> - **Modeling & Analytics:** Python, Pandas, NumPy, Scikit-Learn, and XGBoost.
> - **Model Tracking & Registry:** MLflow for experiment tracking, metric comparisons, and artifact storage.
> - **Inference Serving:** Flask REST API engineered with schema validation and sub-20ms inference latency.
> - **Interactive Frontend:** Streamlit for travel recommendations, price forecasting, and telemetry insights.
> - **Containerization:** Docker multi-stage builds and Docker Compose for portable service orchestration.
> - **Container Orchestration & Scaling:** Kubernetes (K8s) Deployments with Horizontal Pod Autoscalers (HPA).
> - **Workflow Scheduling & ETL:** Apache Airflow Directed Acyclic Graphs (DAGs) for automated ingestion, drift detection, and retraining.
> - **Continuous Integration & Deployment:** Jenkins CI/CD automation managing linting, unit testing, image builds, and automated rolling cluster rollouts.
>
> Let us now delve into the business context and problem formulation."

---

### Part 2: Business Context & Problem Formulation (03:00 – 06:30)

**[Screen: Slide 2 - Problem Statement & Productionization Imperative]**

> "Why is machine learning essential in the travel and tourism industry?
>
> Airline fare pricing is inherently complex and volatile. Ticket prices are not static; they fluctuate continuously based on origin-destination distances, cabin class tiers—such as economic versus firstClass—seasonal shifts, days of travel, and carrier agency competition. For both corporate clients managing business travel budgets and retail travelers looking for fair pricing, unpredictable fares lead to budget overruns and discovery friction.
>
> However, building an ML model inside a Jupyter Notebook alone does not generate business ROI. A notebook is an isolated sandbox. In real-world enterprise operations, several critical challenges arise:
> 1. **Serving in Real Time:** Upstream booking portals require instant price estimation via high-concurrency REST endpoints.
> 2. **Environment Portability:** Code that works on a data scientist’s laptop often fails in production due to dependency conflicts. We need containerization.
> 3. **Scalability Under Traffic Spikes:** Travel search volume spikes exponentially during holidays. The system must autoscale seamlessly without crashing.
> 4. **Data Drift & Model Decay:** Airline pricing strategies evolve over time. Models degrade without automated retraining workflows.
> 5. **Deployment Reliability:** Deploying code manually leads to human error and outages. Automated CI/CD pipelines are mandatory to ensure that only tested, validated code reaches production.
>
> This capstone bridges the gap between machine learning research and enterprise productionization by implementing the complete MLOps lifecycle."

---

### Part 3: Data Understanding & Multi-Dataset Relationships (06:30 – 10:30)

**[Screen: Slide 3 - Entity Relationship Diagram & Data Dictionaries]**

> "Let us look at the data foundation of Voyage Analytics. Our system is built upon three relational datasets totaling over 310,000 transaction records:
>
> 1. **The Flights Dataset (`flights.csv`):**
>    - Contains 271,888 transaction rows.
>    - Core attributes include `travelCode`, `userCode`, `from`, `to`, `flightType`, `price` (our regression target), `time` (flight duration in hours), `distance` (in kilometers), `agency`, and `date`.
>    - Notably, the dataset exhibits zero missing values across all 271,888 rows, providing a clean foundation for modeling.
>
> 2. **The Hotels Dataset (`hotels.csv`):**
>    - Contains 40,552 records of accommodations.
>    - Attributes include `travelCode`, `userCode`, hotel `name`, `place`, `days` of stay, daily `price`, `total` booking price, and `date`.
>
> 3. **The Users Dataset (`users.csv`):**
>    - Contains 1,340 unique traveler profiles.
>    - Fields include user `code`, associated corporate `company`, traveler `name`, `gender`, and `age`.
>
> **Relational Architecture:**
> Notice how these datasets interlink:
> - `userCode` connects the demographic profiles in `users.csv` to both flight itineraries and hotel reservations, allowing us to construct rich behavioral aggregations for classification.
> - `travelCode` binds specific flight journeys with corresponding hotel stays for integrated multi-modal itinerary recommendations.
>
> For our flight price regression model, each flight transaction provides the complete set of geographic, temporal, and carrier features necessary to forecast ticket fares."

---

### Part 4: Exploratory Data Analysis & Feature Engineering (10:30 – 17:00)

**[Screen: Transition to `01_Regression_Flight_Price_Prediction.ipynb` in Colab / VS Code]**

> "Now, let us examine the exploratory data analysis and feature engineering pipeline implemented in our regression notebook.
>
> **Key EDA Findings:**
> 1. **Price Distribution:** Flight prices range from approximately $85 up to $1,800, with a right-skewed multi-modal distribution reflecting clear pricing tiers.
> 2. **Impact of Cabin Class (`flightType`):** This is by far the most influential categorical feature. First class tickets command a significant price premium compared to economic fares for identical route distances.
> 3. **Distance & Time Correlation:** Route distance and duration exhibit a strong positive correlation (Pearson r > 0.85). Fares scale predictably with route distance, but with distinct slope offsets depending on the agency and cabin tier.
> 4. **Agency Pricing Strategies:** Agencies such as 'FlyingDrops', 'CloudFy', and 'Rainbow' display unique pricing structures across Brazilian metropolitan corridors (São Paulo, Brasilia, Recife, Florianopolis, Aracaju, Salvador).
>
> **Feature Engineering Pipeline:**
> Raw timestamps and raw coordinates alone do not provide optimal signals for gradient boosting. We engineered several domain-specific features:
> - **Temporal Extraction:** Decomposed the `date` string into `month`, `day`, `dayofweek`, and a binary `is_weekend` flag (identifying Saturday and Sunday travel).
> - **Kinematic Speed Feature:** We computed `speed = distance / (time + 0.001)`. This derived feature captures aircraft cruising velocity, effectively distinguishing short commuter flights from long-haul trunk routes.
> - **Categorical Encoding:** We employed `ColumnTransformer` with `StandardScaler` for continuous numerical features and `OneHotEncoder(handle_unknown='ignore')` for categorical features (`from`, `to`, `flightType`, `agency`)."

---

### Part 5: Regression Model Training, Tuning & MLflow Tracking (17:00 – 24:00)

**[Screen: Transition to MLflow Tracking UI / `train_models.py` code]**

> "Let us discuss the modeling strategy and performance benchmarks.
>
> We evaluated three progressive regression architectures:
> 1. **Model 1: Ridge Regression (L2 Regularized Linear Model):** Served as our baseline. It yielded an R² score of 0.8120 and an RMSE of $184.25. While linear regression captured the basic distance-to-price relationship, it struggled with non-linear interaction terms between cabin class and flight agencies.
> 2. **Model 2: Random Forest Regressor:** An ensemble of 50 decision trees with a max depth of 12. This achieved a dramatic performance increase, reaching an R² of 0.9962 and an RMSE of $22.40.
> 3. **Model 3: Extreme Gradient Boosting (XGBoost Regressor - Champion Model):**
>    - Configured with `n_estimators=150`, `learning_rate=0.08`, `max_depth=6`, `subsample=0.8`, and `colsample_bytree=0.8`.
>    - The XGBoost Regressor achieved exceptional predictive accuracy:
>      - **R² Score:** `0.9992` (explaining 99.92% of the variance in flight prices)
>      - **Root Mean Squared Error (RMSE):** `$10.60`
>      - **Mean Absolute Error (MAE):** `$8.24`
>
> **MLflow Experiment Tracking Integration:**
> In production ML, reproducibility is non-negotiable. Using `mlflow.set_tracking_uri("sqlite:///mlflow.db")`, every training run automatically logged:
> - Hyperparameters (`n_estimators`, `learning_rate`, `max_depth`)
> - Evaluation metrics (`rmse`, `mae`, `r2_score`)
> - The serialized Scikit-Learn/XGBoost pipeline artifact.
>
> As you can see in the MLflow dashboard, we have an immutable audit trail of runs, enabling seamless comparison and model rollback if needed."

---

### Part 6: Real-Time Serving Architecture: Flask REST API (24:00 – 29:00)

**[Screen: Show `api/app.py` in editor, start server, execute Postman/Curl request]**

> "With our champion model trained and validated, we operationalized it into a production Flask REST API located in `api/app.py`.
>
> **API Architecture Highlights:**
> 1. **Model Decoupling & Fast Cold-Start:** The application initializes the serialized model pipeline once during startup using `joblib.load()`, caching it in memory.
> 2. **Feature Imputation & Schema Validation:** The `/predict` endpoint accepts a JSON payload representing a flight itinerary. Even if upstream clients provide minimal fields, our `preprocess_flight_input()` utility derives temporal attributes (`month`, `day`, `dayofweek`, `is_weekend`) and cruising `speed`.
> 3. **Single & Batch Prediction Support:** The endpoint automatically differentiates between a single flight object and a batch list of flights, returning vectorized predictions.
> 4. **Production Health & Telemetry:**
>    - `GET /health` provides a liveness check for Kubernetes probes.
>    - `GET /metrics` surfaces the registered model version and historical validation scores.
>
> **Live Demonstration:**
> Let us send a POST request with an itinerary from Recife to Florianopolis. Within 12 milliseconds, the API responds with HTTP 200 OK and a predicted fare of $698.42 USD. If we adjust the `flightType` to `firstClass`, the model dynamically recalculates the premium fare."

---

### Part 7: Containerization & Kubernetes Scaling (29:00 – 34:00)

**[Screen: Show `Dockerfile.flask`, `deployment.yaml`, `hpa.yaml`, terminal running `kubectl`]**

> "Now, let us discuss how Voyage Analytics achieves environment isolation and high scalability.
>
> **Containerization with Docker:**
> In `docker/Dockerfile.flask`, we utilize the lightweight `python:3.12-slim` base image. We install only production dependencies, copy the serialized model weights and API code, and configure a container-native `HEALTHCHECK` that queries `/health` every 30 seconds.
>
> **Kubernetes (K8s) Cluster Orchestration:**
> When deploying to Kubernetes, we configured three core manifests in `k8s/`:
> 1. **Deployment (`deployment.yaml`):**
>    - Manages pod replicas with a `RollingUpdate` strategy (`maxSurge: 1`, `maxUnavailable: 0`), guaranteeing zero-downtime updates during new model deployments.
>    - Configures rigorous `livenessProbe` and `readinessProbe` checks so traffic is only routed to healthy pods.
>    - Enforces explicit resource governance with CPU requests of 250m and limits of 1000m.
> 2. **Service (`service.yaml`):**
>    - Exposes a `LoadBalancer` service distributing external HTTP traffic across healthy container pods.
> 3. **Horizontal Pod Autoscaler (`hpa.yaml`):**
>    - In real travel scenarios, flash sales or holiday booking surges drive massive traffic spikes.
>    - Our HPA dynamically scales pod replicas between a minimum of 2 and a maximum of 10 based on real-time CPU utilization (target 75%) and memory utilization (target 80%)."

---

### Part 8: Workflow Automation with Apache Airflow & Jenkins CI/CD (34:00 – 39:00)

**[Screen: Show Airflow Web UI & Jenkins Pipeline View]**

> "Next, let us explore our pipeline automation components: Apache Airflow and Jenkins.
>
> **Workflow Orchestration with Apache Airflow:**
> In `airflow/dags/travel_flight_price_mlops_dag.py`, we created an automated DAG executing on a `@daily` schedule:
> - **Task 1: `validate_data_ingestion`:** Verifies incoming flight data streams for schema completeness and null thresholds.
> - **Task 2: `detect_covariate_drift`:** Monitors mean fare shifts and flags anomalies before downstream training.
> - **Task 3: `feature_engineering_etl`:** Executes automated feature extraction and outputs optimized Parquet tables.
> - **Task 4: `retrain_and_register_model`:** Retrains the XGBoost model, logs metrics to MLflow, and enforces an automated **Quality Gate**: if holdout R² drops below 0.90, the task aborts and alerts engineering, preventing degraded models from reaching production.
>
> **Automated CI/CD with Jenkins:**
> Our `Jenkinsfile` enforces an end-to-end continuous integration and deployment pipeline across 10 stages:
> 1. Code Checkout from GitHub.
> 2. Virtual Environment Setup.
> 3. Code Linting with `flake8`.
> 4. Automated Testing via `pytest` (verifying all API endpoints and batch handling).
> 5. Model Serialization Validation.
> 6. Docker Container Packaging.
> 7. Vulnerability Scanning with Trivy.
> 8. Pushing to Docker Registry.
> 9. Kubernetes Rolling Rollout via `kubectl`.
> 10. Post-Deployment Live Smoke Testing."

---

### Part 9: Streamlit App, Monitoring, Reliability & Q&A Defenses (39:00 – 43:30)

**[Screen: Show Streamlit Web Application running live on localhost:8501]**

> "To complete the travel experience, we developed an interactive Streamlit web application:
> - **Hotel Recommender Page:** Uses collaborative filtering to suggest hotels based on user booking history, with cold-start fallback.
> - **Flight Price Predictor:** Gives non-technical users an intuitive GUI to test flight fare estimations in real time.
> - **Analytics & Fleet Insights:** Displays fare distributions and hotel occupancy trends.
> - **MLOps Health Dashboard:** Displays container statuses, Airflow schedules, and K8s replica metrics.
>
> **Reliability Considerations & Key Learnings:**
> 1. **Production ML vs. Notebook Modeling:** Building a notebook model is only 20% of the battle. The remaining 80% is software engineering: input validation, packaging, orchestration, automated testing, and zero-downtime serving.
> 2. **Reproducibility:** Versioning both data and model artifacts with MLflow guarantees full auditability.
>
> **Addressing Common Evaluator Questions:**
> - *Q: How does the system handle real-time prediction spikes?*
>   *A:* Through our Kubernetes HPA, which automatically scales pods up to 10 instances when CPU usage crosses 75%.
> - *Q: How do we prevent serving a degraded retrained model?*
>   *A:* Through Airflow's automated Quality Gate, which requires holdout R² >= 0.90 before model artifacts are promoted to the production directory.
>
> Thank you so much for your time and guidance. I invite any questions regarding the architecture and implementation!"

---

## 📋 Evaluator Follow-Up Q&A Cheat Sheet

| Question | Core Concept | Model Answer |
| :--- | :--- | :--- |
| **1. How would you productionize an ML model for real-time travel pricing?** | Serving & Latency | Pre-compile transformations into a Scikit-Learn/XGBoost Pipeline, load into memory at Flask server startup, serve behind an async WSGI server (Gunicorn) with horizontal pod autoscaling and Redis caching for frequent routes. |
| **2. Why did you use Docker and Kubernetes in this project?** | Portability & Resilience | Docker eliminates 'works on my machine' dependency drifts. Kubernetes provides automated self-healing, rolling zero-downtime deployments, service discovery, and elastic horizontal pod autoscaling under dynamic load. |
| **3. How does MLflow help in managing model versions?** | Experiment Tracking & Governance | MLflow logs parameters, evaluation metrics, and artifacts to an immutable backend (SQLite/PostgreSQL), allowing engineering to track lineage, compare runs, and manage stage transitions (Staging -> Production -> Archived). |
| **4. How would you handle model retraining in production?** | Orchestration & Quality Gates | Automate through Apache Airflow DAGs triggered on schedule or drift alerts. Enforce automated evaluation gates (e.g., R² > 0.90) and A/B canary routing before fully promoting the new model artifact. |
| **5. How would you scale this system for a large travel platform?** | Enterprise Scalability | Deploy on managed Kubernetes (EKS/GKE), decouple incoming prediction requests using Apache Kafka message streaming, cache route predictions in Redis, and store telemetry in Prometheus/Grafana. |
