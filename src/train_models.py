"""
Voyage Analytics - Production Model Training and MLflow Pipeline
Trains:
1. Flight Price Regression Model (XGBoost & Random Forest)
2. User Gender Classification Model (Random Forest / Logistic Regression)
3. Hotel Recommendation Model (Collaborative + Content-Based Matrix Recommender)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
import mlflow
import mlflow.sklearn

os.makedirs("models", exist_ok=True)
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Voyage_Analytics_MLOps")

print("==================================================")
print("STAGE 1: TRAINING FLIGHT PRICE REGRESSION MODEL")
print("==================================================")

flights = pd.read_csv("data/flights.csv")
print(f"Loaded flights dataset with shape: {flights.shape}")

# Feature Engineering
flights['date'] = pd.to_datetime(flights['date'])
flights['month'] = flights['date'].dt.month
flights['day'] = flights['date'].dt.day
flights['dayofweek'] = flights['date'].dt.dayofweek
flights['is_weekend'] = flights['dayofweek'].isin([5, 6]).astype(int)
flights['route'] = flights['from'] + " -> " + flights['to']
flights['speed'] = flights['distance'] / (flights['time'] + 0.001)

# Sample a representative subset for ultra-fast high-quality training if dataset is huge (270k rows)
train_sample = flights.sample(n=min(60000, len(flights)), random_state=42)

features_reg = ['from', 'to', 'flightType', 'agency', 'time', 'distance', 'speed', 'month', 'day', 'dayofweek', 'is_weekend']
target_reg = 'price'

X_reg = train_sample[features_reg]
y_reg = train_sample[target_reg]

categorical_features = ['from', 'to', 'flightType', 'agency']
numerical_features = ['time', 'distance', 'speed', 'month', 'day', 'dayofweek', 'is_weekend']

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ]
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

with mlflow.start_run(run_name="Flight_Price_Regression_XGBoost"):
    reg_model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    reg_pipeline = Pipeline([
        ('preprocessor', preprocessor_reg),
        ('model', reg_model)
    ])

    reg_pipeline.fit(X_train_r, y_train_r)
    y_pred_r = reg_pipeline.predict(X_test_r)

    rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
    mae = mean_absolute_error(y_test_r, y_pred_r)
    r2 = r2_score(y_test_r, y_pred_r)

    print(f"Regression Results -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    mlflow.log_params({
        "model_type": "XGBRegressor",
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 6
    })
    mlflow.log_metrics({
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2
    })
    mlflow.sklearn.log_model(reg_pipeline, "flight_price_pipeline", serialization_format="pickle")

    joblib.dump(reg_pipeline, "models/flight_price_model.joblib")
    print("Saved 'models/flight_price_model.joblib'")


print("\n==================================================")
print("STAGE 2: TRAINING USER GENDER CLASSIFICATION MODEL")
print("==================================================")

users = pd.read_csv("data/users.csv")
hotels = pd.read_csv("data/hotels.csv")

# Aggregate flight patterns per user
flight_user_agg = flights.groupby('userCode').agg(
    flight_count=('travelCode', 'count'),
    flight_total_spend=('price', 'sum'),
    flight_avg_spend=('price', 'mean'),
    flight_avg_duration=('time', 'mean'),
    first_class_ratio=('flightType', lambda s: (s == 'firstClass').mean()),
    rainbow_agency_ratio=('agency', lambda s: (s == 'Rainbow').mean())
).reset_index()

# Aggregate hotel patterns per user
hotel_user_agg = hotels.groupby('userCode').agg(
    hotel_count=('travelCode', 'count'),
    hotel_total_spend=('total', 'sum'),
    hotel_avg_spend=('total', 'mean'),
    hotel_total_days=('days', 'sum'),
    unique_places_visited=('place', 'nunique')
).reset_index()

# Merge user data with travel profile
user_profile = users.merge(flight_user_agg, left_on='code', right_on='userCode', how='left')
user_profile = user_profile.merge(hotel_user_agg, left_on='code', right_on='userCode', how='left')
user_profile = user_profile.fillna(0)

# Target: gender (male = 1, female = 0)
user_profile['target'] = (user_profile['gender'].str.lower() == 'male').astype(int)

features_clf_num = [
    'age', 'flight_count', 'flight_total_spend', 'flight_avg_spend', 'flight_avg_duration',
    'first_class_ratio', 'hotel_count', 'hotel_total_spend', 'hotel_avg_spend',
    'hotel_total_days', 'unique_places_visited'
]
features_clf_cat = ['company']

X_clf = user_profile[features_clf_num + features_clf_cat]
y_clf = user_profile['target']

preprocessor_clf = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features_clf_num),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_clf_cat)
    ]
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

with mlflow.start_run(run_name="User_Gender_Classification_RF"):
    clf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=4,
        random_state=42
    )

    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor_clf),
        ('model', clf_model)
    ])

    clf_pipeline.fit(X_train_c, y_train_c)
    y_pred_c = clf_pipeline.predict(X_test_c)
    y_prob_c = clf_pipeline.predict_proba(X_test_c)[:, 1]

    acc = accuracy_score(y_test_c, y_pred_c)
    prec = precision_score(y_test_c, y_pred_c, zero_division=0)
    rec = recall_score(y_test_c, y_pred_c, zero_division=0)
    f1 = f1_score(y_test_c, y_pred_c, zero_division=0)
    roc_auc = roc_auc_score(y_test_c, y_prob_c)

    print(f"Classification Results -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 8
    })
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc
    })
    mlflow.sklearn.log_model(clf_pipeline, "user_gender_pipeline", serialization_format="pickle")

    joblib.dump(clf_pipeline, "models/gender_classifier.joblib")
    print("Saved 'models/gender_classifier.joblib'")


print("\n==================================================")
print("STAGE 3: BUILDING HOTEL RECOMMENDATION SYSTEM")
print("==================================================")

# Build User-Item Interaction Matrix from hotel bookings
hotel_ratings = hotels.groupby(['userCode', 'name']).agg(
    stay_count=('travelCode', 'count'),
    total_days=('days', 'sum'),
    avg_price=('price', 'mean'),
    total_spent=('total', 'sum')
).reset_index()

# Define an implicit preference score based on stay frequency and duration
hotel_ratings['preference_score'] = np.log1p(hotel_ratings['stay_count'] * 2 + hotel_ratings['total_days'])

# User-Item Matrix
user_hotel_matrix = hotel_ratings.pivot_table(index='userCode', columns='name', values='preference_score', fill_value=0)

# Item-Item Cosine Similarity for Collaborative Filtering
item_similarity = cosine_similarity(user_hotel_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_hotel_matrix.columns, columns=user_hotel_matrix.columns)

# Hotel metadata catalog for content/popularity fallback
hotel_catalog = hotels.groupby('name').agg(
    popular_place=('place', lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]),
    avg_daily_price=('price', 'mean'),
    total_bookings=('travelCode', 'count')
).reset_index()

recommender_bundle = {
    'user_hotel_matrix': user_hotel_matrix,
    'item_similarity_df': item_similarity_df,
    'hotel_catalog': hotel_catalog
}

joblib.dump(recommender_bundle, "models/hotel_recommender.joblib")
print("Saved 'models/hotel_recommender.joblib'")

# Save Metrics Summary
metrics_summary = {
    "regression": {
        "model": "XGBRegressor",
        "rmse": float(rmse),
        "mae": float(mae),
        "r2_score": float(r2)
    },
    "classification": {
        "model": "RandomForestClassifier",
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    },
    "recommendation": {
        "model": "Collaborative_Item_Similarity_Matrix",
        "unique_hotels": len(hotel_catalog),
        "unique_users_with_history": len(user_hotel_matrix)
    }
}

with open("models/metrics_summary.json", "w") as f:
    json.dump(metrics_summary, f, indent=4)
print("Saved 'models/metrics_summary.json'")

print("\nALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
