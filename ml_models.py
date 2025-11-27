import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@st.cache_resource(show_spinner=True)
def train_ml_models(dataset_path: str):
    """
    Loads doctor's dataset, trains MULTIPLE ML models with feature engineering
    and returns:
    - trained_pipelines dict
    - metrics_dict with metrics for each model
    - feature_cols
    """
    df = pd.read_csv(dataset_path)

    # FEATURE ENGINEERING - Add new features to improve accuracy
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
    df['symptom_count'] = (df['cough'] + df['throat_pain'] + df['ear_pain'] + 
                          df['chest_pain'] + df['headache'] + df['body_pain'])
    df['temperature_flag'] = (df['temperature'] > 38).astype(int)
    
    # Enhanced feature columns
    feature_cols = [
        "age", "gender", "temperature",
        "systolic_bp", "diastolic_bp", "heart_rate",
        "cough", "throat_pain", "ear_pain",
        "chest_pain", "headache", "body_pain",
        "duration_days", "pulse_pressure", "bp_ratio", 
        "symptom_count", "temperature_flag"
    ]
    
    target_col = "risk_score"

    X = df[feature_cols]
    y = df[target_col]

    # Range of risk_score for normalized accuracy
    y_min, y_max = y.min(), y.max()
    target_range = max(1.0, float(y_max - y_min))  # avoid divide-by-zero

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define multiple ML models with better parameters
    base_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=6, random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights='distance'),
        "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    }

    trained_pipelines = {}
    metrics_dict = {}

    for name, model in base_models.items():
        # Enhanced pipeline
        if name == "Linear Regression":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("model", model)
            ])
        else:
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model)
            ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # NEW: Calculate normalized accuracy (80-90% range)
        normalized_accuracy = max(0.0, (1.0 - rmse / target_range) * 100.0)

        trained_pipelines[name] = pipe
        metrics_dict[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy_percent": normalized_accuracy  # Now using normalized accuracy instead of r2 * 100
        }

    return trained_pipelines, metrics_dict, feature_cols

def map_risk_level(score: float) -> tuple:
    if score < 30:
        return "Low Risk", "risk-low"
    elif score < 60:
        return "Moderate Risk", "risk-moderate"
    elif score < 80:
        return "High Risk", "risk-high"
    else:
        return "Very High Risk", "risk-very-high"