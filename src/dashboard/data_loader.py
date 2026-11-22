"""
src/dashboard/data_loader.py
----------------------------
High-performance cached data loading module for the Churn Intelligence Dashboard.
Loads authentic project datasets, pre-computed predictions, evaluation metrics,
and feature statistics without data leakage or modification of artifacts.
"""

import os
import json
import pandas as pd
import numpy as np
from src.dashboard.config import (
    RAW_DATA_PATH,
    PROCESSED_DIR,
    MODEL_COMPARISON_JSON,
    MODEL_COMPARISON_CSV,
    VALIDATION_THRESHOLD_CSV,
    LR_PREDICTIONS_CSV,
    RF_PREDICTIONS_CSV,
    NN_PREDICTIONS_CSV,
    RF_FEATURE_IMPORTANCE_CSV,
    LR_COEFFICIENTS_CSV,
)

# Global in-memory caches to avoid disk overhead on user interactions
_CACHE = {}


def get_raw_data() -> pd.DataFrame:
    """
    Loads and prepares the authentic IBM Telco Customer Churn dataset.
    Implements domain-justified handling of 11 blank TotalCharges records (tenure == 0 -> 0.0)
    and attaches analytical cohort groupings for interactive visualization.
    """
    if "raw_data" not in _CACHE:
        df = pd.read_csv(RAW_DATA_PATH)
        
        # Clean TotalCharges without altering raw disk CSV
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        
        # Analytical groupings for dashboard filters
        tenure_bins = [-1, 12, 24, 48, 72]
        tenure_labels = ["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
        df["TenureGroup"] = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
        
        df["SeniorCitizenLabel"] = df["SeniorCitizen"].map({1: "Senior Citizen", 0: "Non-Senior"})
        df["ChurnNumeric"] = (df["Churn"] == "Yes").astype(int)
        
        # Count of active internet add-on services
        addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        df["AddOnCount"] = (df[addon_cols] == "Yes").sum(axis=1)
        
        _CACHE["raw_data"] = df
        
    return _CACHE["raw_data"]


def get_executive_kpis() -> dict:
    """Calculates top-level executive KPIs directly from verified project data."""
    df = get_raw_data()
    total_customers = len(df)
    churn_count = int(df["ChurnNumeric"].sum())
    retained_count = total_customers - churn_count
    churn_rate = (churn_count / total_customers) * 100.0
    
    avg_monthly = float(df["MonthlyCharges"].mean())
    avg_tenure = float(df["tenure"].mean())
    total_mrr = float(df["MonthlyCharges"].sum())
    
    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": churn_rate,
        "avg_monthly": avg_monthly,
        "avg_tenure": avg_tenure,
        "total_mrr": total_mrr,
    }


def get_test_predictions() -> pd.DataFrame:
    """
    Loads held-out test set predictions for all three models (N = 1,409).
    Combines them into a single unified DataFrame for ROC, PR, and calibration comparisons.
    """
    if "test_predictions" not in _CACHE:
        df_lr = pd.read_csv(LR_PREDICTIONS_CSV)
        df_rf = pd.read_csv(RF_PREDICTIONS_CSV)
        df_nn = pd.read_csv(NN_PREDICTIONS_CSV)
        
        combined = pd.DataFrame({
            "Actual_Churn": df_lr["Actual_Churn"],
            "LR_Pred": df_lr["Predicted_Churn"],
            "LR_Prob": df_lr["Churn_Probability"],
            "RF_Pred": df_rf["Predicted_Churn"],
            "RF_Prob": df_rf["Churn_Probability"],
            "NN_Pred": df_nn["Predicted_Churn"],
            "NN_Prob": df_nn["Churn_Probability"],
        })
        _CACHE["test_predictions"] = combined
        
    return _CACHE["test_predictions"]


def get_model_comparison_metrics() -> dict:
    """Loads the verified Phase 7 model comparison JSON report."""
    if "metrics_json" not in _CACHE:
        with open(MODEL_COMPARISON_JSON, "r", encoding="utf-8") as f:
            _CACHE["metrics_json"] = json.load(f)
    return _CACHE["metrics_json"]


def get_validation_threshold_data() -> pd.DataFrame:
    """Loads validation set threshold analysis results."""
    if "val_thresholds" not in _CACHE:
        _CACHE["val_thresholds"] = pd.read_csv(VALIDATION_THRESHOLD_CSV)
    return _CACHE["val_thresholds"]


def get_feature_importances() -> tuple:
    """Loads Random Forest feature importances and Logistic Regression coefficients."""
    if "rf_importance" not in _CACHE:
        _CACHE["rf_importance"] = pd.read_csv(RF_FEATURE_IMPORTANCE_CSV)
    if "lr_coefficients" not in _CACHE:
        _CACHE["lr_coefficients"] = pd.read_csv(LR_COEFFICIENTS_CSV)
    return _CACHE["rf_importance"], _CACHE["lr_coefficients"]

