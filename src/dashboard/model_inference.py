"""
src/dashboard/model_inference.py
--------------------------------
Real-time customer inference engine.
Reuses the exact Scikit-learn preprocessor pipeline (fitted strictly on training data)
and loaded model artifacts to compute churn probability, binary risk categorization,
and model-specific feature explanations.
"""

import os
import joblib
import pandas as pd
import numpy as np
from src.dashboard.config import (
    PREPROCESSOR_PATH,
    LR_MODEL_PATH,
    RF_MODEL_PATH,
    NN_MODEL_PATH,
)

# Silence TensorFlow verbose logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Singleton model container to prevent redundant re-loading
_LOADED_MODELS = {}


def get_inference_models():
    """Loads all models and the fitted preprocessor once into memory."""
    if "preprocessor" not in _LOADED_MODELS:
        _LOADED_MODELS["preprocessor"] = joblib.load(PREPROCESSOR_PATH)
        _LOADED_MODELS["lr"] = joblib.load(LR_MODEL_PATH)
        _LOADED_MODELS["rf"] = joblib.load(RF_MODEL_PATH)
        
        # Load Keras MLP
        import tensorflow as tf
        _LOADED_MODELS["nn"] = tf.keras.models.load_model(NN_MODEL_PATH)
        
    return _LOADED_MODELS


def prepare_input_dataframe(raw_dict: dict) -> pd.DataFrame:
    """
    Constructs and validates a single-row DataFrame matching the exact schema
    expected by preprocessor.joblib (19 input columns).
    """
    tenure = float(raw_dict.get("tenure", 1))
    monthly_charges = float(raw_dict.get("MonthlyCharges", 50.0))
    
    # Domain-justified TotalCharges handling
    if tenure == 0:
        total_charges = 0.0
    else:
        tc_val = raw_dict.get("TotalCharges")
        if tc_val is not None and str(tc_val).strip() != "":
            try:
                total_charges = float(tc_val)
            except ValueError:
                total_charges = float(round(monthly_charges * tenure, 2))
        else:
            total_charges = float(round(monthly_charges * tenure, 2))

    data = {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender": [str(raw_dict.get("gender", "Female"))],
        "SeniorCitizen": [int(raw_dict.get("SeniorCitizen", 0))],
        "Partner": [str(raw_dict.get("Partner", "No"))],
        "Dependents": [str(raw_dict.get("Dependents", "No"))],
        "PhoneService": [str(raw_dict.get("PhoneService", "Yes"))],
        "MultipleLines": [str(raw_dict.get("MultipleLines", "No"))],
        "InternetService": [str(raw_dict.get("InternetService", "DSL"))],
        "OnlineSecurity": [str(raw_dict.get("OnlineSecurity", "No"))],
        "OnlineBackup": [str(raw_dict.get("OnlineBackup", "No"))],
        "DeviceProtection": [str(raw_dict.get("DeviceProtection", "No"))],
        "TechSupport": [str(raw_dict.get("TechSupport", "No"))],
        "StreamingTV": [str(raw_dict.get("StreamingTV", "No"))],
        "StreamingMovies": [str(raw_dict.get("StreamingMovies", "No"))],
        "Contract": [str(raw_dict.get("Contract", "Month-to-month"))],
        "PaperlessBilling": [str(raw_dict.get("PaperlessBilling", "Yes"))],
        "PaymentMethod": [str(raw_dict.get("PaymentMethod", "Electronic check"))],
    }
    return pd.DataFrame(data)


def predict_churn(customer_data: dict, model_name: str = "Random Forest", threshold: float = 0.50) -> dict:
    """
    Executes end-to-end inference for a customer profile.
    
    Returns:
        dict containing:
            - probability: float (0.0 to 1.0)
            - is_churn: bool (True if prob >= threshold)
            - risk_tier: str ("Low Risk", "Moderate Risk", "High Risk")
            - model_name: str
            - threshold: float
            - explanations: list of dicts with feature impact insights
            - customer_summary: dict of cleaned inputs
    """
    models = get_inference_models()
    preprocessor = models["preprocessor"]
    
    # 1. Prepare raw input DataFrame
    input_df = prepare_input_dataframe(customer_data)
    
    # 2. Transform through fitted preprocessor
    transformed_features = preprocessor.transform(input_df)
    feature_names = preprocessor.get_feature_names_out()
    X_proc_df = pd.DataFrame(transformed_features, columns=feature_names)
    
    # 3. Model Inference
    if model_name == "Logistic Regression":
        prob = float(models["lr"].predict_proba(X_proc_df)[0, 1])
        # Calculate feature contributions: w_i * x_i
        coefs = models["lr"].coef_[0]
        contributions = coefs * transformed_features[0]
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions,
            "Absolute": np.abs(contributions)
        }).sort_values(by="Absolute", ascending=False)
        
        top_risk = contrib_df[contrib_df["Contribution"] > 0].head(3).to_dict(orient="records")
        top_protective = contrib_df[contrib_df["Contribution"] < 0].head(3).to_dict(orient="records")
        explanations = {
            "type": "logistic_regression",
            "top_risk_factors": top_risk,
            "top_protective_factors": top_protective,
        }
        
    elif model_name == "Random Forest":
        prob = float(models["rf"].predict_proba(X_proc_df)[0, 1])
        importances = models["rf"].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
            "Customer_Value": transformed_features[0]
        }).sort_values(by="Importance", ascending=False)
        
        top_imp = imp_df.head(5).to_dict(orient="records")
        explanations = {
            "type": "random_forest",
            "top_features": top_imp,
        }
        
    elif model_name in ["Neural Network", "Neural Network (MLP)"]:
        raw_pred = models["nn"].predict(X_proc_df, verbose=0)
        prob = float(raw_pred[0][0])
        explanations = {
            "type": "neural_network",
            "tenure_months": float(input_df["tenure"].iloc[0]),
            "monthly_charges": float(input_df["MonthlyCharges"].iloc[0]),
            "contract": str(input_df["Contract"].iloc[0]),
            "internet_service": str(input_df["InternetService"].iloc[0]),
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # 4. Binary Decision based on threshold
    is_churn = bool(prob >= threshold)
    
    # 5. Risk Category Assignment
    if prob < 0.35:
        risk_tier = "Low Risk"
        risk_color = "#10B981"
        badge_class = "risk-badge-low"
    elif prob < 0.60:
        risk_tier = "Moderate Risk"
        risk_color = "#F59E0B"
        badge_class = "risk-badge-medium"
    else:
        risk_tier = "High Risk"
        risk_color = "#EF4444"
        badge_class = "risk-badge-high"
        
    return {
        "probability": prob,
        "is_churn": is_churn,
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "badge_class": badge_class,
        "model_name": model_name,
        "threshold": threshold,
        "explanations": explanations,
        "customer_summary": input_df.iloc[0].to_dict(),
    }

