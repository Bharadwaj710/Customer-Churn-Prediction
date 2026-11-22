"""
src/dashboard/config.py
-----------------------
Central configuration for Customer Churn Intelligence Dashboard.
Defines paths, styling constants, color palettes, and Plotly layout templates.
"""

import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File paths
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

# Model checkpoints
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")
LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
NN_MODEL_PATH = os.path.join(MODELS_DIR, "neural_network.keras")

# Metric & Prediction files
MODEL_COMPARISON_JSON = os.path.join(METRICS_DIR, "model_comparison.json")
MODEL_COMPARISON_CSV = os.path.join(METRICS_DIR, "model_comparison.csv")
VALIDATION_THRESHOLD_CSV = os.path.join(METRICS_DIR, "validation_threshold_analysis.csv")
LR_PREDICTIONS_CSV = os.path.join(METRICS_DIR, "logistic_regression_predictions.csv")
RF_PREDICTIONS_CSV = os.path.join(METRICS_DIR, "random_forest_predictions.csv")
NN_PREDICTIONS_CSV = os.path.join(METRICS_DIR, "neural_network_predictions.csv")
RF_FEATURE_IMPORTANCE_CSV = os.path.join(METRICS_DIR, "random_forest_feature_importance.csv")
LR_COEFFICIENTS_CSV = os.path.join(METRICS_DIR, "logistic_regression_coefficients.csv")

# SaaS Design Theme & Color Palette
THEME = {
    "bg_app": "#0B0F19",          # Deep space dark background
    "bg_sidebar": "#0E1526",      # Sleek slate sidebar
    "bg_card": "#131C31",         # Elevated card surface
    "bg_card_hover": "#1A2540",   # Card hover state
    "border": "#212E4A",          # Subtle border line
    "border_light": "#2D3D60",    # Active tab/button border
    
    # Semantic Colors
    "churn": "#EF4444",           # Coral Crimson for Churned / High Risk
    "churn_glow": "rgba(239, 68, 68, 0.2)",
    "retained": "#10B981",        # Emerald for Retained / Low Risk
    "retained_glow": "rgba(16, 185, 129, 0.2)",
    "moderate": "#F59E0B",        # Amber for Medium Risk
    "moderate_glow": "rgba(245, 158, 11, 0.2)",
    
    # Model Accent Colors
    "primary": "#6366F1",         # Indigo for Random Forest & Primary actions
    "secondary": "#06B6D4",       # Cyan for Logistic Regression
    "accent": "#EC4899",          # Fuchsia for Neural Network
    "neutral": "#8B5CF6",         # Purple accent
    
    # Typography
    "text_main": "#F8FAFC",       # High contrast white
    "text_muted": "#94A3B8",      # Slate grey secondary
    "text_dim": "#64748B",        # Low emphasis tertiary
}

def get_plotly_layout_defaults():
    """Returns standardized styling dictionary for all Plotly figures."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                  color=THEME["text_muted"], size=12),
        title_font=dict(size=14, color=THEME["text_main"]),
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(
            bgcolor="rgba(19, 28, 49, 0.8)",
            bordercolor=THEME["border"],
            borderwidth=1,
            font=dict(color=THEME["text_main"], size=11)
        ),
        xaxis=dict(
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            tickfont=dict(color=THEME["text_muted"])
        ),
        yaxis=dict(
            gridcolor=THEME["border"],
            zerolinecolor=THEME["border"],
            tickfont=dict(color=THEME["text_muted"])
        )
    )

