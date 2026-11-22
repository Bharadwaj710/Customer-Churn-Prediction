"""
tests/verify_reproducibility.py
-------------------------------
Comprehensive automated reproducibility and quality assurance test.
Validates imports, model checkpoints, preprocessing compatibility,
single-record inference, batch inference, threshold shifting, and Dash routing.
"""

import os
import sys
import time
import unittest
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class ReproducibilityVerification(unittest.TestCase):
    """End-to-end automated verification suite."""

    def test_01_dependencies_import(self):
        """1 & 2. Verify all core libraries import without error."""
        import numpy
        import pandas
        import matplotlib
        import seaborn
        import sklearn
        import tensorflow
        import plotly
        import dash
        self.assertIsNotNone(numpy.__version__)
        self.assertIsNotNone(pandas.__version__)
        self.assertIsNotNone(sklearn.__version__)
        self.assertIsNotNone(tensorflow.__version__)
        self.assertIsNotNone(plotly.__version__)
        self.assertIsNotNone(dash.__version__)

    def test_02_model_and_preprocessor_loading(self):
        """5. Verify all four serialized checkpoints load correctly."""
        preprocessor = joblib.load(os.path.join(BASE_DIR, "outputs", "models", "preprocessor.joblib"))
        lr = joblib.load(os.path.join(BASE_DIR, "outputs", "models", "logistic_regression.joblib"))
        rf = joblib.load(os.path.join(BASE_DIR, "outputs", "models", "random_forest.joblib"))
        
        import tensorflow as tf
        nn = tf.keras.models.load_model(os.path.join(BASE_DIR, "outputs", "models", "neural_network.keras"))
        
        self.assertIsNotNone(preprocessor)
        self.assertIsNotNone(lr)
        self.assertIsNotNone(rf)
        self.assertIsNotNone(nn)

    def test_03_preprocessing_compatibility(self):
        """6. Verify ColumnTransformer transforms unseen records to exact 30 features."""
        from src.dashboard.model_inference import prepare_input_dataframe, get_inference_models
        models = get_inference_models()
        sample = {
            "tenure": 0, "MonthlyCharges": 45.0, "TotalCharges": "", "gender": "Female",
            "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "PhoneService": "Yes",
            "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
            "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
            "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)"
        }
        df_in = prepare_input_dataframe(sample)
        self.assertEqual(df_in["TotalCharges"].iloc[0], 0.0)
        X_proc = models["preprocessor"].transform(df_in)
        self.assertEqual(X_proc.shape, (1, 30))

    def test_04_single_record_inference(self):
        """7. Verify single-customer inference across all 3 models."""
        from src.dashboard.model_inference import predict_churn
        customer = {
            "tenure": 2, "MonthlyCharges": 90.0, "TotalCharges": 180.0,
            "Contract": "Month-to-month", "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check", "TechSupport": "No", "OnlineSecurity": "No"
        }
        for m in ["Random Forest", "Logistic Regression", "Neural Network"]:
            out = predict_churn(customer, model_name=m, threshold=0.50)
            prob = out["probability"]
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertTrue(out["is_churn"])
            self.assertEqual(out["risk_tier"], "High Risk")

    def test_05_batch_inference_consistency(self):
        """8. Verify batch inference matches saved test predictions."""
        X_test = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "X_test.csv"))
        saved_preds = pd.read_csv(os.path.join(BASE_DIR, "outputs", "metrics", "random_forest_predictions.csv"))
        
        from src.dashboard.model_inference import get_inference_models
        rf = get_inference_models()["rf"]
        batch_probs = rf.predict_proba(X_test)[:, 1]
        
        np.testing.assert_allclose(batch_probs, saved_preds["Churn_Probability"].values, rtol=1e-4, atol=1e-4)

    def test_06_dashboard_routing_and_views(self):
        """9. Verify all 4 dashboard views and callbacks load."""
        from app import render_tab_content
        for tab in ["tab-overview", "tab-analytics", "tab-models", "tab-simulator"]:
            view = render_tab_content(tab)
            self.assertIsNotNone(view)

    def test_07_retention_playbook(self):
        """Verify retention recommendations logic."""
        from src.dashboard.recommendations import generate_retention_playbook
        cust = {"tenure": 1, "Contract": "Month-to-month", "MonthlyCharges": 85.0, "InternetService": "Fiber optic", "TechSupport": "No", "OnlineSecurity": "No", "PaymentMethod": "Electronic check"}
        pb = generate_retention_playbook(cust, prob=0.82, risk_tier="High Risk")
        self.assertIn("HIGH RISK ALERT", pb["summary"])
        self.assertGreaterEqual(len(pb["actions"]), 4)


if __name__ == "__main__":
    unittest.main()

