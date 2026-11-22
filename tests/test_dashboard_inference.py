"""
tests/test_dashboard_inference.py
---------------------------------
Comprehensive unit and integration test suite for the Customer Churn Intelligence Dashboard.
Verifies preprocessor reproduction, multi-model inference, threshold logic,
retention playbooks, metric fidelity, and dashboard layout initialization.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dashboard.model_inference import (
    prepare_input_dataframe,
    predict_churn,
    get_inference_models,
)
from src.dashboard.recommendations import generate_retention_playbook
from src.dashboard.data_loader import (
    get_raw_data,
    get_executive_kpis,
    get_test_predictions,
    get_model_comparison_metrics,
    get_validation_threshold_data,
)
from src.dashboard.tabs.executive_overview import create_overview_tab
from src.dashboard.tabs.customer_analytics import create_analytics_tab
from src.dashboard.tabs.model_evaluation import create_model_evaluation_tab
from src.dashboard.tabs.churn_simulator import create_simulator_tab


class TestDashboardInference(unittest.TestCase):
    """Test suite for preprocessing, model inference, and threshold sensitivity."""

    @classmethod
    def setUpClass(cls):
        """Loads inference models once for the test suite."""
        cls.models = get_inference_models()
        cls.expected_features_path = os.path.join(BASE_DIR, "data", "processed", "X_train.csv")
        cls.expected_features = list(pd.read_csv(cls.expected_features_path, nrows=1).columns)

    def test_preprocessor_transformation_shape_and_names(self):
        """Tests that preprocessing produces exactly 30 features matching X_train columns."""
        sample_dict = {
            "tenure": 12,
            "MonthlyCharges": 70.0,
            "TotalCharges": 840.0,
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }
        df_in = prepare_input_dataframe(sample_dict)
        self.assertEqual(df_in.shape, (1, 19))
        
        preprocessor = self.models["preprocessor"]
        transformed = preprocessor.transform(df_in)
        self.assertEqual(transformed.shape, (1, 30))
        
        feature_names = list(preprocessor.get_feature_names_out())
        self.assertEqual(feature_names, self.expected_features)

    def test_zero_tenure_domain_imputation(self):
        """Tests that tenure == 0 correctly defaults TotalCharges to 0.0 without errors."""
        sample_dict = {
            "tenure": 0,
            "MonthlyCharges": 55.0,
            "TotalCharges": "",  # Blank / missing string
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
        }
        df_in = prepare_input_dataframe(sample_dict)
        self.assertEqual(df_in["TotalCharges"].iloc[0], 0.0)
        self.assertEqual(df_in["tenure"].iloc[0], 0.0)
        
        # Verify preprocessor transforms cleanly
        transformed = self.models["preprocessor"].transform(df_in)
        self.assertEqual(transformed.shape, (1, 30))

    def test_multi_model_inference_validity(self):
        """Tests that all 3 models generate valid probabilities in [0, 1]."""
        high_risk_customer = {
            "tenure": 1,
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0,
            "gender": "Female",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }

        for model_name in ["Logistic Regression", "Random Forest", "Neural Network"]:
            res = predict_churn(high_risk_customer, model_name=model_name, threshold=0.50)
            prob = res["probability"]
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)
            self.assertIn("risk_tier", res)
            self.assertIn("explanations", res)

    def test_threshold_decision_logic(self):
        """Tests that varying threshold modifies the binary classification decision monotonically."""
        customer = {
            "tenure": 6,
            "MonthlyCharges": 75.0,
            "TotalCharges": 450.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        }
        # Run with low threshold (0.20) -> should be churn (True)
        res_low = predict_churn(customer, model_name="Random Forest", threshold=0.20)
        # Run with high threshold (0.95) -> should be retained (False)
        res_high = predict_churn(customer, model_name="Random Forest", threshold=0.95)
        
        self.assertTrue(res_low["is_churn"])
        self.assertFalse(res_high["is_churn"])

    def test_retention_playbook_generation(self):
        """Tests that business recommendations correctly target key risk drivers."""
        sample_risk = {
            "tenure": 3,
            "Contract": "Month-to-month",
            "MonthlyCharges": 85.0,
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaymentMethod": "Electronic check",
        }
        playbook = generate_retention_playbook(sample_risk, prob=0.75, risk_tier="High Risk")
        
        self.assertIn("HIGH RISK ALERT", playbook["summary"])
        self.assertGreater(len(playbook["actions"]), 0)
        
        categories = [a["category"] for a in playbook["actions"]]
        self.assertIn("Contract Structure", categories)
        self.assertIn("Customer Success", categories)
        self.assertIn("Service Value & Support", categories)
        self.assertIn("Billing Operations", categories)

    def test_verified_metrics_fidelity(self):
        """Confirms loaded comparison metrics match Phase 7 verified values."""
        comp = get_model_comparison_metrics()["model_metrics"]
        
        # Logistic Regression
        self.assertAlmostEqual(comp["Logistic Regression"]["Accuracy"], 0.7381, places=4)
        self.assertAlmostEqual(comp["Logistic Regression"]["ROC-AUC"], 0.8417, places=4)
        self.assertAlmostEqual(comp["Logistic Regression"]["Average Precision (AP)"], 0.6331, places=4)

        # Random Forest
        self.assertAlmostEqual(comp["Random Forest"]["Accuracy"], 0.7573, places=4)
        self.assertAlmostEqual(comp["Random Forest"]["ROC-AUC"], 0.8434, places=4)
        self.assertAlmostEqual(comp["Random Forest"]["Average Precision (AP)"], 0.6590, places=4)

        # Neural Network
        self.assertAlmostEqual(comp["Neural Network (MLP)"]["Accuracy"], 0.7459, places=4)
        self.assertAlmostEqual(comp["Neural Network (MLP)"]["ROC-AUC"], 0.8409, places=4)
        self.assertAlmostEqual(comp["Neural Network (MLP)"]["Average Precision (AP)"], 0.6322, places=4)

    def test_dashboard_tab_rendering(self):
        """Verifies that all 4 dashboard views initialize without runtime exceptions."""
        overview_layout = create_overview_tab()
        self.assertIsNotNone(overview_layout)

        analytics_layout = create_analytics_tab()
        self.assertIsNotNone(analytics_layout)

        models_layout = create_model_evaluation_tab()
        self.assertIsNotNone(models_layout)

        simulator_layout = create_simulator_tab()
        self.assertIsNotNone(simulator_layout)


if __name__ == "__main__":
    unittest.main()

