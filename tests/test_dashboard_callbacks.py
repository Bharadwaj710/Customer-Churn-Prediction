"""
tests/test_dashboard_callbacks.py
---------------------------------
Smoke test verifying that all Dash callback functions execute cleanly without
raising exceptions and generate valid figures and HTML elements.
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app, render_tab_content
from src.dashboard.tabs.customer_analytics import update_analytics_charts
from src.dashboard.tabs.model_evaluation import update_confusion_matrix, update_threshold_chart
from src.dashboard.tabs.churn_simulator import run_customer_prediction


class TestDashboardCallbacks(unittest.TestCase):
    """Smoke test for all Dash callbacks and layout rendering."""

    def test_app_initialization(self):
        """Verifies that app object is initialized with layout and server."""
        self.assertIsNotNone(app.layout)
        self.assertIsNotNone(app.server)

    def test_tab_navigation_callback(self):
        """Tests that tab switching returns non-empty layouts for all options."""
        for tab in ["tab-overview", "tab-analytics", "tab-models", "tab-simulator"]:
            content = render_tab_content(tab)
            self.assertIsNotNone(content)

    def test_analytics_callbacks(self):
        """Tests dynamic callback logic for customer analytics filters."""
        banner, f_demo, f_pay, f_ten, f_sec, f_scat, f_addon = update_analytics_charts(
            "Month-to-month", "Fiber optic", "Electronic check", "ALL"
        )
        self.assertIsNotNone(banner)
        self.assertIsNotNone(f_demo)
        self.assertIsNotNone(f_pay)
        self.assertIsNotNone(f_ten)
        self.assertIsNotNone(f_sec)
        self.assertIsNotNone(f_scat)
        self.assertIsNotNone(f_addon)

    def test_confusion_matrix_callback(self):
        """Tests confusion matrix updater for all models and normalization modes."""
        for model in ["LR", "RF", "NN"]:
            for mode in ["counts", "overall_pct", "row_pct"]:
                fig = update_confusion_matrix(model, mode)
                self.assertIsNotNone(fig)

    def test_threshold_sensitivity_callback(self):
        """Tests threshold sensitivity curve updater."""
        for model in ["Logistic Regression", "Random Forest", "Neural Network (MLP)"]:
            fig = update_threshold_chart(model)
            self.assertIsNotNone(fig)

    def test_simulator_prediction_callback(self):
        """Tests end-to-end simulator callback with realistic inputs."""
        res, recs = run_customer_prediction(
            1, 0.50, "Random Forest",
            "Month-to-month", 3, 85.0, None, "Electronic check", "Yes",
            "Fiber optic", "No", "No", "No", "No",
            "Yes", "No", "Yes", "Yes",
            "Female", 0, "No", "No"
        )
        self.assertIsNotNone(res)
        self.assertIsNotNone(recs)


if __name__ == "__main__":
    unittest.main()
