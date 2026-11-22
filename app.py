"""
app.py
------
Main Application Entry Point for Customer Churn Intelligence Dashboard.
Integrates all modules, layouts, and interactive callbacks into a high-performance
Plotly Dash production-grade application.

To run:
    python app.py
"""

import os
import sys
from dash import Dash, html, dcc, callback, Input, Output

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.dashboard.components.navbar import create_sidebar
from src.dashboard.tabs.executive_overview import create_overview_tab
from src.dashboard.tabs.customer_analytics import create_analytics_tab, register_analytics_callbacks
from src.dashboard.tabs.model_evaluation import create_model_evaluation_tab, register_model_callbacks
from src.dashboard.tabs.churn_simulator import create_simulator_tab, register_simulator_callbacks

# Initialize Dash application
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    assets_folder=os.path.join(BASE_DIR, "assets"),
    title="Customer Churn Intelligence Dashboard",
    update_title="Analyzing...",
)
server = app.server

# Core Application Layout
app.layout = html.Div(
    className="dashboard-wrapper",
    children=[
        # URL Location Tracker
        dcc.Location(id="url", refresh=False),
        
        # Left Sidebar Navigation
        create_sidebar(active_tab="tab-overview"),
        
        # Main Dynamic Content Pane
        html.Div(
            className="main-content",
            children=[
                html.Div(id="dashboard-tab-content", children=create_overview_tab()),
            ],
        ),
    ],
)

# Synchronize URL Path with Tab Selector
@app.callback(
    Output("nav-tab-selector", "value"),
    Input("url", "pathname"),
    prevent_initial_call=True
)
def sync_url_to_tab(pathname):
    """Maps URL path to sidebar tab value."""
    if pathname in ["/analytics", "/customer-analytics"]:
        return "tab-analytics"
    elif pathname in ["/models", "/benchmarking"]:
        return "tab-models"
    elif pathname in ["/simulator", "/predict"]:
        return "tab-simulator"
    return "tab-overview"


# Register Tab Navigation Callback
@app.callback(
    Output("dashboard-tab-content", "children"),
    Input("nav-tab-selector", "value")
)
def render_tab_content(tab_name):
    """Renders the appropriate dashboard view when tab navigation changes."""
    if tab_name == "tab-overview":
        return create_overview_tab()
    elif tab_name == "tab-analytics":
        return create_analytics_tab()
    elif tab_name == "tab-models":
        return create_model_evaluation_tab()
    elif tab_name == "tab-simulator":
        return create_simulator_tab()
    return create_overview_tab()


# Register sub-module interactive callbacks
register_analytics_callbacks(app)
register_model_callbacks(app)
register_simulator_callbacks(app)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   CUSTOMER CHURN INTELLIGENCE DASHBOARD - STARTING")
    print("=" * 70)
    print("   Access Dashboard URL: http://127.0.0.1:8050")
    print("   Press CTRL+C in terminal to stop server.")
    print("=" * 70 + "\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
