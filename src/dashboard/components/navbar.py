"""
src/dashboard/components/navbar.py
----------------------------------
Sidebar navigation and application branding component.
"""

from dash import html, dcc
from src.dashboard.config import THEME


def create_sidebar(active_tab: str = "tab-overview"):
    """Builds the left-hand persistent navigation sidebar."""
    return html.Div(
        className="sidebar",
        children=[
            # Brand Header
            html.Div(
                className="brand-header",
                children=[
                    html.Div(
                        className="brand-title",
                        children=[
                            html.Span("⚡", style={"fontSize": "1.4rem"}),
                            html.Span("CHURN INTELLIGENCE"),
                        ],
                    ),
                    html.Div(
                        "B.Tech CSE Capstone · AI & Data Science",
                        className="brand-subtitle",
                    ),
                ],
            ),
            
            # Navigation Menu Tabs using RadioItems or native click links
            html.Div(
                className="nav-menu",
                children=[
                    dcc.RadioItems(
                        id="nav-tab-selector",
                        options=[
                            {
                                "label": html.Span([
                                    html.Span("📊 ", style={"marginRight": "8px"}),
                                    html.Span("Executive Overview")
                                ]),
                                "value": "tab-overview"
                            },
                            {
                                "label": html.Span([
                                    html.Span("🔍 ", style={"marginRight": "8px"}),
                                    html.Span("Customer Analytics")
                                ]),
                                "value": "tab-analytics"
                            },
                            {
                                "label": html.Span([
                                    html.Span("⚖️ ", style={"marginRight": "8px"}),
                                    html.Span("Model Benchmarking")
                                ]),
                                "value": "tab-models"
                            },
                            {
                                "label": html.Span([
                                    html.Span("🎯 ", style={"marginRight": "8px"}),
                                    html.Span("Churn Predictor & ROI")
                                ]),
                                "value": "tab-simulator"
                            },
                        ],
                        value=active_tab,
                        className="custom-radio-nav",
                        labelClassName="nav-item",
                        inputStyle={"display": "none"},
                    )
                ],
            ),
            
            # Sidebar Footer / Metadata
            html.Div(
                className="sidebar-footer",
                children=[
                    html.Div(
                        className="badge-status",
                        children=[
                            html.Span(className="status-dot"),
                            html.Span("7,043 Records Verified"),
                        ],
                        style={"marginBottom": "8px"}
                    ),
                    html.Div("IBM Telco Churn Dataset", style={"color": THEME["text_muted"], "fontWeight": "500"}),
                    html.Div("LR · RF · TF/Keras MLP", style={"color": THEME["text_dim"], "marginTop": "2px"}),
                ],
            ),
        ],
    )
