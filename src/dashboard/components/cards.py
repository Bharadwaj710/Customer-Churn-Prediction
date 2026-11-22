"""
src/dashboard/components/cards.py
---------------------------------
Reusable card containers, KPI metric blocks, and Plotly gauge indicators.
"""

from dash import html, dcc
import plotly.graph_objects as go
from src.dashboard.config import THEME


def render_kpi_card(title: str, value: str, subtext: str, accent: str = "primary"):
    """Creates a styled KPI stat block."""
    accent_class = f"kpi-accent-{accent}"
    return html.Div(
        className=f"kpi-card {accent_class}",
        children=[
            html.Div(title, className="kpi-title"),
            html.Div(value, className="kpi-value"),
            html.Div(subtext, className="kpi-subtext"),
        ],
    )


def render_card_container(title: str, subtitle: str, content, extra_header=None):
    """Wraps arbitrary content in a standard dashboard card."""
    header_children = [
        html.Div([
            html.Div(title, className="card-title"),
            html.Div(subtitle, className="card-subtitle") if subtitle else None,
        ])
    ]
    if extra_header:
        header_children.append(extra_header)
        
    return html.Div(
        className="dashboard-card",
        children=[
            html.Div(className="card-header-bar", children=header_children),
            html.Div(content),
        ],
    )


def render_gauge_chart(probability: float, threshold: float = 0.50) -> go.Figure:
    """
    Creates an interactive gauge indicator for customer churn risk.
    Green (0-35%): Low Risk
    Amber (35-60%): Moderate Risk
    Red (60-100%): Critical Churn Risk
    """
    pct = round(probability * 100, 1)
    
    if pct < 35:
        bar_color = THEME["retained"]
    elif pct < 60:
        bar_color = THEME["moderate"]
    else:
        bar_color = THEME["churn"]
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": THEME["text_main"], "family": "Inter"}},
        title={"text": "Estimated Churn Probability", "font": {"size": 14, "color": THEME["text_muted"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": THEME["border"], "tickfont": {"color": THEME["text_dim"]}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(11, 16, 29, 0.8)",
            "borderwidth": 1,
            "bordercolor": THEME["border"],
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                {"range": [35, 60], "color": "rgba(245, 158, 11, 0.15)"},
                {"range": [60, 100], "color": "rgba(239, 68, 68, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=THEME["text_muted"]),
        height=240,
        margin=dict(l=30, r=30, t=30, b=10),
    )
    return fig

