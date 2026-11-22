"""
src/dashboard/tabs/executive_overview.py
----------------------------------------
Module A: Executive Overview Landing Page.
Calculates high-level portfolio KPIs from verified raw project data,
presents churn distributions by contract and tenure, and summarizes strategic findings.
"""

from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.dashboard.config import THEME, get_plotly_layout_defaults
from src.dashboard.data_loader import get_raw_data, get_executive_kpis
from src.dashboard.components.cards import render_kpi_card, render_card_container


def create_overview_tab():
    """Generates Module A layout with KPIs and executive charts."""
    df = get_raw_data()
    kpis = get_executive_kpis()
    defaults = get_plotly_layout_defaults()

    # --- 1. Overall Churn Donut Chart ---
    churn_counts = df["Churn"].value_counts()
    fig_donut = go.Figure(data=[
        go.Pie(
            labels=["Retained", "Churned"],
            values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
            hole=0.62,
            marker=dict(colors=[THEME["retained"], THEME["churn"]]),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            textfont=dict(size=13, color=THEME["text_main"], family="Inter"),
        )
    ])
    fig_donut.update_layout(
        defaults,
        title=dict(text="Portfolio Churn Distribution", font=dict(color=THEME["text_main"], size=14)),
        annotations=[
            dict(
                text=f"<b>{100 - kpis['churn_rate']:.1f}%</b><br><span style='font-size:11px;color:{THEME['text_muted']}'>Retained</span>",
                x=0.5, y=0.5,
                font_size=18,
                font_color=THEME["text_main"],
                showarrow=False,
            )
        ],
        showlegend=True,
        height=320,
    )

    # --- 2. Churn by Contract Type ---
    contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
    contract_rates = (contract_churn["Yes"] / contract_churn.sum(axis=1) * 100).round(1)
    
    fig_contract = go.Figure(data=[
        go.Bar(
            name="Retained",
            x=contract_churn.index,
            y=contract_churn["No"],
            marker_color=THEME["retained"],
            hovertemplate="Retained: %{y:,}<extra></extra>",
        ),
        go.Bar(
            name="Churned",
            x=contract_churn.index,
            y=contract_churn["Yes"],
            marker_color=THEME["churn"],
            text=[f"{rate}% Churn" for rate in contract_rates],
            textposition="outside",
            textfont=dict(color=THEME["text_main"], size=11),
            hovertemplate="Churned: %{y:,} (%{text})<extra></extra>",
        ),
    ])
    fig_contract.update_layout(
        defaults,
        title=dict(text="Subscribers & Churn Rate by Contract Type", font=dict(color=THEME["text_main"], size=14)),
        barmode="group",
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # --- 3. Tenure Distribution by Churn Status ---
    fig_tenure = go.Figure()
    fig_tenure.add_trace(go.Histogram(
        x=df[df["Churn"] == "No"]["tenure"],
        name="Retained",
        marker_color=THEME["retained"],
        opacity=0.75,
        nbinsx=36,
        hovertemplate="Tenure: %{x} mos<br>Count: %{y}<extra></extra>",
    ))
    fig_tenure.add_trace(go.Histogram(
        x=df[df["Churn"] == "Yes"]["tenure"],
        name="Churned",
        marker_color=THEME["churn"],
        opacity=0.75,
        nbinsx=36,
        hovertemplate="Tenure: %{x} mos<br>Count: %{y}<extra></extra>",
    ))
    fig_tenure.update_layout(
        defaults,
        title=dict(text="Tenure Cohort Distribution (Months Active)", font=dict(color=THEME["text_main"], size=14)),
        barmode="overlay",
        xaxis=dict(title="Tenure (Months)", gridcolor=THEME["border"]),
        yaxis=dict(title="Subscriber Count", gridcolor=THEME["border"]),
        height=320,
    )

    # --- 4. Monthly Charges Distribution by Churn ---
    fig_charges = go.Figure()
    fig_charges.add_trace(go.Box(
        y=df[df["Churn"] == "No"]["MonthlyCharges"],
        name="Retained",
        marker_color=THEME["retained"],
        boxmean=True,
    ))
    fig_charges.add_trace(go.Box(
        y=df[df["Churn"] == "Yes"]["MonthlyCharges"],
        name="Churned",
        marker_color=THEME["churn"],
        boxmean=True,
    ))
    fig_charges.update_layout(
        defaults,
        title=dict(text="Monthly Charges Box Plot & Mean ($/mo)", font=dict(color=THEME["text_main"], size=14)),
        yaxis=dict(title="Monthly Charges (USD)", gridcolor=THEME["border"]),
        height=320,
    )

    # --- Layout Assembly ---
    return html.Div([
        # Page Title
        html.Div(
            className="page-header",
            children=[
                html.Div([
                    html.H1("Executive Portfolio Overview", className="page-title"),
                    html.P("Real-time portfolio metrics, subscriber tenure cohorts, and churn revenue exposure.", className="page-desc"),
                ]),
                html.Div(
                    className="badge-status",
                    children=[
                        html.Span(className="status-dot"),
                        html.Span(f"Total Cohort: {kpis['total_customers']:,} Subscribers"),
                    ],
                ),
            ],
        ),

        # KPI Metrics Row
        html.Div(
            className="kpi-grid",
            children=[
                render_kpi_card("Total Subscribers", f"{kpis['total_customers']:,}", "All verified active & churned accounts", "primary"),
                render_kpi_card("Retained Accounts", f"{kpis['retained_count']:,}", f"{100 - kpis['churn_rate']:.2f}% customer retention rate", "retained"),
                render_kpi_card("Churned Accounts", f"{kpis['churn_count']:,}", f"{kpis['churn_rate']:.2f}% empirical churn rate", "churn"),
                render_kpi_card("Monthly Revenue / Avg", f"${kpis['avg_monthly']:.2f}", f"Avg tenure: {kpis['avg_tenure']:.1f} months", "cyan"),
            ],
        ),

        # Row 1 Visualizations: Donut & Contract
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Churn Distribution", "Aggregate class balance across 7,043 subscribers", dcc.Graph(figure=fig_donut, config={"displayModeBar": False})),
                render_card_container("Contract Risk Profile", "Subscribers and empirical churn percentage by contract structure", dcc.Graph(figure=fig_contract, config={"displayModeBar": False})),
            ],
        ),

        # Row 2 Visualizations: Tenure & Monthly Charges
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Tenure Hazard Analysis", "Distribution of months active highlighting early-tenure vulnerability", dcc.Graph(figure=fig_tenure, config={"displayModeBar": False})),
                render_card_container("Monthly Charges Spread", "Median and mean expenditure comparison between retained and churned cohorts", dcc.Graph(figure=fig_charges, config={"displayModeBar": False})),
            ],
        ),

        # Row 3: Strategic Business Insights
        render_card_container(
            "Executive Key Findings & Business Observations",
            "Synthesized empirical drivers derived from authentic exploratory data analysis",
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "16px"},
                children=[
                    html.Div(
                        className="rec-card",
                        style={"borderLeftColor": THEME["churn"]},
                        children=[
                            html.Div("1. Month-to-Month Contract Vulnerability", style={"fontWeight": "600", "color": THEME["text_main"], "marginBottom": "6px"}),
                            html.P("Subscribers on month-to-month contracts experience a 42.7% churn rate, compared to just 11.3% for 1-year and 2.8% for 2-year contracts. Flexible terms present high operational churn hazard.", className="rec-text"),
                        ]
                    ),
                    html.Div(
                        className="rec-card",
                        style={"borderLeftColor": THEME["moderate"]},
                        children=[
                            html.Div("2. The First-Year Churn Cliff", style={"fontWeight": "600", "color": THEME["text_main"], "marginBottom": "6px"}),
                            html.P("Nearly 50% of all recorded cancellations occur within the first 12 months of service. Onboarding friction and initial billing surprises are the most critical inflection points.", className="rec-text"),
                        ]
                    ),
                    html.Div(
                        className="rec-card",
                        style={"borderLeftColor": THEME["primary"]},
                        children=[
                            html.Div("3. Fiber Optic Support Friction", style={"fontWeight": "600", "color": THEME["text_main"], "marginBottom": "6px"}),
                            html.P("Fiber optic customers display the highest churn (41.9%) despite commanding premium $70-$100/mo fees. When paired without Online Security or Tech Support, churn probability peaks.", className="rec-text"),
                        ]
                    ),
                ]
            )
        ),
    ])

