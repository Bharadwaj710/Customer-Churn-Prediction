"""
src/dashboard/tabs/customer_analytics.py
----------------------------------------
Module B: Customer Analytics & Segment Deep Dive.
Provides interactive multidimensional exploratory visualizations across demographics,
billing methods, tenure cohorts, and service add-on adoption.
Filters interactively update all charts via Dash callbacks.
"""

from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.dashboard.config import THEME, get_plotly_layout_defaults
from src.dashboard.data_loader import get_raw_data
from src.dashboard.components.cards import render_card_container


def create_analytics_tab():
    """Generates Module B layout with interactive filter controls and analytical charts."""
    df = get_raw_data()
    
    contract_options = [{"label": "All Contracts", "value": "ALL"}] + [
        {"label": c, "value": c} for c in sorted(df["Contract"].unique())
    ]
    internet_options = [{"label": "All Internet Services", "value": "ALL"}] + [
        {"label": i, "value": i} for i in sorted(df["InternetService"].unique())
    ]
    payment_options = [{"label": "All Payment Methods", "value": "ALL"}] + [
        {"label": p, "value": p} for p in sorted(df["PaymentMethod"].unique())
    ]
    senior_options = [
        {"label": "All Age Groups", "value": "ALL"},
        {"label": "Senior Citizens (65+)", "value": 1},
        {"label": "Non-Seniors (<65)", "value": 0},
    ]

    return html.Div([
        # Page Title
        html.Div(
            className="page-header",
            children=[
                html.Div([
                    html.H1("Customer Cohort Analytics", className="page-title"),
                    html.P("Segment-level behavioral drivers, billing preferences, and technical service adoption patterns.", className="page-desc"),
                ]),
            ],
        ),

        # Interactive Filter Bar
        html.Div(
            className="filter-bar",
            children=[
                html.Div(
                    className="filter-group",
                    children=[
                        html.Label("Contract Agreement", className="filter-label"),
                        dcc.Dropdown(
                            id="filter-contract",
                            options=contract_options,
                            value="ALL",
                            clearable=False,
                            className="custom-dropdown",
                        ),
                    ],
                ),
                html.Div(
                    className="filter-group",
                    children=[
                        html.Label("Internet Infrastructure", className="filter-label"),
                        dcc.Dropdown(
                            id="filter-internet",
                            options=internet_options,
                            value="ALL",
                            clearable=False,
                            className="custom-dropdown",
                        ),
                    ],
                ),
                html.Div(
                    className="filter-group",
                    children=[
                        html.Label("Billing Channel", className="filter-label"),
                        dcc.Dropdown(
                            id="filter-payment",
                            options=payment_options,
                            value="ALL",
                            clearable=False,
                            className="custom-dropdown",
                        ),
                    ],
                ),
                html.Div(
                    className="filter-group",
                    children=[
                        html.Label("Senior Citizen Filter", className="filter-label"),
                        dcc.Dropdown(
                            id="filter-senior",
                            options=senior_options,
                            value="ALL",
                            clearable=False,
                            className="custom-dropdown",
                        ),
                    ],
                ),
            ],
        ),

        # Dynamic Segment Summary Banner
        html.Div(id="analytics-cohort-banner", style={"marginBottom": "20px"}),

        # Row 1: Demographics & Billing
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Demographics & Churn Rate", "Churn rate broken down by Senior Citizen status and Gender", dcc.Graph(id="chart-demographics", config={"displayModeBar": False})),
                render_card_container("Payment Method & Billing Friction", "Churn percentage by Payment Method and Paperless Billing enrollment", dcc.Graph(id="chart-payment", config={"displayModeBar": False})),
            ],
        ),

        # Row 2: Tenure Cohorts & Service Security
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Tenure Cohort Churn Hazard", "Churn rate across progressive subscriber lifecycle stages", dcc.Graph(id="chart-tenure-cohorts", config={"displayModeBar": False})),
                render_card_container("Cyber Security & Technical Support", "Impact of foundational support add-ons on churn risk", dcc.Graph(id="chart-security-support", config={"displayModeBar": False})),
            ],
        ),

        # Row 3: Charges Relationship & Service Bundle Adoption
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Monthly vs Total Charges", "Expenditure correlation and subscriber longevity colored by churn status", dcc.Graph(id="chart-charges-scatter", config={"displayModeBar": False})),
                render_card_container("Service Adoption Multi-Product Breadth", "Churn rate as a function of the number of active add-on services", dcc.Graph(id="chart-addon-depth", config={"displayModeBar": False})),
            ],
        ),

        # Associative Disclaimer
        html.Div(
            style={
                "padding": "14px 18px",
                "backgroundColor": "rgba(33, 46, 74, 0.4)",
                "border": f"1px solid {THEME['border']}",
                "borderRadius": "8px",
                "fontSize": "0.78rem",
                "color": THEME["text_dim"],
                "marginBottom": "24px",
            },
            children=[
                html.Strong("Analytical & Epistemological Note: "),
                "All visualizations in this section reflect empirical bivariate and multivariate observations from the verified IBM Telco dataset. "
                "These observational associations describe historical subscriber patterns and do not imply direct individual-level causation."
            ]
        )
    ])


def update_analytics_charts(contract_val, internet_val, payment_val, senior_val):
    """Computes filtered cohort metrics and returns all Module B figures."""
    df = get_raw_data().copy()
    defaults = get_plotly_layout_defaults()

    # Apply Filters
    if contract_val != "ALL":
        df = df[df["Contract"] == contract_val]
    if internet_val != "ALL":
        df = df[df["InternetService"] == internet_val]
    if payment_val != "ALL":
        df = df[df["PaymentMethod"] == payment_val]
    if senior_val != "ALL":
        df = df[df["SeniorCitizen"] == senior_val]

    # Cohort summary stats
    cohort_n = len(df)
    if cohort_n > 0:
        cohort_churn_count = int(df["ChurnNumeric"].sum())
        cohort_churn_pct = (cohort_churn_count / cohort_n) * 100.0
        cohort_avg_monthly = float(df["MonthlyCharges"].mean())
    else:
        cohort_churn_count = 0
        cohort_churn_pct = 0.0
        cohort_avg_monthly = 0.0

    banner = html.Div(
        style={
            "display": "flex",
            "gap": "16px",
            "padding": "12px 18px",
            "backgroundColor": "rgba(99, 102, 241, 0.08)",
            "border": f"1px solid {THEME['border']}",
            "borderRadius": "8px",
            "alignItems": "center",
            "flexWrap": "wrap",
        },
        children=[
            html.Span(f"Active Filtered Segment: {cohort_n:,} Subscribers", style={"fontWeight": "700", "color": THEME["text_main"]}),
            html.Span("·", style={"color": THEME["text_dim"]}),
            html.Span(f"Churn Count: {cohort_churn_count:,}", style={"color": THEME["churn"]}),
            html.Span("·", style={"color": THEME["text_dim"]}),
            html.Span(f"Segment Churn Rate: {cohort_churn_pct:.1f}%", style={"fontWeight": "700", "color": THEME["churn"] if cohort_churn_pct > 26.5 else THEME["retained"]}),
            html.Span("·", style={"color": THEME["text_dim"]}),
            html.Span(f"Avg Monthly Charges: ${cohort_avg_monthly:.2f}/mo", style={"color": THEME["text_muted"]}),
        ]
    )

    if cohort_n == 0:
        empty_fig = go.Figure().update_layout(defaults, title="No subscribers match selected filter criteria.")
        return banner, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig

    # --- 1. Demographics Chart ---
    demo_df = df.groupby(["SeniorCitizenLabel", "gender"])["ChurnNumeric"].agg(["count", "mean"]).reset_index()
    demo_df["ChurnRate"] = (demo_df["mean"] * 100).round(1)
    
    fig_demo = px.bar(
        demo_df,
        x="SeniorCitizenLabel",
        y="ChurnRate",
        color="gender",
        barmode="group",
        text="ChurnRate",
        color_discrete_map={"Female": THEME["accent"], "Male": THEME["secondary"]},
    )
    fig_demo.update_traces(texttemplate="%{text}%", textposition="outside", textfont=dict(color=THEME["text_main"]))
    fig_demo.update_layout(
        defaults,
        title=dict(text="Churn Rate by Age Group & Gender (%)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Demographic Group", gridcolor=THEME["border"]),
        yaxis=dict(title="Churn Rate (%)", range=[0, max(demo_df["ChurnRate"].max() * 1.25, 40)], gridcolor=THEME["border"]),
        height=300,
    )

    # --- 2. Payment Method Chart ---
    pay_df = df.groupby(["PaymentMethod", "PaperlessBilling"])["ChurnNumeric"].agg(["count", "mean"]).reset_index()
    pay_df["ChurnRate"] = (pay_df["mean"] * 100).round(1)
    
    fig_pay = px.bar(
        pay_df,
        x="PaymentMethod",
        y="ChurnRate",
        color="PaperlessBilling",
        barmode="group",
        text="ChurnRate",
        color_discrete_map={"Yes": THEME["churn"], "No": THEME["retained"]},
        labels={"PaperlessBilling": "Paperless Billing"},
    )
    fig_pay.update_traces(texttemplate="%{text}%", textposition="outside", textfont=dict(color=THEME["text_main"]))
    fig_pay.update_layout(
        defaults,
        title=dict(text="Churn Rate by Payment Channel & Paperless Billing (%)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="", tickangle=-12, gridcolor=THEME["border"]),
        yaxis=dict(title="Churn Rate (%)", range=[0, max(pay_df["ChurnRate"].max() * 1.25, 40)], gridcolor=THEME["border"]),
        height=300,
    )

    # --- 3. Tenure Cohorts Chart ---
    tenure_df = df.groupby("TenureGroup", observed=False)["ChurnNumeric"].agg(["count", "mean"]).reset_index()
    tenure_df["ChurnRate"] = (tenure_df["mean"] * 100).round(1)
    
    fig_tenure = px.bar(
        tenure_df,
        x="TenureGroup",
        y="ChurnRate",
        text="ChurnRate",
        color="ChurnRate",
        color_continuous_scale=[[0, THEME["retained"]], [1, THEME["churn"]]],
    )
    fig_tenure.update_traces(texttemplate="%{text}%", textposition="outside", textfont=dict(color=THEME["text_main"]))
    fig_tenure.update_layout(
        defaults,
        coloraxis_showscale=False,
        title=dict(text="Churn Rate Across Subscriber Tenure Stages (%)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Tenure Lifecycle Cohort", gridcolor=THEME["border"]),
        yaxis=dict(title="Churn Rate (%)", range=[0, max(tenure_df["ChurnRate"].max() * 1.25, 40)], gridcolor=THEME["border"]),
        height=300,
    )

    # --- 4. Security & Tech Support Chart ---
    sec_df = df.groupby(["OnlineSecurity", "TechSupport"])["ChurnNumeric"].agg(["count", "mean"]).reset_index()
    sec_df["ChurnRate"] = (sec_df["mean"] * 100).round(1)
    sec_df["CombinedLabel"] = "Sec: " + sec_df["OnlineSecurity"] + " | Tech: " + sec_df["TechSupport"]
    sec_df = sec_df.sort_values(by="ChurnRate", ascending=False)
    
    fig_sec = px.bar(
        sec_df,
        x="CombinedLabel",
        y="ChurnRate",
        text="ChurnRate",
        color="ChurnRate",
        color_continuous_scale=[[0, THEME["retained"]], [1, THEME["churn"]]],
    )
    fig_sec.update_traces(texttemplate="%{text}%", textposition="outside", textfont=dict(color=THEME["text_main"]))
    fig_sec.update_layout(
        defaults,
        coloraxis_showscale=False,
        title=dict(text="Churn Risk: Online Security & Tech Support Combination", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="", tickangle=-20, gridcolor=THEME["border"]),
        yaxis=dict(title="Churn Rate (%)", range=[0, max(sec_df["ChurnRate"].max() * 1.25, 40)], gridcolor=THEME["border"]),
        height=300,
    )

    # --- 5. Charges Scatter Plot ---
    sample_df = df.sample(min(len(df), 800), random_state=42)
    fig_charges = px.scatter(
        sample_df,
        x="MonthlyCharges",
        y="TotalCharges",
        color="Churn",
        color_discrete_map={"Yes": THEME["churn"], "No": THEME["retained"]},
        opacity=0.65,
        hover_data=["tenure", "Contract", "PaymentMethod"],
    )
    fig_charges.update_layout(
        defaults,
        title=dict(text="Monthly Charges vs Total Cumulative Charges ($)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Monthly Charges ($/mo)", gridcolor=THEME["border"]),
        yaxis=dict(title="Total Charges ($)", gridcolor=THEME["border"]),
        height=300,
    )

    # --- 6. Addon Breadth Chart ---
    addon_df = df.groupby("AddOnCount")["ChurnNumeric"].agg(["count", "mean"]).reset_index()
    addon_df["ChurnRate"] = (addon_df["mean"] * 100).round(1)
    
    fig_addon = px.line(
        addon_df,
        x="AddOnCount",
        y="ChurnRate",
        markers=True,
        text="ChurnRate",
    )
    fig_addon.update_traces(
        line=dict(color=THEME["primary"], width=3),
        marker=dict(size=9, color=THEME["secondary"]),
        textposition="top center",
        textfont=dict(color=THEME["text_main"]),
    )
    fig_addon.update_layout(
        defaults,
        title=dict(text="Churn Rate vs Number of Active Add-on Services", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Active Add-on Count (0 to 6 Services)", tickmode="linear", tick0=0, dtick=1, gridcolor=THEME["border"]),
        yaxis=dict(title="Churn Rate (%)", range=[0, max(addon_df["ChurnRate"].max() * 1.25, 45)], gridcolor=THEME["border"]),
        height=300,
    )

    return banner, fig_demo, fig_pay, fig_tenure, fig_sec, fig_charges, fig_addon


def register_analytics_callbacks(app):
    """Registers Dash callback to dynamically update all Module B charts based on filters."""
    app.callback(
        [
            Output("analytics-cohort-banner", "children"),
            Output("chart-demographics", "figure"),
            Output("chart-payment", "figure"),
            Output("chart-tenure-cohorts", "figure"),
            Output("chart-security-support", "figure"),
            Output("chart-charges-scatter", "figure"),
            Output("chart-addon-depth", "figure"),
        ],
        [
            Input("filter-contract", "value"),
            Input("filter-internet", "value"),
            Input("filter-payment", "value"),
            Input("filter-senior", "value"),
        ]
    )(update_analytics_charts)
