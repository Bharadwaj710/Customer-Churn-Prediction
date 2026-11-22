"""
src/dashboard/tabs/churn_simulator.py
-------------------------------------
Modules D & E: Customer Churn Prediction Simulator & Retention Intelligence Playbook.
Accepts raw customer attributes, passes them through preprocessor.joblib,
runs inference across Logistic Regression, Random Forest, or Keras MLP,
and displays threshold-dependent risk scores, model-specific explanations,
and domain-specific retention recommendations.
"""

from dash import html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from src.dashboard.config import THEME
from src.dashboard.model_inference import predict_churn
from src.dashboard.recommendations import generate_retention_playbook
from src.dashboard.components.cards import render_card_container, render_gauge_chart


def create_simulator_tab():
    """Generates Modules D & E interactive customer prediction layout."""
    return html.Div([
        # Page Title
        html.Div(
            className="page-header",
            children=[
                html.Div([
                    html.H1("Customer Churn Prediction Simulator", className="page-title"),
                    html.P("Real-time inference using fitted preprocessor and trained machine learning models with customized retention playbooks.", className="page-desc"),
                ]),
                html.Div(
                    className="badge-status",
                    children=[
                        html.Span(className="status-dot"),
                        html.Span("Zero Data Leakage Pipeline Active"),
                    ],
                ),
            ],
        ),

        # 2-Column Layout: Left Form, Right Results
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1.1fr 1fr", "gap": "24px"},
            children=[
                # Left Column: Input Form
                html.Div([
                    render_card_container(
                        "Subscriber Profile & Model Controls",
                        "Configure customer attributes and inference parameters",
                        html.Div([
                            # Model & Threshold Controls
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "marginBottom": "16px"},
                                children=[
                                    html.Div([
                                        html.Label("Machine Learning Model", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-model-choice",
                                            options=[
                                                {"label": "Random Forest (Recommended)", "value": "Random Forest"},
                                                {"label": "Logistic Regression (Interpretable)", "value": "Logistic Regression"},
                                                {"label": "Neural Network (TensorFlow MLP)", "value": "Neural Network"},
                                            ],
                                            value="Random Forest",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Div([
                                            html.Label("Decision Threshold (tau): ", className="filter-label", style={"display": "inline"}),
                                            html.Span(id="threshold-val-display", style={"color": THEME["primary"], "fontWeight": "700"}),
                                        ]),
                                        dcc.Slider(
                                            id="sim-threshold-slider",
                                            min=0.10,
                                            max=0.90,
                                            step=0.05,
                                            value=0.50,
                                            marks={0.35: "0.35", 0.50: "0.50", 0.65: "0.65"},
                                            tooltip={"placement": "bottom", "always_visible": False},
                                        ),
                                    ]),
                                ]
                            ),

                            # Section 1: Contract & Tenure
                            html.Div("1. Contract & Account Tenure", className="form-section-title"),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                                children=[
                                    html.Div([
                                        html.Label("Contract Commitment", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-contract",
                                            options=[
                                                {"label": "Month-to-month", "value": "Month-to-month"},
                                                {"label": "One year", "value": "One year"},
                                                {"label": "Two year", "value": "Two year"},
                                            ],
                                            value="Month-to-month",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Div([
                                            html.Label("Tenure (Months): ", className="filter-label", style={"display": "inline"}),
                                            html.Span(id="tenure-val-display", style={"color": THEME["secondary"], "fontWeight": "700"}),
                                        ]),
                                        dcc.Slider(
                                            id="sim-tenure",
                                            min=0,
                                            max=72,
                                            step=1,
                                            value=4,
                                            marks={0: "0", 12: "1 yr", 24: "2 yr", 48: "4 yr", 72: "6 yr"},
                                        ),
                                    ]),
                                ]
                            ),

                            # Section 2: Financials & Billing
                            html.Div("2. Monthly Charges & Payment Channel", className="form-section-title"),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                                children=[
                                    html.Div([
                                        html.Div([
                                            html.Label("Monthly Charges ($): ", className="filter-label", style={"display": "inline"}),
                                            html.Span(id="monthly-val-display", style={"color": THEME["accent"], "fontWeight": "700"}),
                                        ]),
                                        dcc.Slider(
                                            id="sim-monthly-charges",
                                            min=18.0,
                                            max=120.0,
                                            step=1.0,
                                            value=85.0,
                                            marks={20: "$20", 50: "$50", 80: "$80", 110: "$110"},
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Payment Method", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-payment-method",
                                            options=[
                                                {"label": "Electronic check", "value": "Electronic check"},
                                                {"label": "Mailed check", "value": "Mailed check"},
                                                {"label": "Bank transfer (automatic)", "value": "Bank transfer (automatic)"},
                                                {"label": "Credit card (automatic)", "value": "Credit card (automatic)"},
                                            ],
                                            value="Electronic check",
                                            clearable=False,
                                        ),
                                    ]),
                                ]
                            ),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px", "marginTop": "10px"},
                                children=[
                                    html.Div([
                                        html.Label("Paperless Billing", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-paperless",
                                            options=[{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}],
                                            value="Yes",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Total Charges Override (Leave blank for auto)", className="filter-label"),
                                        dcc.Input(
                                            id="sim-total-charges-input",
                                            type="number",
                                            placeholder="Auto: Tenure × Monthly",
                                            className="text-input",
                                        ),
                                    ]),
                                ]
                            ),

                            # Section 3: Internet & Security Services
                            html.Div("3. Internet Infrastructure & Support Services", className="form-section-title"),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                                children=[
                                    html.Div([
                                        html.Label("Internet Service", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-internet",
                                            options=[
                                                {"label": "Fiber optic", "value": "Fiber optic"},
                                                {"label": "DSL", "value": "DSL"},
                                                {"label": "No", "value": "No"},
                                            ],
                                            value="Fiber optic",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Tech Support", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-tech-support",
                                            options=[
                                                {"label": "No", "value": "No"},
                                                {"label": "Yes", "value": "Yes"},
                                                {"label": "No internet service", "value": "No internet service"},
                                            ],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                ]
                            ),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px", "marginTop": "10px"},
                                children=[
                                    html.Div([
                                        html.Label("Online Security", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-online-security",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No internet", "value": "No internet service"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Online Backup", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-online-backup",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No internet", "value": "No internet service"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Device Protection", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-device-protection",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No internet", "value": "No internet service"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                ]
                            ),

                            # Section 4: Phone & Entertainment
                            html.Div("4. Phone & Entertainment Add-ons", className="form-section-title"),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "10px"},
                                children=[
                                    html.Div([
                                        html.Label("Phone Service", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-phone-service",
                                            options=[{"label": "Yes", "value": "Yes"}, {"label": "No", "value": "No"}],
                                            value="Yes",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Multiple Lines", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-multiple-lines",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No phone", "value": "No phone service"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Streaming TV", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-streaming-tv",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No internet", "value": "No internet service"}],
                                            value="Yes",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Streaming Movies", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-streaming-movies",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}, {"label": "No internet", "value": "No internet service"}],
                                            value="Yes",
                                            clearable=False,
                                        ),
                                    ]),
                                ]
                            ),

                            # Section 5: Demographics
                            html.Div("5. Demographics", className="form-section-title"),
                            html.Div(
                                style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "10px", "marginBottom": "20px"},
                                children=[
                                    html.Div([
                                        html.Label("Gender", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-gender",
                                            options=[{"label": "Female", "value": "Female"}, {"label": "Male", "value": "Male"}],
                                            value="Female",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Senior Citizen", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-senior",
                                            options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}],
                                            value=0,
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Partner", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-partner",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Dependents", className="filter-label"),
                                        dcc.Dropdown(
                                            id="sim-dependents",
                                            options=[{"label": "No", "value": "No"}, {"label": "Yes", "value": "Yes"}],
                                            value="No",
                                            clearable=False,
                                        ),
                                    ]),
                                ]
                            ),

                            # Submit Button
                            html.Button(
                                "⚡ Calculate Churn Risk & Retention Strategy",
                                id="btn-run-simulation",
                                n_clicks=0,
                                className="btn-primary",
                            ),
                        ])
                    )
                ]),

                # Right Column: Output Panel (Simulator Results & Retention Recommendations)
                html.Div([
                    # Prediction Output Card
                    render_card_container(
                        "Inference Risk Diagnostic",
                        "Real-time model prediction output and feature influence",
                        html.Div(id="simulation-results-container")
                    ),

                    # Module E: Retention Recommendations Card
                    render_card_container(
                        "Actionable Retention Intelligence Playbook",
                        "Prescriptive retention strategies tailored to subscriber profile",
                        html.Div(id="simulation-recommendations-container")
                    ),
                ]),
            ],
        ),
    ])


def run_customer_prediction(
    n_clicks, threshold, model_name,
    contract, tenure, monthly, total_charges_input, payment, paperless,
    internet, tech_support, online_sec, online_backup, dev_protect,
    phone, mult_lines, stream_tv, stream_movies,
    gender, senior, partner, dependents
):
    """Executes customer churn inference and returns prediction diagnostics + retention playbook."""
    # Assemble dictionary
    customer_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total_charges_input,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": dev_protect,
        "TechSupport": tech_support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
    }

    # Run pipeline inference
    result = predict_churn(customer_dict, model_name=model_name, threshold=threshold)
    prob = result["probability"]
    is_churn = result["is_churn"]
    risk_tier = result["risk_tier"]
    risk_color = result["risk_color"]
    badge_class = result["badge_class"]
    explanations = result["explanations"]

    # Gauge Chart
    fig_gauge = render_gauge_chart(prob, threshold=threshold)

    # Decision Banner
    if is_churn:
        banner_title = "CRITICAL CHURN RISK PREDICTED"
        banner_desc = f"Probability ({prob*100:.1f}%) exceeds decision threshold ({threshold*100:.0f}%). Proactive retention intervention required."
        banner_style = {
            "backgroundColor": "rgba(239, 68, 68, 0.15)",
            "border": "1px solid rgba(239, 68, 68, 0.4)",
            "color": "#F87171",
        }
    else:
        banner_title = "ACCOUNT LIKELY RETAINED"
        banner_desc = f"Probability ({prob*100:.1f}%) is below decision threshold ({threshold*100:.0f}%). Account is statistically stable."
        banner_style = {
            "backgroundColor": "rgba(16, 185, 129, 0.15)",
            "border": "1px solid rgba(16, 185, 129, 0.4)",
            "color": "#34D399",
        }

    # Model-specific explanation markup
    explanation_items = []
    if explanations["type"] == "logistic_regression":
        risk_factors = explanations.get("top_risk_factors", [])
        prot_factors = explanations.get("top_protective_factors", [])
        
        explanation_items.append(html.Div([
            html.Div("Top Risk Escalation Drivers (Log-Odds Impact):", style={"fontWeight": "600", "color": THEME["churn"], "fontSize": "0.82rem", "marginBottom": "4px"}),
            html.Ul([
                html.Li(f"{f['Feature']}: +{f['Contribution']:.3f} log-odds", style={"fontSize": "0.78rem", "color": THEME["text_muted"]})
                for f in risk_factors
            ], style={"paddingLeft": "18px", "marginBottom": "8px"}),
            html.Div("Top Protective Factors (Retention Drivers):", style={"fontWeight": "600", "color": THEME["retained"], "fontSize": "0.82rem", "marginBottom": "4px"}),
            html.Ul([
                html.Li(f"{f['Feature']}: {f['Contribution']:.3f} log-odds", style={"fontSize": "0.78rem", "color": THEME["text_muted"]})
                for f in prot_factors
            ], style={"paddingLeft": "18px"}),
        ]))
    elif explanations["type"] == "random_forest":
        top_features = explanations.get("top_features", [])
        explanation_items.append(html.Div([
            html.Div("Random Forest Key Split Features (MDI Importance):", style={"fontWeight": "600", "color": THEME["primary"], "fontSize": "0.82rem", "marginBottom": "4px"}),
            html.Ul([
                html.Li(f"{f['Feature']} (Global Weight: {f['Importance']*100:.1f}%): Active value = {f['Customer_Value']:.2f}", style={"fontSize": "0.78rem", "color": THEME["text_muted"]})
                for f in top_features
            ], style={"paddingLeft": "18px"}),
        ]))
    else:  # Neural Network
        explanation_items.append(html.Div([
            html.Div("Neural Network Representation Diagnostic:", style={"fontWeight": "600", "color": THEME["accent"], "fontSize": "0.82rem", "marginBottom": "4px"}),
            html.P(f"Nonlinear activation output: {prob:.4f} across 64-relu / 32-relu dense layers. Evaluated with contract={explanations['contract']}, tenure={explanations['tenure_months']} mos, monthly=${explanations['monthly_charges']:.2f}.", style={"fontSize": "0.78rem", "color": THEME["text_muted"]}),
        ]))

    # Render Inference Diagnostic Container
    results_markup = html.Div([
        # Top Banner
        html.Div(
            style={
                "padding": "12px 16px",
                "borderRadius": "8px",
                "marginBottom": "16px",
                **banner_style,
            },
            children=[
                html.Div(banner_title, style={"fontWeight": "800", "fontSize": "0.95rem"}),
                html.Div(banner_desc, style={"fontSize": "0.8rem", "marginTop": "2px"}),
            ]
        ),

        # Gauge & Risk Tier Row
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "16px", "marginBottom": "12px"},
            children=[
                html.Div(dcc.Graph(figure=fig_gauge, config={"displayModeBar": False}), style={"flex": "1.2"}),
                html.Div(
                    style={"flex": "0.8", "display": "flex", "flexDirection": "column", "gap": "8px"},
                    children=[
                        html.Div([
                            html.Span("Risk Tier: ", style={"fontSize": "0.78rem", "color": THEME["text_dim"]}),
                            html.Span(risk_tier, className=f"risk-badge {badge_class}"),
                        ]),
                        html.Div([
                            html.Span("Active Model: ", style={"fontSize": "0.78rem", "color": THEME["text_dim"]}),
                            html.Span(model_name, style={"fontSize": "0.82rem", "fontWeight": "600", "color": THEME["text_main"]}),
                        ]),
                        html.Div([
                            html.Span("Applied Cutoff: ", style={"fontSize": "0.78rem", "color": THEME["text_dim"]}),
                            html.Span(f"tau = {threshold:.2f}", style={"fontSize": "0.82rem", "fontWeight": "600", "color": THEME["primary"]}),
                        ]),
                    ]
                ),
            ]
        ),

        # Feature Influences
        html.Div(
            style={"padding": "12px 14px", "backgroundColor": "rgba(19, 28, 49, 0.6)", "borderRadius": "8px", "border": f"1px solid {THEME['border']}"},
            children=explanation_items,
        )
    ])

    # --- Module E: Retention Recommendations Playbook ---
    playbook = generate_retention_playbook(customer_dict, prob, risk_tier)
    
    playbook_cards = []
    for act in playbook["actions"]:
        priority = act["priority"]
        badge_class = f"rec-badge-{priority.lower()}"
        
        playbook_cards.append(html.Div(
            className=f"rec-card priority-{priority}",
            children=[
                html.Div(
                    className="rec-header",
                    children=[
                        html.Span(f"{act['category']} · {act['title']}", className="rec-title"),
                        html.Span(f"{priority} Priority", className=f"rec-badge {badge_class}"),
                    ]
                ),
                html.Div(html.Strong("Observation: "), style={"fontSize": "0.76rem", "color": THEME["text_dim"]}),
                html.P(act["observation"], className="rec-text"),
                html.Div(html.Strong("Recommended Intervention: "), style={"fontSize": "0.76rem", "color": THEME["text_dim"], "marginTop": "4px"}),
                html.P(act["intervention"], className="rec-text", style={"color": THEME["text_main"], "fontWeight": "500"}),
                html.Div(f"Expected Outcome: {act['impact']}", className="rec-impact"),
            ]
        ))

    recs_markup = html.Div([
        html.Div(
            playbook["summary"],
            style={"fontSize": "0.85rem", "fontWeight": "600", "color": risk_color, "marginBottom": "14px"}
        ),
        html.Div(playbook_cards),
        html.Div(
            playbook["disclaimer"],
            style={"fontSize": "0.74rem", "color": THEME["text_dim"], "marginTop": "12px", "padding": "8px 12px", "backgroundColor": "rgba(33, 46, 74, 0.3)", "borderRadius": "6px"}
        )
    ])

    return results_markup, recs_markup


def register_simulator_callbacks(app):
    """Registers callbacks for dynamic value displays and simulation prediction output."""
    
    # Live label updates for sliders
    @app.callback(Output("threshold-val-display", "children"), Input("sim-threshold-slider", "value"))
    def update_threshold_display(val):
        return f"{val:.2f}"

    @app.callback(Output("tenure-val-display", "children"), Input("sim-tenure", "value"))
    def update_tenure_display(val):
        return f"{val} mo"

    @app.callback(Output("monthly-val-display", "children"), Input("sim-monthly-charges", "value"))
    def update_monthly_display(val):
        return f"${val:.2f}"

    # Main Simulation Callback
    app.callback(
        [
            Output("simulation-results-container", "children"),
            Output("simulation-recommendations-container", "children"),
        ],
        [
            Input("btn-run-simulation", "n_clicks"),
            Input("sim-threshold-slider", "value"),
            Input("sim-model-choice", "value"),
        ],
        [
            State("sim-contract", "value"),
            State("sim-tenure", "value"),
            State("sim-monthly-charges", "value"),
            State("sim-total-charges-input", "value"),
            State("sim-payment-method", "value"),
            State("sim-paperless", "value"),
            State("sim-internet", "value"),
            State("sim-tech-support", "value"),
            State("sim-online-security", "value"),
            State("sim-online-backup", "value"),
            State("sim-device-protection", "value"),
            State("sim-phone-service", "value"),
            State("sim-multiple-lines", "value"),
            State("sim-streaming-tv", "value"),
            State("sim-streaming-movies", "value"),
            State("sim-gender", "value"),
            State("sim-senior", "value"),
            State("sim-partner", "value"),
            State("sim-dependents", "value"),
        ]
    )(run_customer_prediction)
