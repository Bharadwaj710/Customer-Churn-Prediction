"""
src/dashboard/tabs/model_evaluation.py
-------------------------------------
Module C: Model Evaluation & Benchmarking.
Displays verified Phase 7 comparative metrics, combined ROC curves, combined Precision-Recall
curves (strictly labeled Average Precision / AP), interactive confusion matrices,
validation threshold trade-offs, and generalization gap analysis.
"""

from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from src.dashboard.config import THEME, get_plotly_layout_defaults
from src.dashboard.data_loader import (
    get_test_predictions,
    get_model_comparison_metrics,
    get_validation_threshold_data,
)
from src.dashboard.components.cards import render_card_container


def create_model_evaluation_tab():
    """Generates Module C layout with verified comparative benchmarks and diagnostic plots."""
    preds_df = get_test_predictions()
    comp_json = get_model_comparison_metrics()
    val_thresh_df = get_validation_threshold_data()
    defaults = get_plotly_layout_defaults()

    y_true = preds_df["Actual_Churn"].values
    lr_prob = preds_df["LR_Prob"].values
    rf_prob = preds_df["RF_Prob"].values
    nn_prob = preds_df["NN_Prob"].values

    # --- 1. Combined ROC Curves ---
    fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_prob)
    fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_prob)
    fpr_nn, tpr_nn, _ = roc_curve(y_true, nn_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_rf, y=tpr_rf, mode="lines",
        name="Random Forest (ROC-AUC = 0.8434)",
        line=dict(color=THEME["primary"], width=2.5),
        hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra>Random Forest</extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=fpr_lr, y=tpr_lr, mode="lines",
        name="Logistic Regression (ROC-AUC = 0.8417)",
        line=dict(color=THEME["secondary"], width=2),
        hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra>Logistic Regression</extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=fpr_nn, y=tpr_nn, mode="lines",
        name="Neural Network (ROC-AUC = 0.8409)",
        line=dict(color=THEME["accent"], width=2, dash="dash"),
        hovertemplate="FPR: %{x:.3f}<br>TPR (Recall): %{y:.3f}<extra>Neural Network</extra>",
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Chance Diagonal (AUC = 0.5000)",
        line=dict(color=THEME["text_dim"], width=1, dash="dot"),
        hoverinfo="none",
    ))
    fig_roc.update_layout(
        defaults,
        title=dict(text="Combined ROC Curves (Discrimination Concordance)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[-0.02, 1.02], gridcolor=THEME["border"]),
        yaxis=dict(title="True Positive Rate (Recall / Sensitivity)", range=[-0.02, 1.02], gridcolor=THEME["border"]),
        legend=dict(orientation="v", yanchor="bottom", y=0.04, xanchor="right", x=0.98),
        height=340,
    )

    # --- 2. Combined Precision-Recall Curves (AP Labeled) ---
    p_rf, r_rf, _ = precision_recall_curve(y_true, rf_prob)
    p_lr, r_lr, _ = precision_recall_curve(y_true, lr_prob)
    p_nn, r_nn, _ = precision_recall_curve(y_true, nn_prob)
    prevalence = float(y_true.mean())

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=r_rf, y=p_rf, mode="lines",
        name="Random Forest (AP = 0.6590)",
        line=dict(color=THEME["primary"], width=2.5),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>Random Forest</extra>",
    ))
    fig_pr.add_trace(go.Scatter(
        x=r_lr, y=p_lr, mode="lines",
        name="Logistic Regression (AP = 0.6331)",
        line=dict(color=THEME["secondary"], width=2),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>Logistic Regression</extra>",
    ))
    fig_pr.add_trace(go.Scatter(
        x=r_nn, y=p_nn, mode="lines",
        name="Neural Network (AP = 0.6322)",
        line=dict(color=THEME["accent"], width=2, dash="dash"),
        hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra>Neural Network</extra>",
    ))
    fig_pr.add_trace(go.Scatter(
        x=[0, 1], y=[prevalence, prevalence], mode="lines",
        name=f"Class Prevalence ({prevalence*100:.1f}%)",
        line=dict(color=THEME["text_dim"], width=1, dash="dot"),
        hoverinfo="none",
    ))
    fig_pr.update_layout(
        defaults,
        title=dict(text="Combined Precision-Recall Curves (Average Precision)", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Recall (Sensitivity)", range=[-0.02, 1.02], gridcolor=THEME["border"]),
        yaxis=dict(title="Precision (Positive Predictive Value)", range=[-0.02, 1.02], gridcolor=THEME["border"]),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98),
        height=340,
    )

    # --- 3. Training vs Testing Generalization Gap Bar Chart ---
    gen_data = [
        {"Model": "Logistic Regression", "Split": "Train", "Metric": "Accuracy", "Value": 0.7533},
        {"Model": "Logistic Regression", "Split": "Test", "Metric": "Accuracy", "Value": 0.7381},
        {"Model": "Logistic Regression", "Split": "Train", "Metric": "F1-Score", "Value": 0.6344},
        {"Model": "Logistic Regression", "Split": "Test", "Metric": "F1-Score", "Value": 0.6136},
        {"Model": "Logistic Regression", "Split": "Train", "Metric": "ROC-AUC", "Value": 0.8492},
        {"Model": "Logistic Regression", "Split": "Test", "Metric": "ROC-AUC", "Value": 0.8417},

        {"Model": "Random Forest", "Split": "Train", "Metric": "Accuracy", "Value": 0.8126},
        {"Model": "Random Forest", "Split": "Test", "Metric": "Accuracy", "Value": 0.7573},
        {"Model": "Random Forest", "Split": "Train", "Metric": "F1-Score", "Value": 0.7146},
        {"Model": "Random Forest", "Split": "Test", "Metric": "F1-Score", "Value": 0.6330},
        {"Model": "Random Forest", "Split": "Train", "Metric": "ROC-AUC", "Value": 0.9160},
        {"Model": "Random Forest", "Split": "Test", "Metric": "ROC-AUC", "Value": 0.8434},

        {"Model": "Neural Network (MLP)", "Split": "Train", "Metric": "Accuracy", "Value": 0.7632},
        {"Model": "Neural Network (MLP)", "Split": "Test", "Metric": "Accuracy", "Value": 0.7459},
        {"Model": "Neural Network (MLP)", "Split": "Train", "Metric": "F1-Score", "Value": 0.6450},
        {"Model": "Neural Network (MLP)", "Split": "Test", "Metric": "F1-Score", "Value": 0.6200},
        {"Model": "Neural Network (MLP)", "Split": "Train", "Metric": "ROC-AUC", "Value": 0.8637},
        {"Model": "Neural Network (MLP)", "Split": "Test", "Metric": "ROC-AUC", "Value": 0.8409},
    ]
    df_gen = pd.DataFrame(gen_data)

    fig_gen = px.bar(
        df_gen,
        x="Metric",
        y="Value",
        color="Split",
        barmode="group",
        facet_col="Model",
        color_discrete_map={"Train": THEME["secondary"], "Test": THEME["primary"]},
        text_auto=".3f",
    )
    fig_gen.update_layout(
        defaults,
        title=dict(text="Generalization Capacity: Training vs Unseen Held-Out Test Evaluation", font=dict(color=THEME["text_main"], size=13)),
        yaxis=dict(title="Score", range=[0.5, 1.0], gridcolor=THEME["border"]),
        height=320,
    )
    fig_gen.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>", font=dict(color=THEME["text_main"], size=12)))

    # --- Benchmark Table Markup ---
    table_rows = [
        html.Tr([
            html.Td("Accuracy"),
            html.Td("73.81%"),
            html.Td("75.73%", className="benchmark-best"),
            html.Td("74.59%"),
            html.Td("Random Forest (+1.92% vs LR)"),
        ]),
        html.Tr([
            html.Td("Precision (PPV)"),
            html.Td("50.43%"),
            html.Td("52.87%", className="benchmark-best"),
            html.Td("51.41%"),
            html.Td("Random Forest (lowest false alarm rate)"),
        ]),
        html.Tr([
            html.Td("Recall (Sensitivity)"),
            html.Td("78.34%"),
            html.Td("78.88%", className="benchmark-best"),
            html.Td("78.07%"),
            html.Td("Random Forest (catches 295 / 374 churners)"),
        ]),
        html.Tr([
            html.Td("Specificity (TNR)"),
            html.Td("72.17%"),
            html.Td("74.59%", className="benchmark-best"),
            html.Td("73.33%"),
            html.Td("Random Forest (772 / 1,035 retained true negatives)"),
        ]),
        html.Tr([
            html.Td("F1-Score (Harmonic Mean)"),
            html.Td("0.6136"),
            html.Td("0.6330", className="benchmark-best"),
            html.Td("0.6200"),
            html.Td("Random Forest (+0.0194 vs LR)"),
        ]),
        html.Tr([
            html.Td("ROC-AUC (Discrimination)"),
            html.Td("0.8417"),
            html.Td("0.8434", className="benchmark-best"),
            html.Td("0.8409"),
            html.Td("Bootstrap CI spans zero across all pairs (statistically equivalent)"),
        ]),
        html.Tr([
            html.Td("Average Precision (AP)"),
            html.Td("0.6331"),
            html.Td("0.6590", className="benchmark-best"),
            html.Td("0.6322"),
            html.Td("Random Forest (+0.0259 advantage under class imbalance)"),
        ]),
        html.Tr([
            html.Td("False Alarms (FP / 1,035)"),
            html.Td("288 (27.8%)"),
            html.Td("263 (25.4%)", className="benchmark-best"),
            html.Td("276 (26.7%)"),
            html.Td("Random Forest saves 25 false positive outreach costs vs LR"),
        ]),
    ]

    return html.Div([
        # Page Title
        html.Div(
            className="page-header",
            children=[
                html.Div([
                    html.H1("Multi-Model Benchmarking & Diagnostic Audit", className="page-title"),
                    html.P("Empirical evaluation across Logistic Regression, Random Forest, and TensorFlow/Keras MLP on unseen test data (N = 1,409).", className="page-desc"),
                ]),
                html.Div(
                    className="badge-status",
                    children=[
                        html.Span(className="status-dot"),
                        html.Span("Verified Phase 7 Artifacts"),
                    ],
                ),
            ],
        ),

        # Comparative Benchmark Table
        render_card_container(
            "Unified Test-Set Performance Benchmark (Threshold = 0.50)",
            "Strictly computed on the held-out test cohort (1,035 Retained, 374 Churned) without data leakage",
            html.Div(
                children=[
                    html.Table(
                        className="benchmark-table",
                        children=[
                            html.Thead(
                                html.Tr([
                                    html.Th("Metric"),
                                    html.Th("Logistic Regression"),
                                    html.Th("Random Forest (Recommended)"),
                                    html.Th("Neural Network (MLP)"),
                                    html.Th("Statistical & Operational Takeaway"),
                                ])
                            ),
                            html.Tbody(table_rows),
                        ],
                    ),
                    html.Div(
                        style={"marginTop": "14px", "fontSize": "0.78rem", "color": THEME["text_dim"]},
                        children=[
                            html.Strong("Bootstrap Significance Audit: "),
                            "Paired percentile bootstrap (B = 1,000, seed 42) indicates the 95% confidence intervals for ΔROC-AUC span zero across all three pairwise comparisons. "
                            "While Random Forest achieves the highest point estimates in accuracy, precision, and Average Precision (AP), the models share statistically comparable overall discriminative power."
                        ]
                    )
                ]
            )
        ),

        # Row 1: Combined ROC & Precision-Recall Curves
        html.Div(
            className="grid-2col",
            children=[
                render_card_container("Discrimination Capacity: Combined ROC", "True Positive Rate vs False Positive Rate across all thresholds", dcc.Graph(figure=fig_roc, config={"displayModeBar": False})),
                render_card_container("Imbalance Performance: Precision-Recall (AP)", "Precision vs Recall across thresholds with Average Precision (AP) rankings", dcc.Graph(figure=fig_pr, config={"displayModeBar": False})),
            ],
        ),

        # Row 2: Interactive Confusion Matrix & Validation Threshold Sensitivity
        html.Div(
            className="grid-2col",
            children=[
                render_card_container(
                    "Interactive Confusion Matrix Diagnostic",
                    "Select model and normalization mode to inspect classification counts",
                    html.Div([
                        html.Div(
                            style={"display": "flex", "gap": "16px", "marginBottom": "12px", "flexWrap": "wrap"},
                            children=[
                                html.Div([
                                    html.Label("Select Model:", style={"fontSize": "0.75rem", "color": THEME["text_dim"], "display": "block", "marginBottom": "4px"}),
                                    dcc.RadioItems(
                                        id="cm-model-selector",
                                        options=[
                                            {"label": " Logistic Regression ", "value": "LR"},
                                            {"label": " Random Forest ", "value": "RF"},
                                            {"label": " Neural Network ", "value": "NN"},
                                        ],
                                        value="RF",
                                        inline=True,
                                        style={"color": THEME["text_main"], "fontSize": "0.85rem"},
                                    ),
                                ]),
                                html.Div([
                                    html.Label("Display Format:", style={"fontSize": "0.75rem", "color": THEME["text_dim"], "display": "block", "marginBottom": "4px"}),
                                    dcc.RadioItems(
                                        id="cm-norm-selector",
                                        options=[
                                            {"label": " Absolute Counts ", "value": "counts"},
                                            {"label": " % of Test Set (N=1409) ", "value": "overall_pct"},
                                            {"label": " Row Normalized (Recall/Spec) ", "value": "row_pct"},
                                        ],
                                        value="counts",
                                        inline=True,
                                        style={"color": THEME["text_main"], "fontSize": "0.85rem"},
                                    ),
                                ]),
                            ],
                        ),
                        dcc.Graph(id="interactive-cm-chart", config={"displayModeBar": False}),
                    ])
                ),
                render_card_container(
                    "Classification Threshold Sensitivity Explorer",
                    "Empirical validation split trade-offs across thresholds tau in {0.35, 0.50, 0.65}",
                    html.Div([
                        html.Div(
                            style={"marginBottom": "12px"},
                            children=[
                                html.Label("Select Model for Threshold Trade-off:", style={"fontSize": "0.75rem", "color": THEME["text_dim"], "display": "block", "marginBottom": "4px"}),
                                dcc.RadioItems(
                                    id="thresh-model-selector",
                                    options=[
                                        {"label": " Logistic Regression ", "value": "Logistic Regression"},
                                        {"label": " Random Forest ", "value": "Random Forest"},
                                        {"label": " Neural Network (MLP) ", "value": "Neural Network (MLP)"},
                                    ],
                                    value="Random Forest",
                                    inline=True,
                                    style={"color": THEME["text_main"], "fontSize": "0.85rem"},
                                ),
                            ],
                        ),
                        dcc.Graph(id="threshold-sensitivity-chart", config={"displayModeBar": False}),
                    ])
                ),
            ],
        ),

        # Row 3: Generalization Gap Analysis
        render_card_container(
            "Training vs Testing Generalization Gap Analysis",
            "Comparing training fit with held-out testing generalization to assess model capacity and overfitting resistance",
            html.Div([
                dcc.Graph(figure=fig_gen, config={"displayModeBar": False}),
                html.Div(
                    style={"marginTop": "14px", "padding": "12px 16px", "backgroundColor": "rgba(33, 46, 74, 0.3)", "borderRadius": "8px", "fontSize": "0.8rem", "color": THEME["text_muted"]},
                    children=[
                        html.Strong("Academic Rigor Assessment on Random Forest Generalization: "),
                        "Random Forest demonstrates an expected generalization delta between training (0.8126 accuracy, 0.9160 ROC-AUC) and test evaluations (0.7573 accuracy, 0.8434 ROC-AUC). "
                        "Because Random Forest still achieves the highest held-out test score among all models (75.73% accuracy, 0.6590 AP), this gap reflects tree depth capacity (d=10) on complex non-linear decision boundaries rather than pathological overfitting."
                    ]
                )
            ])
        ),
    ])


def update_confusion_matrix(model_key, norm_mode):
    """Computes and renders the interactive confusion matrix heatmap."""
    defaults = get_plotly_layout_defaults()
    preds_df = get_test_predictions()
    
    y_true = preds_df["Actual_Churn"].values
    if model_key == "LR":
        y_pred = preds_df["LR_Pred"].values
        title_name = "Logistic Regression"
    elif model_key == "RF":
        y_pred = preds_df["RF_Pred"].values
        title_name = "Random Forest"
    else:
        y_pred = preds_df["NN_Pred"].values
        title_name = "Neural Network (MLP)"

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    total = tn + fp + fn + tp

    matrix_counts = [[tn, fp], [fn, tp]]

    if norm_mode == "counts":
        z_vals = matrix_counts
        annot_text = [
            [f"<b>TN: {tn:,}</b><br><span style='font-size:11px'>{(tn/total)*100:.1f}%</span>",
             f"<b>FP: {fp:,}</b><br><span style='font-size:11px'>{(fp/total)*100:.1f}%</span>"],
            [f"<b>FN: {fn:,}</b><br><span style='font-size:11px'>{(fn/total)*100:.1f}%</span>",
             f"<b>TP: {tp:,}</b><br><span style='font-size:11px'>{(tp/total)*100:.1f}%</span>"]
        ]
        colorbar_title = "Count"
    elif norm_mode == "overall_pct":
        z_vals = [[(tn/total)*100, (fp/total)*100], [(fn/total)*100, (tp/total)*100]]
        annot_text = [
            [f"<b>TN: {(tn/total)*100:.2f}%</b><br><span style='font-size:11px'>{tn:,} / {total:,}</span>",
             f"<b>FP: {(fp/total)*100:.2f}%</b><br><span style='font-size:11px'>{fp:,} / {total:,}</span>"],
            [f"<b>FN: {(fn/total)*100:.2f}%</b><br><span style='font-size:11px'>{fn:,} / {total:,}</span>",
             f"<b>TP: {(tp/total)*100:.2f}%</b><br><span style='font-size:11px'>{tp:,} / {total:,}</span>"]
        ]
        colorbar_title = "% Total"
    else:  # Row normalized (Class specific recall / specificity)
        row_0_tot = tn + fp
        row_1_tot = fn + tp
        z_vals = [[(tn/row_0_tot)*100, (fp/row_0_tot)*100], [(fn/row_1_tot)*100, (tp/row_1_tot)*100]]
        annot_text = [
            [f"<b>Specificity: {(tn/row_0_tot)*100:.1f}%</b><br><span style='font-size:11px'>{tn:,} / {row_0_tot:,}</span>",
             f"<b>FPR: {(fp/row_0_tot)*100:.1f}%</b><br><span style='font-size:11px'>{fp:,} / {row_0_tot:,}</span>"],
            [f"<b>FNR: {(fn/row_1_tot)*100:.1f}%</b><br><span style='font-size:11px'>{fn:,} / {row_1_tot:,}</span>",
             f"<b>Recall: {(tp/row_1_tot)*100:.1f}%</b><br><span style='font-size:11px'>{tp:,} / {row_1_tot:,}</span>"]
        ]
        colorbar_title = "Row %"

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=["Predicted: Retained (0)", "Predicted: Churned (1)"],
        y=["Actual: Retained (0)", "Actual: Churned (1)"],
        text=annot_text,
        texttemplate="%{text}",
        colorscale=[[0, "#131C31"], [0.5, "#312E81"], [1, "#4F46E5"]],
        showscale=True,
        colorbar=dict(title=colorbar_title, tickfont=dict(color=THEME["text_muted"])),
    ))
    fig.update_layout(
        defaults,
        yaxis=dict(autorange="reversed", gridcolor=THEME["border"]),
        title=dict(text=f"{title_name} Confusion Matrix ({norm_mode.replace('_', ' ').title()})", font=dict(color=THEME["text_main"], size=13)),
        height=280,
        margin=dict(l=60, r=40, t=40, b=40),
    )
    return fig


def update_threshold_chart(model_name):
    """Computes and renders threshold sensitivity curves."""
    defaults = get_plotly_layout_defaults()
    val_df = get_validation_threshold_data()
    
    subset = val_df[val_df["Model"] == model_name].sort_values(by="Threshold")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=subset["Threshold"], y=subset["Accuracy"] * 100,
        mode="lines+markers", name="Accuracy (%)",
        line=dict(color=THEME["text_main"], width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=subset["Threshold"], y=subset["Precision"] * 100,
        mode="lines+markers", name="Precision (%)",
        line=dict(color=THEME["secondary"], width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=subset["Threshold"], y=subset["Recall"] * 100,
        mode="lines+markers", name="Recall (%)",
        line=dict(color=THEME["churn"], width=2.5),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=subset["Threshold"], y=subset["Specificity"] * 100,
        mode="lines+markers", name="Specificity (%)",
        line=dict(color=THEME["retained"], width=2),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=subset["Threshold"], y=subset["F1_Score"] * 100,
        mode="lines+markers", name="F1-Score (%)",
        line=dict(color=THEME["moderate"], width=2, dash="dash"),
        marker=dict(size=8),
    ))

    fig.update_layout(
        defaults,
        title=dict(text=f"{model_name}: Precision-Recall Trade-off vs Decision Threshold", font=dict(color=THEME["text_main"], size=13)),
        xaxis=dict(title="Classification Decision Threshold (tau)", tickvals=[0.35, 0.50, 0.65], gridcolor=THEME["border"]),
        yaxis=dict(title="Metric Score (%)", range=[35, 100], gridcolor=THEME["border"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
    )
    return fig


def register_model_callbacks(app):
    """Registers callbacks for interactive confusion matrix and threshold sensitivity charts."""
    app.callback(
        Output("interactive-cm-chart", "figure"),
        [
            Input("cm-model-selector", "value"),
            Input("cm-norm-selector", "value"),
        ]
    )(update_confusion_matrix)

    app.callback(
        Output("threshold-sensitivity-chart", "figure"),
        Input("thresh-model-selector", "value")
    )(update_threshold_chart)
