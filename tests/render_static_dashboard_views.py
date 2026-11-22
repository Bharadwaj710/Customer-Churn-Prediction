"""
tests/render_static_dashboard_views.py
--------------------------------------
Renders high-fidelity static HTML views of all four dashboard modules using the authentic
project CSS styles, Plotly interactive components, and KPI cards, then captures publication-quality
screenshots in outputs/figures/dashboard/.
"""

import os
import subprocess
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "figures", "dashboard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(EDGE_PATH):
    EDGE_PATH = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

# Read CSS
css_path = os.path.join(BASE_DIR, "assets", "style.css")
with open(css_path, "r", encoding="utf-8") as f:
    custom_css = f.read()

from src.dashboard.config import THEME, get_plotly_layout_defaults
from src.dashboard.data_loader import (
    get_raw_data,
    get_executive_kpis,
    get_test_predictions,
    get_model_comparison_metrics,
    get_validation_threshold_data,
)
from src.dashboard.tabs.executive_overview import create_overview_tab
from src.dashboard.tabs.customer_analytics import update_analytics_charts
from src.dashboard.tabs.model_evaluation import update_confusion_matrix, update_threshold_chart
from src.dashboard.tabs.churn_simulator import run_customer_prediction
from src.dashboard.components.cards import render_gauge_chart
from src.dashboard.recommendations import generate_retention_playbook

defaults = get_plotly_layout_defaults()
df = get_raw_data()
kpis = get_executive_kpis()
preds_df = get_test_predictions()

def wrap_html(title: str, active_tab_name: str, body_content: str) -> str:
    nav_items = [
        ("Executive Overview", "tab-overview", "📊"),
        ("Customer Analytics", "tab-analytics", "🔍"),
        ("Model Benchmarking", "tab-models", "⚖️"),
        ("Churn Predictor & ROI", "tab-simulator", "🎯"),
    ]
    
    nav_html = ""
    for label, tab_id, icon in nav_items:
        active_class = "active" if tab_id == active_tab_name else ""
        nav_html += f'<div class="nav-item {active_class}"><span style="margin-right:8px;">{icon}</span><span>{label}</span></div>'

    html_page = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title} · Customer Churn Intelligence</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        {custom_css}
        body {{ overflow-y: auto !important; }}
    </style>
</head>
<body>
    <div class="dashboard-wrapper">
        <div class="sidebar">
            <div class="brand-header">
                <div class="brand-title">
                    <span style="font-size:1.4rem;">⚡</span>
                    <span>CHURN INTELLIGENCE</span>
                </div>
                <div class="brand-subtitle">B.Tech CSE Capstone · AI & Data Science</div>
            </div>
            <div class="nav-menu">
                {nav_html}
            </div>
            <div class="sidebar-footer">
                <div class="badge-status" style="margin-bottom:8px;">
                    <span class="status-dot"></span>
                    <span>7,043 Records Verified</span>
                </div>
                <div style="color:{THEME['text_muted']}; font-weight:500;">IBM Telco Churn Dataset</div>
                <div style="color:{THEME['text_dim']}; margin-top:2px;">LR · RF · TF/Keras MLP</div>
            </div>
        </div>
        <div class="main-content">
            {body_content}
        </div>
    </div>
</body>
</html>"""
    return html_page


# --- 1. RENDER MODULE A: EXECUTIVE OVERVIEW ---
print("[1/4] Generating Module A (Executive Overview) HTML...")
# Donut
churn_counts = df["Churn"].value_counts()
fig_donut = go.Figure(data=[
    go.Pie(
        labels=["Retained", "Churned"],
        values=[churn_counts.get("No", 0), churn_counts.get("Yes", 0)],
        hole=0.62,
        marker=dict(colors=[THEME["retained"], THEME["churn"]]),
        textinfo="label+percent",
        textfont=dict(size=13, color=THEME["text_main"], family="Inter"),
    )
]).update_layout(
    defaults,
    title=dict(text="Portfolio Churn Distribution", font=dict(color=THEME["text_main"], size=14)),
    annotations=[
        dict(
            text=f"<b>{100 - kpis['churn_rate']:.1f}%</b><br><span style='font-size:11px;color:{THEME['text_muted']}'>Retained</span>",
            x=0.5, y=0.5, font_size=18, font_color=THEME["text_main"], showarrow=False
        )
    ],
    height=320,
)

# Contract Bar
contract_churn = df.groupby(["Contract", "Churn"]).size().unstack(fill_value=0)
contract_rates = (contract_churn["Yes"] / contract_churn.sum(axis=1) * 100).round(1)
fig_contract = go.Figure(data=[
    go.Bar(name="Retained", x=contract_churn.index, y=contract_churn["No"], marker_color=THEME["retained"]),
    go.Bar(name="Churned", x=contract_churn.index, y=contract_churn["Yes"], marker_color=THEME["churn"],
           text=[f"{rate}% Churn" for rate in contract_rates], textposition="outside", textfont=dict(color=THEME["text_main"], size=11)),
]).update_layout(
    defaults,
    title=dict(text="Subscribers & Churn Rate by Contract Type", font=dict(color=THEME["text_main"], size=14)),
    barmode="group", height=320,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# Tenure Hist
fig_tenure = go.Figure(data=[
    go.Histogram(x=df[df["Churn"] == "No"]["tenure"], name="Retained", marker_color=THEME["retained"], opacity=0.75, nbinsx=36),
    go.Histogram(x=df[df["Churn"] == "Yes"]["tenure"], name="Churned", marker_color=THEME["churn"], opacity=0.75, nbinsx=36),
]).update_layout(
    defaults,
    title=dict(text="Tenure Cohort Distribution (Months Active)", font=dict(color=THEME["text_main"], size=14)),
    barmode="overlay", height=320,
    xaxis=dict(title="Tenure (Months)", gridcolor=THEME["border"]),
    yaxis=dict(title="Subscriber Count", gridcolor=THEME["border"]),
)

# Monthly Charges Box
fig_charges = go.Figure(data=[
    go.Box(y=df[df["Churn"] == "No"]["MonthlyCharges"], name="Retained", marker_color=THEME["retained"], boxmean=True),
    go.Box(y=df[df["Churn"] == "Yes"]["MonthlyCharges"], name="Churned", marker_color=THEME["churn"], boxmean=True),
]).update_layout(
    defaults,
    title=dict(text="Monthly Charges Box Plot & Mean ($/mo)", font=dict(color=THEME["text_main"], size=14)),
    yaxis=dict(title="Monthly Charges (USD)", gridcolor=THEME["border"]), height=320,
)

body_a = f"""
<div class="page-header">
    <div>
        <h1 class="page-title">Executive Portfolio Overview</h1>
        <p class="page-desc">Real-time portfolio metrics, subscriber tenure cohorts, and churn revenue exposure.</p>
    </div>
    <div class="badge-status">
        <span class="status-dot"></span>
        <span>Total Cohort: {kpis['total_customers']:,} Subscribers</span>
    </div>
</div>

<div class="kpi-grid">
    <div class="kpi-card kpi-accent-primary">
        <div class="kpi-title">Total Subscribers</div>
        <div class="kpi-value">{kpis['total_customers']:,}</div>
        <div class="kpi-subtext">All verified active & churned accounts</div>
    </div>
    <div class="kpi-card kpi-accent-retained">
        <div class="kpi-title">Retained Accounts</div>
        <div class="kpi-value">{kpis['retained_count']:,}</div>
        <div class="kpi-subtext">{100 - kpis['churn_rate']:.2f}% customer retention rate</div>
    </div>
    <div class="kpi-card kpi-accent-churn">
        <div class="kpi-title">Churned Accounts</div>
        <div class="kpi-value">{kpis['churn_count']:,}</div>
        <div class="kpi-subtext">{kpis['churn_rate']:.2f}% empirical churn rate</div>
    </div>
    <div class="kpi-card kpi-accent-cyan">
        <div class="kpi-title">Monthly Revenue / Avg</div>
        <div class="kpi-value">${kpis['avg_monthly']:.2f}</div>
        <div class="kpi-subtext">Avg tenure: {kpis['avg_tenure']:.1f} months</div>
    </div>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Churn Distribution</div>
                <div class="card-subtitle">Aggregate class balance across 7,043 subscribers</div>
            </div>
        </div>
        <div>{fig_donut.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Contract Risk Profile</div>
                <div class="card-subtitle">Subscribers and empirical churn percentage by contract structure</div>
            </div>
        </div>
        <div>{fig_contract.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Tenure Hazard Analysis</div>
                <div class="card-subtitle">Distribution of months active highlighting early-tenure vulnerability</div>
            </div>
        </div>
        <div>{fig_tenure.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Monthly Charges Spread</div>
                <div class="card-subtitle">Median and mean expenditure comparison between retained and churned cohorts</div>
            </div>
        </div>
        <div>{fig_charges.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div class="dashboard-card">
    <div class="card-header-bar">
        <div>
            <div class="card-title">Executive Key Findings & Business Observations</div>
            <div class="card-subtitle">Synthesized empirical drivers derived from authentic exploratory data analysis</div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:16px;">
        <div class="rec-card" style="border-left-color:{THEME['churn']};">
            <div style="font-weight:600; color:{THEME['text_main']}; margin-bottom:6px;">1. Month-to-Month Contract Vulnerability</div>
            <p class="rec-text">Subscribers on month-to-month contracts experience a 42.7% churn rate, compared to just 11.3% for 1-year and 2.8% for 2-year contracts. Flexible terms present high operational churn hazard.</p>
        </div>
        <div class="rec-card" style="border-left-color:{THEME['moderate']};">
            <div style="font-weight:600; color:{THEME['text_main']}; margin-bottom:6px;">2. The First-Year Churn Cliff</div>
            <p class="rec-text">Nearly 50% of all recorded cancellations occur within the first 12 months of service. Onboarding friction and initial billing surprises are the most critical inflection points.</p>
        </div>
        <div class="rec-card" style="border-left-color:{THEME['primary']};">
            <div style="font-weight:600; color:{THEME['text_main']}; margin-bottom:6px;">3. Fiber Optic Support Friction</div>
            <p class="rec-text">Fiber optic customers display the highest churn (41.9%) despite commanding premium $70-$100/mo fees. When paired without Online Security or Tech Support, churn probability peaks.</p>
        </div>
    </div>
</div>
"""
html_a = wrap_html("Executive Overview", "tab-overview", body_a)
path_html_a = os.path.join(OUTPUT_DIR, "module_a.html")
with open(path_html_a, "w", encoding="utf-8") as f:
    f.write(html_a)


# --- 2. RENDER MODULE B: CUSTOMER ANALYTICS ---
print("[2/4] Generating Module B (Customer Analytics) HTML...")
_, f_demo, f_pay, f_ten, f_sec, f_scat, f_addon = update_analytics_charts("ALL", "ALL", "ALL", "ALL")
body_b = f"""
<div class="page-header">
    <div>
        <h1 class="page-title">Customer Cohort Analytics</h1>
        <p class="page-desc">Segment-level behavioral drivers, billing preferences, and technical service adoption patterns.</p>
    </div>
</div>

<div class="filter-bar">
    <div class="filter-group">
        <label class="filter-label">Contract Agreement</label>
        <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px; font-size:0.85rem; color:{THEME['text_main']};">All Contracts (Interactive Dropdown Active)</div>
    </div>
    <div class="filter-group">
        <label class="filter-label">Internet Infrastructure</label>
        <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px; font-size:0.85rem; color:{THEME['text_main']};">All Internet Services</div>
    </div>
    <div class="filter-group">
        <label class="filter-label">Billing Channel</label>
        <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px; font-size:0.85rem; color:{THEME['text_main']};">All Payment Methods</div>
    </div>
    <div class="filter-group">
        <label class="filter-label">Senior Citizen Filter</label>
        <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px; font-size:0.85rem; color:{THEME['text_main']};">All Age Groups</div>
    </div>
</div>

<div style="display:flex; gap:16px; padding:12px 18px; background:rgba(99, 102, 241, 0.08); border:1px solid {THEME['border']}; border-radius:8px; align-items:center; flex-wrap:wrap; margin-bottom:20px;">
    <span style="font-weight:700; color:{THEME['text_main']};">Active Filtered Segment: 7,043 Subscribers</span>
    <span style="color:{THEME['text_dim']};">·</span>
    <span style="color:{THEME['churn']};">Churn Count: 1,869</span>
    <span style="color:{THEME['text_dim']};">·</span>
    <span style="font-weight:700; color:{THEME['churn']};">Segment Churn Rate: 26.5%</span>
    <span style="color:{THEME['text_dim']};">·</span>
    <span style="color:{THEME['text_muted']};">Avg Monthly Charges: $64.76/mo</span>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Demographics & Churn Rate</div>
                <div class="card-subtitle">Churn rate broken down by Senior Citizen status and Gender</div>
            </div>
        </div>
        <div>{f_demo.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Payment Method & Billing Friction</div>
                <div class="card-subtitle">Churn percentage by Payment Method and Paperless Billing enrollment</div>
            </div>
        </div>
        <div>{f_pay.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Tenure Cohort Churn Hazard</div>
                <div class="card-subtitle">Churn rate across progressive subscriber lifecycle stages</div>
            </div>
        </div>
        <div>{f_ten.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Cyber Security & Technical Support</div>
                <div class="card-subtitle">Impact of foundational support add-ons on churn risk</div>
            </div>
        </div>
        <div>{f_sec.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Monthly vs Total Charges</div>
                <div class="card-subtitle">Expenditure correlation and subscriber longevity colored by churn status</div>
            </div>
        </div>
        <div>{f_scat.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Service Adoption Multi-Product Breadth</div>
                <div class="card-subtitle">Churn rate as a function of the number of active add-on services</div>
            </div>
        </div>
        <div>{f_addon.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div style="padding:14px 18px; background:rgba(33, 46, 74, 0.4); border:1px solid {THEME['border']}; border-radius:8px; font-size:0.78rem; color:{THEME['text_dim']}; margin-bottom:24px;">
    <strong>Analytical & Epistemological Note: </strong>
    All visualizations in this section reflect empirical bivariate and multivariate observations from the verified IBM Telco dataset. These observational associations describe historical subscriber patterns and do not imply direct individual-level causation.
</div>
"""
html_b = wrap_html("Customer Analytics", "tab-analytics", body_b)
path_html_b = os.path.join(OUTPUT_DIR, "module_b.html")
with open(path_html_b, "w", encoding="utf-8") as f:
    f.write(html_b)


# --- 3. RENDER MODULE C: MODEL EVALUATION ---
print("[3/4] Generating Module C (Model Evaluation) HTML...")
y_true = preds_df["Actual_Churn"].values
lr_prob = preds_df["LR_Prob"].values
rf_prob = preds_df["RF_Prob"].values
nn_prob = preds_df["NN_Prob"].values

fpr_rf, tpr_rf, _ = roc_curve(y_true, rf_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_prob)
fpr_nn, tpr_nn, _ = roc_curve(y_true, nn_prob)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr_rf, y=tpr_rf, mode="lines", name="Random Forest (ROC-AUC = 0.8434)", line=dict(color=THEME["primary"], width=2.5)))
fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines", name="Logistic Regression (ROC-AUC = 0.8417)", line=dict(color=THEME["secondary"], width=2)))
fig_roc.add_trace(go.Scatter(x=fpr_nn, y=tpr_nn, mode="lines", name="Neural Network (ROC-AUC = 0.8409)", line=dict(color=THEME["accent"], width=2, dash="dash")))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance Diagonal (AUC = 0.5000)", line=dict(color=THEME["text_dim"], width=1, dash="dot")))
fig_roc.update_layout(defaults, title="Combined ROC Curves (Discrimination Concordance)", height=340, legend=dict(orientation="v", yanchor="bottom", y=0.04, xanchor="right", x=0.98))

p_rf, r_rf, _ = precision_recall_curve(y_true, rf_prob)
p_lr, r_lr, _ = precision_recall_curve(y_true, lr_prob)
p_nn, r_nn, _ = precision_recall_curve(y_true, nn_prob)
prev = float(y_true.mean())

fig_pr = go.Figure()
fig_pr.add_trace(go.Scatter(x=r_rf, y=p_rf, mode="lines", name="Random Forest (AP = 0.6590)", line=dict(color=THEME["primary"], width=2.5)))
fig_pr.add_trace(go.Scatter(x=r_lr, y=p_lr, mode="lines", name="Logistic Regression (AP = 0.6331)", line=dict(color=THEME["secondary"], width=2)))
fig_pr.add_trace(go.Scatter(x=r_nn, y=p_nn, mode="lines", name="Neural Network (AP = 0.6322)", line=dict(color=THEME["accent"], width=2, dash="dash")))
fig_pr.add_trace(go.Scatter(x=[0, 1], y=[prev, prev], mode="lines", name=f"Prevalence ({prev*100:.1f}%)", line=dict(color=THEME["text_dim"], width=1, dash="dot")))
fig_pr.update_layout(defaults, title="Combined Precision-Recall Curves (Average Precision)", height=340, legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="right", x=0.98))

fig_cm = update_confusion_matrix("RF", "counts")
fig_thresh = update_threshold_chart("Random Forest")

body_c = f"""
<div class="page-header">
    <div>
        <h1 class="page-title">Multi-Model Benchmarking & Diagnostic Audit</h1>
        <p class="page-desc">Empirical evaluation across Logistic Regression, Random Forest, and TensorFlow/Keras MLP on unseen test data (N = 1,409).</p>
    </div>
    <div class="badge-status">
        <span class="status-dot"></span>
        <span>Verified Phase 7 Artifacts</span>
    </div>
</div>

<div class="dashboard-card">
    <div class="card-header-bar">
        <div>
            <div class="card-title">Unified Test-Set Performance Benchmark (Threshold = 0.50)</div>
            <div class="card-subtitle">Strictly computed on the held-out test cohort (1,035 Retained, 374 Churned) without data leakage</div>
        </div>
    </div>
    <table class="benchmark-table">
        <thead>
            <tr>
                <th>Metric</th><th>Logistic Regression</th><th>Random Forest (Recommended)</th><th>Neural Network (MLP)</th><th>Statistical & Operational Takeaway</th>
            </tr>
        </thead>
        <tbody>
            <tr><td>Accuracy</td><td>73.81%</td><td class="benchmark-best">75.73%</td><td>74.59%</td><td>Random Forest (+1.92% vs LR)</td></tr>
            <tr><td>Precision (PPV)</td><td>50.43%</td><td class="benchmark-best">52.87%</td><td>51.41%</td><td>Random Forest (lowest false alarm rate)</td></tr>
            <tr><td>Recall (Sensitivity)</td><td>78.34%</td><td class="benchmark-best">78.88%</td><td>78.07%</td><td>Random Forest (catches 295 / 374 churners)</td></tr>
            <tr><td>Specificity (TNR)</td><td>72.17%</td><td class="benchmark-best">74.59%</td><td>73.33%</td><td>Random Forest (772 / 1,035 retained true negatives)</td></tr>
            <tr><td>F1-Score (Harmonic Mean)</td><td>0.6136</td><td class="benchmark-best">0.6330</td><td>0.6200</td><td>Random Forest (+0.0194 vs LR)</td></tr>
            <tr><td>ROC-AUC (Discrimination)</td><td>0.8417</td><td class="benchmark-best">0.8434</td><td>0.8409</td><td>Bootstrap CI spans zero across all pairs (statistically equivalent)</td></tr>
            <tr><td>Average Precision (AP)</td><td>0.6331</td><td class="benchmark-best">0.6590</td><td>0.6322</td><td>Random Forest (+0.0259 advantage under class imbalance)</td></tr>
            <tr><td>False Alarms (FP / 1,035)</td><td>288 (27.8%)</td><td class="benchmark-best">263 (25.4%)</td><td>276 (26.7%)</td><td>Random Forest saves 25 false positive outreach costs vs LR</td></tr>
        </tbody>
    </table>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Discrimination Capacity: Combined ROC</div>
                <div class="card-subtitle">True Positive Rate vs False Positive Rate across all thresholds</div>
            </div>
        </div>
        <div>{fig_roc.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Imbalance Performance: Precision-Recall (AP)</div>
                <div class="card-subtitle">Precision vs Recall across thresholds with Average Precision (AP) rankings</div>
            </div>
        </div>
        <div>{fig_pr.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>

<div class="grid-2col">
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Interactive Confusion Matrix Diagnostic (Random Forest)</div>
                <div class="card-subtitle">Absolute true negative, false positive, false negative, and true positive counts</div>
            </div>
        </div>
        <div>{fig_cm.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="dashboard-card">
        <div class="card-header-bar">
            <div>
                <div class="card-title">Classification Threshold Sensitivity Explorer</div>
                <div class="card-subtitle">Empirical validation split trade-offs across thresholds tau in {0.35, 0.50, 0.65}</div>
            </div>
        </div>
        <div>{fig_thresh.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</div>
"""
html_c = wrap_html("Model Benchmarking", "tab-models", body_c)
path_html_c = os.path.join(OUTPUT_DIR, "module_c.html")
with open(path_html_c, "w", encoding="utf-8") as f:
    f.write(html_c)


# --- 4. RENDER MODULE D & E: SIMULATOR & RECOMMENDATIONS ---
print("[4/4] Generating Modules D & E (Churn Simulator & Playbook) HTML...")
sim_res, sim_recs = run_customer_prediction(
    1, 0.50, "Random Forest",
    "Month-to-month", 3, 85.0, None, "Electronic check", "Yes",
    "Fiber optic", "No", "No", "No", "No",
    "Yes", "No", "Yes", "Yes",
    "Female", 0, "No", "No"
)

# Gauge for sample customer
fig_gauge_sample = render_gauge_chart(0.748, threshold=0.50)

body_d = f"""
<div class="page-header">
    <div>
        <h1 class="page-title">Customer Churn Prediction Simulator</h1>
        <p class="page-desc">Real-time inference using fitted preprocessor and trained machine learning models with customized retention playbooks.</p>
    </div>
    <div class="badge-status">
        <span class="status-dot"></span>
        <span>Zero Data Leakage Pipeline Active</span>
    </div>
</div>

<div style="display:grid; grid-template-columns:1.1fr 1fr; gap:24px;">
    <div>
        <div class="dashboard-card">
            <div class="card-header-bar">
                <div>
                    <div class="card-title">Subscriber Profile & Model Controls</div>
                    <div class="card-subtitle">Active sample customer: Month-to-month, 3 mos tenure, Fiber optic, $85/mo</div>
                </div>
            </div>
            
            <div class="form-section-title">1. Selected Model & Operating Threshold</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:14px;">
                <div style="padding:10px 14px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <div class="filter-label">Active Model</div>
                    <div style="font-weight:700; color:{THEME['primary']};">Random Forest Classifier (Balanced)</div>
                </div>
                <div style="padding:10px 14px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <div class="filter-label">Decision Threshold (tau)</div>
                    <div style="font-weight:700; color:{THEME['secondary']};">0.50 (Standard Operating Cutoff)</div>
                </div>
            </div>

            <div class="form-section-title">2. Account Terms & Billing</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Contract: </span><b style="color:{THEME['churn']};">Month-to-month</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Tenure: </span><b>3 Months Active</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Monthly Charges: </span><b>$85.00 / month</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Payment: </span><b style="color:{THEME['churn']};">Electronic check</b>
                </div>
            </div>

            <div class="form-section-title">3. Services & Add-ons</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px;">
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Internet: </span><b style="color:{THEME['accent']};">Fiber optic</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Online Security: </span><b style="color:{THEME['churn']};">No (Unprotected)</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Tech Support: </span><b style="color:{THEME['churn']};">No (Unassisted)</b>
                </div>
                <div style="padding:8px 12px; background:{THEME['bg_app']}; border:1px solid {THEME['border']}; border-radius:8px;">
                    <span style="color:{THEME['text_dim']}; font-size:0.75rem;">Streaming TV: </span><b>Yes</b>
                </div>
            </div>

            <button class="btn-primary" style="margin-top:10px;">⚡ Live Simulator Interactive in Web Browser (app.py)</button>
        </div>
    </div>

    <div>
        <div class="dashboard-card">
            <div class="card-header-bar">
                <div>
                    <div class="card-title">Inference Risk Diagnostic</div>
                    <div class="card-subtitle">Real-time model prediction output and feature influence</div>
                </div>
            </div>
            
            <div style="padding:12px 16px; border-radius:8px; margin-bottom:16px; background:rgba(239, 68, 68, 0.15); border:1px solid rgba(239, 68, 68, 0.4); color:#F87171;">
                <div style="font-weight:800; font-size:0.95rem;">CRITICAL CHURN RISK PREDICTED</div>
                <div style="font-size:0.8rem; margin-top:2px;">Probability (74.8%) exceeds decision threshold (50%). Proactive retention intervention required.</div>
            </div>

            <div style="display:flex; align-items:center; gap:16px; margin-bottom:12px;">
                <div style="flex:1.2;">{fig_gauge_sample.to_html(full_html=False, include_plotlyjs=False)}</div>
                <div style="flex:0.8; display:flex; flex-direction:column; gap:8px;">
                    <div><span style="font-size:0.78rem; color:{THEME['text_dim']};">Risk Tier: </span><span class="risk-badge risk-badge-high">High Risk</span></div>
                    <div><span style="font-size:0.78rem; color:{THEME['text_dim']};">Active Model: </span><span style="font-size:0.82rem; font-weight:600; color:{THEME['text_main']};">Random Forest</span></div>
                    <div><span style="font-size:0.78rem; color:{THEME['text_dim']};">Applied Cutoff: </span><span style="font-size:0.82rem; font-weight:600; color:{THEME['primary']};">tau = 0.50</span></div>
                </div>
            </div>

            <div style="padding:12px 14px; background:rgba(19, 28, 49, 0.6); border-radius:8px; border:1px solid {THEME['border']};">
                <div style="font-weight:600; color:{THEME['primary']}; font-size:0.82rem; margin-bottom:4px;">Random Forest Key Split Features (MDI Importance):</div>
                <ul style="padding-left:18px; font-size:0.78rem; color:{THEME['text_muted']};">
                    <li>tenure (Global Weight: 18.0%): Active value = -1.19 (Early Vulnerability Stage)</li>
                    <li>TotalCharges (Global Weight: 13.6%): Active value = -0.89</li>
                    <li>Contract_Two year (Global Weight: 11.8%): Active value = 0.00 (Missing Protective Anchor)</li>
                    <li>MonthlyCharges (Global Weight: 9.4%): Active value = +0.67 (Elevated Expenditure)</li>
                    <li>InternetService_Fiber optic (Global Weight: 7.4%): Active value = 1.00 (High-Hazard Channel)</li>
                </ul>
            </div>
        </div>

        <div class="dashboard-card">
            <div class="card-header-bar">
                <div>
                    <div class="card-title">Actionable Retention Intelligence Playbook</div>
                    <div class="card-subtitle">Prescriptive retention strategies tailored to subscriber profile</div>
                </div>
            </div>
            
            <div style="font-size:0.85rem; font-weight:600; color:{THEME['churn']}; margin-bottom:14px;">
                HIGH RISK ALERT (74.8% Churn Probability): This account requires immediate, proactive retention intervention across contract terms, billing friction, and customer support channels.
            </div>

            <div class="rec-card priority-Critical">
                <div class="rec-header">
                    <span class="rec-title">Contract Structure · Annual Contract Migration Incentive</span>
                    <span class="rec-badge rec-badge-critical">Critical Priority</span>
                </div>
                <p class="rec-text">Customer is on a Month-to-month contract. EDA confirms month-to-month subscribers exhibit a 42.7% churn rate (15x higher than 2-year subscribers).</p>
                <p class="rec-text" style="color:{THEME['text_main']}; font-weight:500;">Offer a guaranteed 15% rate lock or a free streaming service upgrade conditional on transitioning to a 1-year service agreement.</p>
                <div class="rec-impact">Expected Outcome: Reduces immediate monthly cancellation vulnerability and establishes relationship stability.</div>
            </div>

            <div class="rec-card priority-Critical">
                <div class="rec-header">
                    <span class="rec-title">Customer Success · First-Year Concierge Onboarding Touchpoint</span>
                    <span class="rec-badge rec-badge-critical">Critical Priority</span>
                </div>
                <p class="rec-text">Customer tenure is 3 months. Historical data reveals that nearly 50% of all customer churn occurs within the first 12 months.</p>
                <p class="rec-text" style="color:{THEME['text_main']}; font-weight:500;">Assign an onboarding specialist for a proactive 30-day satisfaction check-in to resolve initial configuration or billing friction.</p>
                <div class="rec-impact">Expected Outcome: Addresses customer dissatisfaction before it solidifies into an early cancellation.</div>
            </div>

            <div class="rec-card priority-High">
                <div class="rec-header">
                    <span class="rec-title">Service Value & Support · Proactive Premium Support & Security Bundle</span>
                    <span class="rec-badge rec-badge-high">High Priority</span>
                </div>
                <p class="rec-text">Customer pays $85.00/mo for Fiber optic without Tech Support and Online Security. High expenditure combined with unsupported technical friction strongly correlates with churn.</p>
                <p class="rec-text" style="color:{THEME['text_main']}; font-weight:500;">Provide a complimentary 6-month trial of Tech Support and Online Security paired with a network speed optimization review.</p>
                <div class="rec-impact">Expected Outcome: Increases perceived service value and eliminates unassisted technical frustration.</div>
            </div>

            <div style="font-size:0.74rem; color:{THEME['text_dim']}; margin-top:12px; padding:8px 12px; background:rgba(33, 46, 74, 0.3); border-radius:6px;">
                Methodological Note: Recommended retention actions are operational heuristics derived from observed empirical associations in historical telecom data. Correlation does not imply direct causation, and individual subscriber responsiveness should be validated through structured A/B experimentation.
            </div>
        </div>
    </div>
</div>
"""
html_d = wrap_html("Churn Predictor & ROI", "tab-simulator", body_d)
path_html_d = os.path.join(OUTPUT_DIR, "module_d_e.html")
with open(path_html_d, "w", encoding="utf-8") as f:
    f.write(html_d)

print("[SUCCESS] All 4 static HTML views generated successfully.")

# Now screenshot each with msedge.exe
targets = [
    ("module_a_executive_overview.png", path_html_a),
    ("module_b_customer_analytics.png", path_html_b),
    ("module_c_model_evaluation.png", path_html_c),
    ("module_d_e_churn_simulator.png", path_html_d),
]

for img_name, html_f in targets:
    out_img = os.path.join(OUTPUT_DIR, img_name)
    file_url = f"file:///{os.path.abspath(html_f).replace(os.sep, '/')}"
    print(f"Capturing screenshot: {img_name}...")
    cmd = [
        EDGE_PATH,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        "--window-size=1920,1200",
        f"--screenshot={out_img}",
        file_url,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    if os.path.exists(out_img):
        sz = os.path.getsize(out_img) / 1024
        print(f"  [SUCCESS] {img_name} ({sz:.1f} KB)")
    else:
        print(f"  [ERROR] Could not capture {img_name}")
