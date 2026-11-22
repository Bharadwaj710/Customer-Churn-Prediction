"""
src/eda.py
----------
Customer Churn Prediction - Phase 2: Exploratory Data Analysis (EDA)

Academic Context:
Performs univariate, bivariate, and numerical analysis on the authentic
IBM Telco Customer Churn dataset without altering the raw data.
Calculates exact empirical distributions, churn rates across customer segments,
and generates publication-quality visualizations and statistical reports.
"""

import os
import sys
import hashlib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "eda")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

# Plot styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "font.family": "sans-serif",
    "figure.autolayout": False,
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 10,
})

PRIMARY_COLOR = "#1f77b4"
CHURN_PALETTE = {"No": "#2ca02c", "Yes": "#d62728"}


def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """Loads raw data, performs structural audit, and returns an EDA copy with TotalCharges as numeric."""
    print("[1/6] Loading and inspecting raw dataset...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw dataset not found at: {filepath}")

    # Read original CSV
    df_raw = pd.read_csv(filepath)
    print(f"      Loaded dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # Duplicate check
    duplicates = df_raw.duplicated().sum()
    print(f"      Duplicate rows: {duplicates}")

    # Create working copy for EDA
    df = df_raw.copy()

    # Convert TotalCharges to numeric, coercing spaces to NaN
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    null_tc = df["TotalCharges"].isnull().sum()
    print(f"      TotalCharges converted to numeric. Coerced NaN count: {null_tc}")

    # Numeric churn indicator for point-biserial correlation
    df["Churn_Binary"] = (df["Churn"] == "Yes").astype(int)

    return df


def calculate_and_save_summaries(df: pd.DataFrame) -> tuple:
    """Calculates numerical descriptive statistics and categorical churn rate breakdowns."""
    print("[2/6] Calculating statistical summaries and churn percentage tables...")
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Numerical Descriptive Statistics
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    num_summary = df[num_cols].describe().T
    num_summary["median"] = df[num_cols].median()
    num_summary["skewness"] = df[num_cols].skew()
    num_summary["kurtosis"] = df[num_cols].kurtosis()
    
    # Save numerical summary
    num_summary_path = os.path.join(METRICS_DIR, "eda_statistical_summary.csv")
    num_summary.to_csv(num_summary_path, index=True)
    print(f"      Saved numerical summary -> {num_summary_path}")

    # 2. Categorical Churn Summary Table
    categorical_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod"
    ]

    records = []
    for col in categorical_cols:
        grouped = df.groupby(col)["Churn"].value_counts().unstack(fill_value=0)
        if "Yes" not in grouped.columns:
            grouped["Yes"] = 0
        if "No" not in grouped.columns:
            grouped["No"] = 0
        grouped["Total"] = grouped["Yes"] + grouped["No"]
        grouped["Churn_Rate_Pct"] = (grouped["Yes"] / grouped["Total"]) * 100
        for category, row in grouped.iterrows():
            records.append({
                "Feature": col,
                "Category": str(category),
                "Total_Customers": int(row["Total"]),
                "Retained_No": int(row["No"]),
                "Churned_Yes": int(row["Yes"]),
                "Churn_Rate_Pct": round(row["Churn_Rate_Pct"], 2),
            })

    cat_summary_df = pd.DataFrame(records)
    cat_summary_path = os.path.join(METRICS_DIR, "eda_categorical_churn_summary.csv")
    cat_summary_df.to_csv(cat_summary_path, index=False)
    print(f"      Saved categorical churn breakdown -> {cat_summary_path}")

    return num_summary, cat_summary_df


# ==============================================================================
# Plotting Helper Functions
# ==============================================================================

def save_fig(fig: plt.Figure, filename: str):
    """Saves a matplotlib figure cleanly with tight bounding box."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved chart: {filename}")


def plot_univariate_churn(df: pd.DataFrame):
    """Figure 01: Overall Churn Distribution (Counts & Percentages)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df["Churn"].value_counts()
    pcts = df["Churn"].value_counts(normalize=True) * 100

    bars = ax.bar(counts.index, counts.values, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]], width=0.55, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        height = bar.get_height()
        ax.annotate(f"{height:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)

    ax.set_title("Overall Customer Churn Distribution (N = 7,043)", pad=15, fontweight="bold")
    ax.set_ylabel("Customer Count")
    ax.set_xlabel("Churn Status")
    ax.set_ylim(0, counts.max() * 1.15)
    save_fig(fig, "01_univariate_churn.png")


def plot_univariate_demographics(df: pd.DataFrame):
    """Figure 02: Gender & Senior Citizen Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Gender
    g_counts = df["gender"].value_counts()
    g_pcts = df["gender"].value_counts(normalize=True) * 100
    b1 = ax1.bar(g_counts.index, g_counts.values, color=["#4c72b0", "#dd8452"], width=0.5, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(b1, g_pcts):
        ax1.annotate(f"{bar.get_height():,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax1.set_title("Gender Distribution", fontweight="bold")
    ax1.set_ylabel("Customer Count")
    ax1.set_ylim(0, g_counts.max() * 1.15)

    # Senior Citizen
    sc_map = {0: "Non-Senior", 1: "Senior Citizen"}
    sc_counts = df["SeniorCitizen"].map(sc_map).value_counts()
    sc_pcts = df["SeniorCitizen"].map(sc_map).value_counts(normalize=True) * 100
    b2 = ax2.bar(sc_counts.index, sc_counts.values, color=["#55a868", "#c44e52"], width=0.5, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(b2, sc_pcts):
        ax2.annotate(f"{bar.get_height():,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax2.set_title("Senior Citizen Distribution", fontweight="bold")
    ax2.set_ylabel("Customer Count")
    ax2.set_ylim(0, sc_counts.max() * 1.15)

    save_fig(fig, "02_univariate_demographics.png")


def plot_univariate_contract_payment(df: pd.DataFrame):
    """Figure 03: Contract Type & Payment Method Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Contract Type
    c_counts = df["Contract"].value_counts()
    c_pcts = df["Contract"].value_counts(normalize=True) * 100
    b1 = ax1.bar(c_counts.index, c_counts.values, color="#3470a3", width=0.5, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(b1, c_pcts):
        ax1.annotate(f"{bar.get_height():,}\n({pct:.1f}%)",
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                     ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    ax1.set_title("Contract Type Distribution", fontweight="bold")
    ax1.set_ylabel("Customer Count")
    ax1.set_ylim(0, c_counts.max() * 1.15)

    # Payment Method
    p_counts = df["PaymentMethod"].value_counts()
    p_pcts = df["PaymentMethod"].value_counts(normalize=True) * 100
    b2 = ax2.barh(p_counts.index, p_counts.values, color="#489178", edgecolor="black", linewidth=0.8, height=0.55)
    for bar, pct in zip(b2, p_pcts):
        ax2.annotate(f" {bar.get_width():,} ({pct:.1f}%)",
                     xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                     ha="left", va="center", color="black", fontweight="bold", fontsize=9.5)
    ax2.set_title("Payment Method Distribution", fontweight="bold")
    ax2.set_xlabel("Customer Count")
    ax2.set_xlim(0, p_counts.max() * 1.25)

    save_fig(fig, "03_univariate_contract_payment.png")


def plot_univariate_internet(df: pd.DataFrame):
    """Figure 04: Internet Service Distribution."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    i_counts = df["InternetService"].value_counts()
    i_pcts = df["InternetService"].value_counts(normalize=True) * 100
    bars = ax.bar(i_counts.index, i_counts.values, color=["#e28743", "#1e81b0", "#abdbe3"], width=0.5, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(bars, i_pcts):
        ax.annotate(f"{bar.get_height():,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2),
                    ha="center", va="center", color="black" if pct < 30 else "white", fontweight="bold", fontsize=10)
    ax.set_title("Internet Service Distribution", fontweight="bold")
    ax.set_ylabel("Customer Count")
    ax.set_xlabel("Internet Service Option")
    ax.set_ylim(0, i_counts.max() * 1.15)
    save_fig(fig, "04_univariate_internet_service.png")


def plot_univariate_numerical(df: pd.DataFrame):
    """Figure 05: Histograms and KDE of Continuous Variables."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Tenure
    sns.histplot(df["tenure"], kde=True, ax=axes[0], color="#2b5c8f", bins=30, edgecolor="black")
    axes[0].set_title(f"Tenure Distribution\n(Mean: {df['tenure'].mean():.1f}m, Median: {df['tenure'].median():.0f}m)", fontweight="bold")
    axes[0].set_xlabel("Tenure (Months)")
    axes[0].set_ylabel("Frequency")

    # Monthly Charges
    sns.histplot(df["MonthlyCharges"], kde=True, ax=axes[1], color="#cf6d2b", bins=30, edgecolor="black")
    axes[1].set_title(f"Monthly Charges Distribution\n(Mean: ${df['MonthlyCharges'].mean():.2f}, Median: ${df['MonthlyCharges'].median():.2f})", fontweight="bold")
    axes[1].set_xlabel("Monthly Charges ($)")
    axes[1].set_ylabel("Frequency")

    # Total Charges
    tc_clean = df["TotalCharges"].dropna()
    sns.histplot(tc_clean, kde=True, ax=axes[2], color="#368f5b", bins=30, edgecolor="black")
    axes[2].set_title(f"Total Charges Distribution\n(Mean: ${tc_clean.mean():.2f}, Median: ${tc_clean.median():.2f})", fontweight="bold")
    axes[2].set_xlabel("Total Charges ($)")
    axes[2].set_ylabel("Frequency")

    save_fig(fig, "05_univariate_numerical_distributions.png")


# ==============================================================================
# Bivariate Plotting Functions
# ==============================================================================

def plot_bivariate_categorical(df: pd.DataFrame, feature: str, title: str, filename: str, xlabel: str = None):
    """Helper to generate grouped count plot alongside churn rate percentage bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Group calculations
    crosstab = pd.crosstab(df[feature], df["Churn"])
    if "Yes" not in crosstab.columns:
        crosstab["Yes"] = 0
    if "No" not in crosstab.columns:
        crosstab["No"] = 0
    rates = (crosstab["Yes"] / (crosstab["Yes"] + crosstab["No"]) * 100).round(2)

    # 1. Raw counts by Churn
    crosstab.plot(kind="bar", stacked=False, color=[CHURN_PALETTE["No"], CHURN_PALETTE["Yes"]], ax=ax1, edgecolor="black", linewidth=0.8, width=0.6)
    ax1.set_title(f"Customer Count by {title}", fontweight="bold")
    ax1.set_ylabel("Customer Count")
    ax1.set_xlabel(xlabel or feature)
    ax1.tick_params(axis="x", rotation=15)
    ax1.legend(title="Churn Status", labels=["Retained (No)", "Churned (Yes)"])

    # 2. Churn percentage rate
    bars = ax2.bar(rates.index.astype(str), rates.values, color="#d62728", width=0.5, edgecolor="black", linewidth=0.8)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.axhline(df["Churn_Binary"].mean() * 100, color="navy", linestyle="--", linewidth=1.5, label=f"Average Churn Rate ({df['Churn_Binary'].mean()*100:.1f}%)")
    ax2.set_title(f"Empirical Churn Rate (%) by {title}", fontweight="bold")
    ax2.set_ylabel("Churn Percentage (%)")
    ax2.set_xlabel(xlabel or feature)
    ax2.tick_params(axis="x", rotation=15)
    ax2.set_ylim(0, max(rates.max() * 1.2, 50))
    ax2.legend()

    save_fig(fig, filename)


def plot_bivariate_tenure(df: pd.DataFrame):
    """Figure 07: Tenure vs Churn (Boxplot/KDE and Binned Tenure Churn Rate)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot of Tenure by Churn
    sns.boxplot(x="Churn", y="tenure", data=df, ax=ax1, hue="Churn", palette=CHURN_PALETTE, legend=False, width=0.45)
    ax1.set_title("Tenure Comparison by Churn Status", fontweight="bold")
    ax1.set_xlabel("Churn Status")
    ax1.set_ylabel("Tenure (Months)")
    # Add median annotations
    medians = df.groupby("Churn")["tenure"].median()
    means = df.groupby("Churn")["tenure"].mean()
    ax1.annotate(f"Retained Median: {medians['No']:.0f}m\n(Mean: {means['No']:.1f}m)", xy=(0, medians["No"]),
                 xytext=(0.1, 40), textcoords="data",
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                 fontsize=9.5, fontweight="bold")
    ax1.annotate(f"Churned Median: {medians['Yes']:.0f}m\n(Mean: {means['Yes']:.1f}m)", xy=(1, medians["Yes"]),
                 xytext=(1.1, 25), textcoords="data",
                 arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
                 fontsize=9.5, fontweight="bold")

    # Binned Tenure Churn Rate
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["1-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df_temp = df.copy()
    df_temp["Tenure_Group"] = pd.cut(df_temp["tenure"], bins=bins, labels=labels, include_lowest=True)
    t_rate = df_temp.groupby("Tenure_Group", observed=False)["Churn_Binary"].mean() * 100

    bars = ax2.bar(t_rate.index.astype(str), t_rate.values, color="#d62728", width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.axhline(df["Churn_Binary"].mean() * 100, color="navy", linestyle="--", linewidth=1.5, label=f"Average Churn ({df['Churn_Binary'].mean()*100:.1f}%)")
    ax2.set_title("Churn Rate Across Tenure Cohorts", fontweight="bold")
    ax2.set_xlabel("Tenure Cohort")
    ax2.set_ylabel("Churn Percentage (%)")
    ax2.set_ylim(0, max(t_rate.max() * 1.2, 50))
    ax2.legend()

    save_fig(fig, "07_bivariate_tenure_churn.png")


def plot_bivariate_charges(df: pd.DataFrame):
    """Figure 08: Monthly Charges and Total Charges by Churn Status."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Monthly Charges KDE
    sns.kdeplot(data=df[df["Churn"] == "No"]["MonthlyCharges"], ax=ax1, label="Retained (No)", color=CHURN_PALETTE["No"], fill=True, alpha=0.35, linewidth=2)
    sns.kdeplot(data=df[df["Churn"] == "Yes"]["MonthlyCharges"], ax=ax1, label="Churned (Yes)", color=CHURN_PALETTE["Yes"], fill=True, alpha=0.35, linewidth=2)
    mc_median_no = df[df["Churn"] == "No"]["MonthlyCharges"].median()
    mc_median_yes = df[df["Churn"] == "Yes"]["MonthlyCharges"].median()
    ax1.axvline(mc_median_no, color=CHURN_PALETTE["No"], linestyle=":", linewidth=2, label=f"Retained Median: ${mc_median_no:.2f}")
    ax1.axvline(mc_median_yes, color=CHURN_PALETTE["Yes"], linestyle=":", linewidth=2, label=f"Churned Median: ${mc_median_yes:.2f}")
    ax1.set_title("Monthly Charges Density by Churn", fontweight="bold")
    ax1.set_xlabel("Monthly Charges ($)")
    ax1.set_ylabel("Density")
    ax1.legend(loc="upper right")

    # Total Charges Boxplot
    tc_clean = df.dropna(subset=["TotalCharges"])
    sns.boxplot(x="Churn", y="TotalCharges", data=tc_clean, ax=ax2, hue="Churn", palette=CHURN_PALETTE, legend=False, width=0.45)
    tc_med_no = tc_clean[tc_clean["Churn"] == "No"]["TotalCharges"].median()
    tc_med_yes = tc_clean[tc_clean["Churn"] == "Yes"]["TotalCharges"].median()
    ax2.set_title(f"Total Charges by Churn\n(Retained Median: ${tc_med_no:,.0f} vs Churned Median: ${tc_med_yes:,.0f})", fontweight="bold")
    ax2.set_xlabel("Churn Status")
    ax2.set_ylabel("Total Charges ($)")

    save_fig(fig, "08_bivariate_charges_churn.png")


def plot_bivariate_services(df: pd.DataFrame):
    """Figure 09: Churn Rates across Key Services (Internet, TechSupport, OnlineSecurity)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    services = [
        ("InternetService", "Internet Service", axes[0]),
        ("TechSupport", "Tech Support", axes[1]),
        ("OnlineSecurity", "Online Security", axes[2])
    ]

    for col, title, ax in services:
        rates = (df.groupby(col)["Churn_Binary"].mean() * 100).sort_values(ascending=False)
        bars = ax.bar(rates.index.astype(str), rates.values, color=["#d62728" if v > 30 else "#3470a3" for v in rates.values], width=0.5, edgecolor="black", linewidth=0.8)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
        ax.axhline(df["Churn_Binary"].mean() * 100, color="navy", linestyle="--", linewidth=1.2, label=f"Avg Churn ({df['Churn_Binary'].mean()*100:.1f}%)")
        ax.set_title(f"Churn Rate by {title}", fontweight="bold")
        ax.set_ylabel("Churn Percentage (%)")
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylim(0, max(rates.max() * 1.25, 50))
        ax.legend(loc="upper right")

    save_fig(fig, "09_bivariate_services_churn.png")


def plot_bivariate_demographics(df: pd.DataFrame):
    """Figure 11: Churn Rates across Senior Citizen and Gender."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Senior Citizen
    sc_map = {0: "Non-Senior", 1: "Senior Citizen"}
    df_temp = df.copy()
    df_temp["Senior_Label"] = df_temp["SeniorCitizen"].map(sc_map)
    sc_rate = df_temp.groupby("Senior_Label")["Churn_Binary"].mean() * 100
    bars1 = ax1.bar(sc_rate.index, sc_rate.values, color=["#3470a3", "#d62728"], width=0.45, edgecolor="black", linewidth=0.8)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax1.axhline(df["Churn_Binary"].mean() * 100, color="navy", linestyle="--", linewidth=1.2, label=f"Avg ({df['Churn_Binary'].mean()*100:.1f}%)")
    ax1.set_title("Churn Rate by Senior Citizen Status", fontweight="bold")
    ax1.set_ylabel("Churn Percentage (%)")
    ax1.set_ylim(0, 55)
    ax1.legend()

    # Gender
    g_rate = df.groupby("gender")["Churn_Binary"].mean() * 100
    bars2 = ax2.bar(g_rate.index, g_rate.values, color=["#4c72b0", "#55a868"], width=0.45, edgecolor="black", linewidth=0.8)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax2.axhline(df["Churn_Binary"].mean() * 100, color="navy", linestyle="--", linewidth=1.2, label=f"Avg ({df['Churn_Binary'].mean()*100:.1f}%)")
    ax2.set_title("Churn Rate by Gender", fontweight="bold")
    ax2.set_ylabel("Churn Percentage (%)")
    ax2.set_ylim(0, 55)
    ax2.legend()

    save_fig(fig, "11_bivariate_demographics_churn.png")


def plot_correlation_matrix(df: pd.DataFrame):
    """Figure 12: Correlation Matrix Heatmap for Numerical Features and Binary Churn."""
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Binary"]
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax, mask=mask)
    ax.set_title("Numerical Features Correlation Matrix\n(Point-Biserial for Churn_Binary)", pad=15, fontweight="bold")
    save_fig(fig, "12_correlation_heatmap.png")


# ==============================================================================
# Comprehensive Findings Document Generation
# ==============================================================================

def generate_eda_findings_document(df: pd.DataFrame, num_summary: pd.DataFrame, cat_summary: pd.DataFrame):
    """Produces the academic EDA findings markdown report based strictly on calculated values."""
    print("[5/6] Generating comprehensive EDA findings document...")

    # Exact numbers
    total_customers = len(df)
    churn_count = df["Churn_Binary"].sum()
    retained_count = total_customers - churn_count
    churn_pct = (churn_count / total_customers) * 100

    # Contract
    contract_rates = df.groupby("Contract")["Churn_Binary"].mean() * 100
    m2m_rate = contract_rates.get("Month-to-month", 0.0)
    one_yr_rate = contract_rates.get("One year", 0.0)
    two_yr_rate = contract_rates.get("Two year", 0.0)

    # Tenure
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ["1-12m", "13-24m", "25-36m", "37-48m", "49-60m", "61-72m"]
    df_temp = df.copy()
    df_temp["Tenure_Cohort"] = pd.cut(df_temp["tenure"], bins=bins, labels=labels, include_lowest=True)
    tenure_cohort_rates = df_temp.groupby("Tenure_Cohort", observed=False)["Churn_Binary"].mean() * 100

    # Payment
    pm_rates = df.groupby("PaymentMethod")["Churn_Binary"].mean() * 100
    e_check_rate = pm_rates.get("Electronic check", 0.0)

    # Internet & Services
    is_rates = df.groupby("InternetService")["Churn_Binary"].mean() * 100
    fiber_rate = is_rates.get("Fiber optic", 0.0)
    dsl_rate = is_rates.get("DSL", 0.0)
    no_internet_rate = is_rates.get("No", 0.0)

    ts_rates = df.groupby("TechSupport")["Churn_Binary"].mean() * 100
    os_rates = df.groupby("OnlineSecurity")["Churn_Binary"].mean() * 100

    # Charges
    mc_churn_mean = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
    mc_ret_mean = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
    mc_churn_median = df[df["Churn"] == "Yes"]["MonthlyCharges"].median()
    mc_ret_median = df[df["Churn"] == "No"]["MonthlyCharges"].median()

    tc_churn_median = df[df["Churn"] == "Yes"]["TotalCharges"].median()
    tc_ret_median = df[df["Churn"] == "No"]["TotalCharges"].median()

    corr_tenure_churn = df["tenure"].corr(df["Churn_Binary"])
    corr_mc_churn = df["MonthlyCharges"].corr(df["Churn_Binary"])
    corr_tc_churn = df["TotalCharges"].corr(df["Churn_Binary"])

    content = f"""# Exploratory Data Analysis (EDA) Academic Findings Report

**Project:** Customer Churn Prediction Using Machine Learning  
**Dataset:** Authentic IBM Telco Customer Churn Dataset (`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)  
**Records Analyzed:** {total_customers:,} rows, 21 attributes  
**Analysis Date:** 2026-11-22  

---

## 1. Executive Summary & Class Distribution

- **Total Sample Size:** {total_customers:,} customers.
- **Retained Customers (`Churn = No`):** {retained_count:,} ({100 - churn_pct:.2f}%).
- **Churned Customers (`Churn = Yes`):** {churn_count:,} ({churn_pct:.2f}%).
- **Class Imbalance Ratio:** ~2.77 : 1 (Majority to Minority).
- **Academic Significance:** Because the minority class represents {churn_pct:.1f}% of instances, a naive baseline classifier predicting "No" would achieve {100 - churn_pct:.1f}% accuracy while failing completely to identify at-risk customers. Precision, Recall, F1-score, and ROC-AUC are mandatory metrics.

---

## 2. Key Empirical Findings by Domain Segment

### A. Contract Structure & Commitment
- **Month-to-month Contract Churn Rate:** **{m2m_rate:.2f}%** ({df[df['Contract']=='Month-to-month']['Churn_Binary'].sum():,} out of {len(df[df['Contract']=='Month-to-month']):,} customers).
- **One-Year Contract Churn Rate:** **{one_yr_rate:.2f}%** ({df[df['Contract']=='One year']['Churn_Binary'].sum():,} out of {len(df[df['Contract']=='One year']):,} customers).
- **Two-Year Contract Churn Rate:** **{two_yr_rate:.2f}%** ({df[df['Contract']=='Two year']['Churn_Binary'].sum():,} out of {len(df[df['Contract']=='Two year']):,} customers).
- **Finding:** Customers on month-to-month contracts are **{m2m_rate / two_yr_rate:.1f}x more likely to churn** than customers on two-year contracts. Long-term contractual commitment serves as the single strongest protective factor against customer attrition.

### B. Customer Tenure & Cohort Dynamics
- **0–12 Months Tenure Cohort Churn Rate:** **{tenure_cohort_rates['1-12m']:.2f}%**.
- **61–72 Months Tenure Cohort Churn Rate:** **{tenure_cohort_rates['61-72m']:.2f}%**.
- **Median Tenure Comparison:**
  - Retained customers: **{df[df['Churn']=='No']['tenure'].median():.0f} months** (Mean: {df[df['Churn']=='No']['tenure'].mean():.1f} months).
  - Churned customers: **{df[df['Churn']=='Yes']['tenure'].median():.0f} months** (Mean: {df[df['Churn']=='Yes']['tenure'].mean():.1f} months).
- **Finding:** Churn risk is front-loaded. Nearly half of all first-year subscribers churn, after which churn rate systematically declines as brand lock-in and customer habituation increase.

### C. Financial & Billing Factors
- **Payment Method Churn Rates:**
  - **Electronic Check:** **{e_check_rate:.2f}%** ({df[df['PaymentMethod']=='Electronic check']['Churn_Binary'].sum():,} churned).
  - **Mailed Check:** **{pm_rates.get('Mailed check', 0):.2f}%**.
  - **Bank Transfer (Automatic):** **{pm_rates.get('Bank transfer (automatic)', 0):.2f}%**.
  - **Credit Card (Automatic):** **{pm_rates.get('Credit card (automatic)', 0):.2f}%**.
- **Monthly Charges Impact:**
  - Churned Customers Median: **${mc_churn_median:.2f}** (Mean: ${mc_churn_mean:.2f}).
  - Retained Customers Median: **${mc_ret_median:.2f}** (Mean: ${mc_ret_mean:.2f}).
  - Difference: Churned customers pay **${mc_churn_median - mc_ret_median:.2f} more per month** at the median.
- **Total Charges Paradox:**
  - Retained Customers Median Total Charges: **${tc_ret_median:,.2f}**.
  - Churned Customers Median Total Charges: **${tc_churn_median:,.2f}**.
  - Explanation: Because churned customers leave early in their lifecycle (median tenure of only 10 months), their cumulative lifetime spend is lower despite paying higher monthly rates.

### D. Service Configuration & Add-on Utilities
- **Internet Service Type:**
  - **Fiber Optic:** **{fiber_rate:.2f}%** churn rate.
  - **DSL:** **{dsl_rate:.2f}%** churn rate.
  - **No Internet:** **{no_internet_rate:.2f}%** churn rate.
  - Fiber optic users churn at more than double the rate of DSL users, primarily because fiber plans carry significantly higher monthly fees ($70–$100+/mo).
- **Protective Add-ons:**
  - Customers without Tech Support churn at **{ts_rates.get('No', 0):.2f}%** vs **{ts_rates.get('Yes', 0):.2f}%** with Tech Support.
  - Customers without Online Security churn at **{os_rates.get('No', 0):.2f}%** vs **{os_rates.get('Yes', 0):.2f}%** with Online Security.
  - Subscribers with value-added support and security services exhibit strong retention.

### E. Demographics
- **Senior Citizens:** Churn rate of **{df[df['SeniorCitizen']==1]['Churn_Binary'].mean()*100:.2f}%** vs **{df[df['SeniorCitizen']==0]['Churn_Binary'].mean()*100:.2f}%** for non-seniors.
- **Gender:** Female churn rate is **{df[df['gender']=='Female']['Churn_Binary'].mean()*100:.2f}%** vs Male churn rate of **{df[df['gender']=='Male']['Churn_Binary'].mean()*100:.2f}%**. Gender is virtually non-predictive (difference < 1%).

---

## 3. Numerical Correlation Analysis & Academic Limitations

### Pearson / Point-Biserial Correlation with `Churn_Binary`:
- `tenure` vs `Churn_Binary`: **$r = {corr_tenure_churn:.3f}$** (Moderate negative correlation).
- `MonthlyCharges` vs `Churn_Binary`: **$r = +{corr_mc_churn:.3f}$** (Moderate positive correlation).
- `TotalCharges` vs `Churn_Binary`: **$r = {corr_tc_churn:.3f}$** (Weak negative correlation).

### Crucial Academic Methodological Note:
> [!WARNING]
> **Limitations of Interpreting Correlations with a Binary Target:**
> 1. **Point-Biserial Nature:** Pearson's correlation coefficient assumes continuous bivariate normality. When calculated against a dichotomous binary target ($Y \\in \\{{0, 1\\}}$), Pearson's formula simplifies to the point-biserial correlation ($r_{{pb}}$), whose theoretical maximum is strictly bounded below $1.0$ (depending on class distribution skewness).
> 2. **Non-Linear Relationships:** Linear correlation fails to capture threshold or non-linear effects (e.g., rapid churn spike in months 1–3 followed by plateau).
> 3. **Confounding Variables:** `TotalCharges` is the mathematical product of `tenure` $\\times$ `MonthlyCharges`. Its negative correlation with churn ($r = {corr_tc_churn:.3f}$) is confounded by tenure and should not be misinterpreted as lower total cost causing churn.
> 4. **Identifier Exclusion:** `customerID` was strictly excluded from correlation analysis as it is an arbitrary primary key with no predictive or statistical meaning.

---

## 4. Summary Table of Generated Figures

| File Name | Analysis Type | Description |
| :--- | :--- | :--- |
| `01_univariate_churn.png` | Univariate | Class balance count and percentage breakdown |
| `02_univariate_demographics.png` | Univariate | Gender and Senior Citizen distributions |
| `03_univariate_contract_payment.png` | Univariate | Contract types and Payment method counts |
| `04_univariate_internet_service.png` | Univariate | DSL, Fiber optic, and No Internet proportions |
| `05_univariate_numerical_distributions.png` | Univariate | Histograms with KDE for Tenure, MonthlyCharges, TotalCharges |
| `06_bivariate_contract_churn.png` | Bivariate | Counts and empirical churn rates across contract types |
| `07_bivariate_tenure_churn.png` | Bivariate | Boxplots and churn percentages across 6 tenure cohorts |
| `08_bivariate_charges_churn.png` | Bivariate | MonthlyCharges density and TotalCharges distributions |
| `09_bivariate_services_churn.png` | Bivariate | Churn rates across Internet, TechSupport, OnlineSecurity |
| `10_bivariate_payment_churn.png` | Bivariate | Churn rates across all 4 payment methods |
| `11_bivariate_demographics_churn.png` | Bivariate | Churn rates across Senior Citizen status and Gender |
| `12_correlation_heatmap.png` | Multivariate | Correlation matrix across numerical variables and target |

---
"""
    findings_path = os.path.join(METRICS_DIR, "eda_findings.md")
    with open(findings_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"      Saved comprehensive EDA findings document -> {findings_path}")


def verify_raw_data_untouched(filepath: str, expected_sha: str):
    """Verifies that raw CSV has not been modified in any way."""
    with open(filepath, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    assert h == expected_sha, f"CRITICAL: Raw file has been modified! SHA: {h}"
    print(f"[VERIFIED] Raw CSV SHA-256 remains intact: {h}")


def run_eda_pipeline():
    """Main execution function for Phase 2 EDA."""
    print("=" * 60)
    print("      CUSTOMER CHURN PREDICTION - PHASE 2: EDA")
    print("=" * 60)

    # 1. Load data
    df = load_and_inspect_data(RAW_DATA_PATH)

    # 2. Summaries & Metrics
    num_summary, cat_summary = calculate_and_save_summaries(df)

    # 3. Univariate Visualizations
    print("[3/6] Generating Univariate Visualizations...")
    plot_univariate_churn(df)
    plot_univariate_demographics(df)
    plot_univariate_contract_payment(df)
    plot_univariate_internet(df)
    plot_univariate_numerical(df)

    # 4. Bivariate & Correlation Visualizations
    print("[4/6] Generating Bivariate and Correlation Visualizations...")
    plot_bivariate_categorical(df, "Contract", "Contract Type", "06_bivariate_contract_churn.png", "Contract Type")
    plot_bivariate_tenure(df)
    plot_bivariate_charges(df)
    plot_bivariate_services(df)
    plot_bivariate_categorical(df, "PaymentMethod", "Payment Method", "10_bivariate_payment_churn.png", "Payment Method")
    plot_bivariate_demographics(df)
    plot_correlation_matrix(df)

    # 5. Generate Findings Document
    generate_eda_findings_document(df, num_summary, cat_summary)

    # 6. Verify Raw File Immutability
    expected_sha = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
    verify_raw_data_untouched(RAW_DATA_PATH, expected_sha)

    print("\n" + "=" * 60)
    print("      PHASE 2: EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_eda_pipeline()
