"""
src/model_comparison.py
-----------------------
Customer Churn Prediction - Phase 7: Comprehensive Model Evaluation & Benchmarking.

Academic Context:
1. Performs unified comparative evaluation across all three trained models:
   - Logistic Regression (Linear baseline with balanced class weights)
   - Random Forest (Bagged decision tree ensemble with balanced class weights)
   - Neural Network (Multi-Layer Perceptron with balanced class weights)
2. Computes Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC, PR-AUC, FPR, FNR,
   confusion matrix components, and train-vs-test generalization gaps.
3. Conducts paired bootstrap hypothesis testing (B = 1,000 resamples) to compute 95%
   confidence intervals on performance differences across identical test samples.
4. Generates publication-grade comparative visualizations:
   - Grouped bar chart comparing key metrics
   - Combined ROC Curves (with chance baseline)
   - Combined Precision-Recall Curves (with no-skill baseline)
   - Side-by-side 1x3 confusion matrices (with counts and row-normalized percentages)
5. Exports comparison artifacts:
   - outputs/metrics/model_comparison.json
   - outputs/metrics/model_comparison.csv
   - outputs/figures/comparison/
"""

import os
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

# Set random seed for bootstrap reproducibility
np.random.seed(42)

# File and directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "comparison")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

LR_METRICS_PATH = os.path.join(METRICS_DIR, "logistic_regression_metrics.json")
RF_METRICS_PATH = os.path.join(METRICS_DIR, "random_forest_metrics.json")
NN_METRICS_PATH = os.path.join(METRICS_DIR, "neural_network_metrics.json")

LR_PRED_PATH = os.path.join(METRICS_DIR, "logistic_regression_predictions.csv")
RF_PRED_PATH = os.path.join(METRICS_DIR, "random_forest_predictions.csv")
NN_PRED_PATH = os.path.join(METRICS_DIR, "neural_network_predictions.csv")

LR_MODEL_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
NN_MODEL_PATH = os.path.join(MODELS_DIR, "neural_network.keras")

OUTPUT_JSON_PATH = os.path.join(METRICS_DIR, "model_comparison.json")
OUTPUT_CSV_PATH = os.path.join(METRICS_DIR, "model_comparison.csv")
VAL_THRESH_PATH = os.path.join(METRICS_DIR, "validation_threshold_analysis.csv")

FIG_PERF_PATH = os.path.join(FIGURES_DIR, "model_performance_comparison.png")
FIG_ROC_PATH = os.path.join(FIGURES_DIR, "combined_roc_curve.png")
FIG_PR_PATH = os.path.join(FIGURES_DIR, "combined_precision_recall_curve.png")
FIG_CM_PATH = os.path.join(FIGURES_DIR, "model_confusion_matrices.png")

# Palette styling
PALETTE = {
    "Logistic Regression": "#1f77b4",  # Blue
    "Random Forest": "#2ca02c",        # Green
    "Neural Network (MLP)": "#9467bd"  # Purple
}


def load_and_verify_predictions() -> tuple:
    """Loads prediction files and verifies identical test instances and ground truth labels."""
    print("[1/5] Loading and verifying model predictions on held-out test set...")
    lr_df = pd.read_csv(LR_PRED_PATH)
    rf_df = pd.read_csv(RF_PRED_PATH)
    nn_df = pd.read_csv(NN_PRED_PATH)

    assert len(lr_df) == len(rf_df) == len(nn_df) == 1409, "Sample size mismatch in prediction files!"
    assert (lr_df["Actual_Churn"] == rf_df["Actual_Churn"]).all(), "Ground truth mismatch between LR and RF!"
    assert (lr_df["Actual_Churn"] == nn_df["Actual_Churn"]).all(), "Ground truth mismatch between LR and NN!"

    y_true = lr_df["Actual_Churn"].values
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    print(f"      Total test records: {len(y_true):,} (Retained: {n_neg:,}, Churned: {n_pos:,})")
    print(f"      Ground truth labels match perfectly across all 3 models.")

    predictions = {
        "Logistic Regression": (lr_df["Predicted_Churn"].values, lr_df["Churn_Probability"].values),
        "Random Forest": (rf_df["Predicted_Churn"].values, rf_df["Churn_Probability"].values),
        "Neural Network (MLP)": (nn_df["Predicted_Churn"].values, nn_df["Churn_Probability"].values)
    }

    # Load training metrics for generalization gap analysis
    train_metrics = {}
    for name, path in [("Logistic Regression", LR_METRICS_PATH),
                       ("Random Forest", RF_METRICS_PATH),
                       ("Neural Network (MLP)", NN_METRICS_PATH)]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                train_metrics[name] = data.get("train_metrics", {})

    return y_true, predictions, train_metrics


def compute_comparative_metrics(y_true: np.ndarray, predictions: dict, train_metrics: dict) -> tuple:
    """Computes comprehensive test metrics, confusion counts, and generalization gaps for all models."""
    print("[2/5] Computing unified comparative metrics...")
    comparison_records = []
    comparison_dict = {}

    for name, (y_pred, y_prob) in predictions.items():
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        spec = tn / (tn + fp)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        fpr = fp / (tn + fp)
        fnr = fn / (fn + tp)

        tr = train_metrics.get(name, {})
        tr_acc = tr.get("accuracy", np.nan)
        tr_f1 = tr.get("f1_score", np.nan)
        tr_auc = tr.get("roc_auc", np.nan)

        record = {
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall (Sensitivity)": round(float(rec), 4),
            "Specificity": round(float(spec), 4),
            "F1-Score": round(float(f1), 4),
            "ROC-AUC": round(float(roc_auc), 4),
            "Average Precision (AP)": round(float(pr_auc), 4),
            "False Positive Rate (FPR)": round(float(fpr), 4),
            "False Negative Rate (FNR)": round(float(fnr), 4),
            "True Negatives (TN)": tn,
            "False Positives (FP)": fp,
            "False Negatives (FN)": fn,
            "True Positives (TP)": tp,
            "Train Accuracy": round(float(tr_acc), 4) if not np.isnan(tr_acc) else None,
            "Accuracy Gen Gap": round(float(tr_acc - acc), 4) if not np.isnan(tr_acc) else None,
            "Train F1": round(float(tr_f1), 4) if not np.isnan(tr_f1) else None,
            "F1 Gen Gap": round(float(tr_f1 - f1), 4) if not np.isnan(tr_f1) else None,
            "Train ROC-AUC": round(float(tr_auc), 4) if not np.isnan(tr_auc) else None,
            "ROC-AUC Gen Gap": round(float(tr_auc - roc_auc), 4) if not np.isnan(tr_auc) else None,
        }
        comparison_records.append(record)
        comparison_dict[name] = record

    comparison_df = pd.DataFrame(comparison_records)
    return comparison_df, comparison_dict


def compute_paired_bootstrap_cis(y_true: np.ndarray, predictions: dict, B: int = 1000) -> dict:
    """Computes paired bootstrap 95% confidence intervals on model metric differences."""
    print(f"[3/5] Running paired bootstrap testing (B = {B:,} resamples on held-out test set)...")
    n = len(y_true)
    models = list(predictions.keys())

    diff_records = {
        "RF - LR": {"accuracy": [], "f1": [], "roc_auc": [], "recall": [], "precision": []},
        "NN - LR": {"accuracy": [], "f1": [], "roc_auc": [], "recall": [], "precision": []},
        "RF - NN": {"accuracy": [], "f1": [], "roc_auc": [], "recall": [], "precision": []}
    }

    lr_pred, lr_prob = predictions["Logistic Regression"]
    rf_pred, rf_prob = predictions["Random Forest"]
    nn_pred, nn_prob = predictions["Neural Network (MLP)"]

    for _ in range(B):
        idx = np.random.randint(0, n, size=n)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue

        # LR
        acc_lr = accuracy_score(y_b, lr_pred[idx])
        f1_lr = f1_score(y_b, lr_pred[idx], zero_division=0)
        auc_lr = roc_auc_score(y_b, lr_prob[idx])
        rec_lr = recall_score(y_b, lr_pred[idx], zero_division=0)
        prec_lr = precision_score(y_b, lr_pred[idx], zero_division=0)

        # RF
        acc_rf = accuracy_score(y_b, rf_pred[idx])
        f1_rf = f1_score(y_b, rf_pred[idx], zero_division=0)
        auc_rf = roc_auc_score(y_b, rf_prob[idx])
        rec_rf = recall_score(y_b, rf_pred[idx], zero_division=0)
        prec_rf = precision_score(y_b, rf_pred[idx], zero_division=0)

        # NN
        acc_nn = accuracy_score(y_b, nn_pred[idx])
        f1_nn = f1_score(y_b, nn_pred[idx], zero_division=0)
        auc_nn = roc_auc_score(y_b, nn_prob[idx])
        rec_nn = recall_score(y_b, nn_pred[idx], zero_division=0)
        prec_nn = precision_score(y_b, nn_pred[idx], zero_division=0)

        # Pairwise differences
        diff_records["RF - LR"]["accuracy"].append(acc_rf - acc_lr)
        diff_records["RF - LR"]["f1"].append(f1_rf - f1_lr)
        diff_records["RF - LR"]["roc_auc"].append(auc_rf - auc_lr)
        diff_records["RF - LR"]["recall"].append(rec_rf - rec_lr)
        diff_records["RF - LR"]["precision"].append(prec_rf - prec_lr)

        diff_records["NN - LR"]["accuracy"].append(acc_nn - acc_lr)
        diff_records["NN - LR"]["f1"].append(f1_nn - f1_lr)
        diff_records["NN - LR"]["roc_auc"].append(auc_nn - auc_lr)
        diff_records["NN - LR"]["recall"].append(rec_nn - rec_lr)
        diff_records["NN - LR"]["precision"].append(prec_nn - prec_lr)

        diff_records["RF - NN"]["accuracy"].append(acc_rf - acc_nn)
        diff_records["RF - NN"]["f1"].append(f1_rf - f1_nn)
        diff_records["RF - NN"]["roc_auc"].append(auc_rf - auc_nn)
        diff_records["RF - NN"]["recall"].append(rec_rf - rec_nn)
        diff_records["RF - NN"]["precision"].append(prec_rf - prec_nn)

    ci_summary = {}
    for pair, metrics in diff_records.items():
        ci_summary[pair] = {}
        for m_name, vals in metrics.items():
            mean_val = float(np.mean(vals))
            low_ci = float(np.percentile(vals, 2.5))
            high_ci = float(np.percentile(vals, 97.5))
            spans_zero = bool(low_ci <= 0.0 <= high_ci)
            ci_summary[pair][m_name] = {
                "mean_difference": round(mean_val, 4),
                "ci_95_low": round(low_ci, 4),
                "ci_95_high": round(high_ci, 4),
                "ci_excludes_zero": not spans_zero,
                "note": "95% CI excludes zero (unadjusted for multiple comparisons)" if not spans_zero else "Inconclusive (95% CI spans zero)"
            }

    print("\n      --- Paired Bootstrap 95% Empirical Confidence Intervals (Percentile Method) ---")
    print("      Note: Shared resampling indices ensure pairing. CIs quantify parameter uncertainty,")
    print("      not formal hypothesis testing p-values. Evaluating multiple metrics inflates family-wise Type I error.")
    for pair, metrics in ci_summary.items():
        print(f"      Pair: {pair}")
        for m_name, res in metrics.items():
            status = "CI EXCLUDES ZERO" if res["ci_excludes_zero"] else "INCONCLUSIVE (spans zero)"
            print(f"        {m_name:<10}: Diff = {res['mean_difference']:+.4f} [{res['ci_95_low']:+.4f}, {res['ci_95_high']:+.4f}] -> {status}")
    return ci_summary


def generate_visualizations(y_true: np.ndarray, predictions: dict, comparison_df: pd.DataFrame):
    """Generates all 4 comparative publication figures."""
    print("[4/6] Generating comparative visualizations...")
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Grouped Bar Chart: Model Performance Comparison
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1-Score", "ROC-AUC", "Average Precision (AP)"]
    x = np.arange(len(metrics_to_plot))
    width = 0.26

    for i, (name, color) in enumerate(PALETTE.items()):
        row = comparison_df[comparison_df["Model"] == name].iloc[0]
        vals = [row[m] for m in metrics_to_plot]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color, edgecolor="black", linewidth=0.7)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=7.8, fontweight="bold", rotation=0)

    ax.set_title("Comprehensive Model Performance Comparison (Unseen Test Set, N = 1,409)", pad=15, fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontweight="bold", fontsize=9.5)
    ax.set_ylabel("Metric Score (0.0 to 1.0)", fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.savefig(FIG_PERF_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved: {FIG_PERF_PATH}")

    # --------------------------------------------------------------------------
    # 2. Combined ROC Curves
    # --------------------------------------------------------------------------
    fig_roc, ax_roc = plt.subplots(figsize=(7.5, 6.5))
    for name, (_, prob) in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        auc_score = roc_auc_score(y_true, prob)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.4f})", color=PALETTE[name], linewidth=2.2)

    ax_roc.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.2, label="Random Chance Baseline (AUC = 0.5000)")
    ax_roc.set_title("Combined Receiver Operating Characteristic (ROC) Curves\n(Unseen Held-Out Test Set, N = 1,409)",
                     pad=15, fontweight="bold", fontsize=12)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.legend(loc="lower right", frameon=True, fontsize=10)
    fig_roc.savefig(FIG_ROC_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_roc)
    print(f"      Saved: {FIG_ROC_PATH}")

    # --------------------------------------------------------------------------
    # 3. Combined Precision-Recall Curves
    # --------------------------------------------------------------------------
    fig_pr, ax_pr = plt.subplots(figsize=(7.5, 6.5))
    no_skill = float(y_true.sum() / len(y_true))  # 374 / 1409 = 0.2654

    for name, (_, prob) in predictions.items():
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, prob)
        pr_auc = average_precision_score(y_true, prob)
        ax_pr.plot(recall_vals, precision_vals, label=f"{name} (PR-AUC = {pr_auc:.4f})", color=PALETTE[name], linewidth=2.2)

    ax_pr.axhline(no_skill, color="black", linestyle="--", linewidth=1.2, label=f"No-Skill Baseline ({no_skill*100:.2f}%)")
    ax_pr.set_title("Combined Precision-Recall (PR) Curves\n(Class Imbalance: 26.54% Positive Churn Rate)",
                    pad=15, fontweight="bold", fontsize=12)
    ax_pr.set_xlabel("Recall (Sensitivity)", fontweight="bold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.legend(loc="upper right", frameon=True, fontsize=10)
    fig_pr.savefig(FIG_PR_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_pr)
    print(f"      Saved: {FIG_PR_PATH}")

    # --------------------------------------------------------------------------
    # 4. Side-by-Side Confusion Matrices (1x3 Grid)
    # --------------------------------------------------------------------------
    fig_cm, axes = plt.subplots(1, 3, figsize=(17, 5))
    total_samples = len(y_true)

    for ax, (name, (pred, _)) in zip(axes, predictions.items()):
        cm = confusion_matrix(y_true, pred)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        row_ret = tn + fp
        row_chu = fn + tp

        # Compute normalized percentages
        norm_tn = (tn / row_ret) * 100
        norm_fp = (fp / row_ret) * 100
        norm_fn = (fn / row_chu) * 100
        norm_tp = (tp / row_chu) * 100

        labels = [
            [f"TN: {tn:,}\n({norm_tn:.1f}% row)\n[{tn/total_samples*100:.1f}% tot]",
             f"FP: {fp:,}\n({norm_fp:.1f}% row)\n[{fp/total_samples*100:.1f}% tot]"],
            [f"FN: {fn:,}\n({norm_fn:.1f}% row)\n[{fn/total_samples*100:.1f}% tot]",
             f"TP: {tp:,}\n({norm_tp:.1f}% row)\n[{tp/total_samples*100:.1f}% tot]"]
        ]

        cm_norm = np.array([[norm_tn, norm_fp], [norm_fn, norm_tp]])
        cmap_name = "Blues" if "Logistic" in name else ("Greens" if "Forest" in name else "Purples")

        sns.heatmap(cm_norm, annot=labels, fmt="", cmap=cmap_name, cbar=False, ax=ax, vmin=0, vmax=100,
                    xticklabels=["Retained (0)", "Churned (1)"],
                    yticklabels=["Retained (0)", "Churned (1)"],
                    annot_kws={"fontsize": 9.5, "fontweight": "bold"})
        ax.set_title(f"{name}\nAcc: {accuracy_score(y_true, pred):.4f} | Recall: {recall_score(y_true, pred):.4f}",
                     fontweight="bold", fontsize=11, pad=10)
        ax.set_xlabel("Predicted Label", fontweight="bold")
        ax.set_ylabel("Actual Label", fontweight="bold")

    fig_cm.suptitle("Comparative Confusion Matrices (Unseen Held-Out Test Set, N = 1,409)", fontsize=13, fontweight="bold", y=1.03)
    fig_cm.savefig(FIG_CM_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_cm)
    print(f"      Saved: {FIG_CM_PATH}")


def evaluate_validation_thresholds() -> pd.DataFrame:
    """
    Evaluates classification thresholds strictly using the validation split
    (last 20% of X_train, N = 1,127), NEVER the held-out test set.
    """
    print("[5/6] Evaluating classification thresholds on validation data...")
    import joblib
    import tensorflow as tf

    X_train_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "X_train.csv"))
    y_train_full = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "y_train.csv"))["Churn"].values

    # Validation slice (20% of training data, matching Phase 6 internal split)
    split_idx = int(len(X_train_df) * 0.8)
    X_val_df = X_train_df.iloc[split_idx:]
    X_val_np = X_val_df.values.astype(np.float32)
    y_val = y_train_full[split_idx:]

    lr = joblib.load(LR_MODEL_PATH)
    rf = joblib.load(RF_MODEL_PATH)
    nn = tf.keras.models.load_model(NN_MODEL_PATH)

    val_probs = {
        "Logistic Regression": lr.predict_proba(X_val_df)[:, 1],
        "Random Forest": rf.predict_proba(X_val_df)[:, 1],
        "Neural Network (MLP)": nn.predict(X_val_np, verbose=0).ravel()
    }

    thresholds = [0.35, 0.50, 0.65]
    records = []

    for name, prob in val_probs.items():
        for tau in thresholds:
            pred = (prob >= tau).astype(int)
            cm = confusion_matrix(y_val, pred)
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]
            acc = accuracy_score(y_val, pred)
            prec = precision_score(y_val, pred, zero_division=0)
            rec = recall_score(y_val, pred, zero_division=0)
            spec = tn / (tn + fp)
            f1 = f1_score(y_val, pred, zero_division=0)
            records.append({
                "Model": name,
                "Threshold": tau,
                "Accuracy": round(float(acc), 4),
                "Precision": round(float(prec), 4),
                "Recall": round(float(rec), 4),
                "Specificity": round(float(spec), 4),
                "F1_Score": round(float(f1), 4),
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                "Validation_N": len(y_val)
            })

    val_df = pd.DataFrame(records)
    val_df.to_csv(VAL_THRESH_PATH, index=False)
    print(f"      Saved validation threshold analysis -> {VAL_THRESH_PATH}")
    return val_df


def save_comparison_deliverables(comparison_df: pd.DataFrame, comparison_dict: dict, bootstrap_cis: dict, val_threshold_df: pd.DataFrame):
    """Saves comparison metrics CSV and JSON."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[6/6] Saved unified metrics table -> {OUTPUT_CSV_PATH}")

    final_json = {
        "evaluation_dataset": {
            "source": "Unseen Held-Out Test Set",
            "total_samples": 1409,
            "retained_count": 1035,
            "churned_count": 374,
            "churn_rate_pct": 26.54
        },
        "metric_definitions_note": {
            "Average_Precision_AP": "Calculated via sklearn average_precision_score (weighted mean of precisions at each threshold), distinct from trapezoidal rule PR-AUC which can overestimate area.",
            "ROC_AUC": "Measures ranking concordance (discrimination), distinct from probability calibration."
        },
        "model_metrics": comparison_dict,
        "paired_bootstrap_empirical_cis": {
            "iterations": 1000,
            "random_seed": 42,
            "resampling_method": "Paired percentile bootstrap on identical held-out test records",
            "confidence_level": 0.95,
            "results": bootstrap_cis
        },
        "validation_threshold_analysis": val_threshold_df.to_dict(orient="records")
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=2)
    print(f"      Saved comparative JSON summary -> {OUTPUT_JSON_PATH}")


def run_model_comparison_pipeline():
    """Main execution function for Phase 7."""
    print("=" * 70)
    print("   CUSTOMER CHURN PREDICTION - PHASE 7: BENCHMARKING & COMPARISON")
    print("=" * 70)

    # 1. Load and verify
    y_true, predictions, train_metrics = load_and_verify_predictions()

    # 2. Compute metrics
    comparison_df, comparison_dict = compute_comparative_metrics(y_true, predictions, train_metrics)

    # 3. Bootstrap CIs
    bootstrap_cis = compute_paired_bootstrap_cis(y_true, predictions, B=1000)

    # 4. Visualizations
    generate_visualizations(y_true, predictions, comparison_df)

    # 5. Validation threshold analysis
    val_threshold_df = evaluate_validation_thresholds()

    # 6. Save deliverables
    save_comparison_deliverables(comparison_df, comparison_dict, bootstrap_cis, val_threshold_df)

    print("\n" + "=" * 70)
    print("   PHASE 7: COMPARATIVE EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_model_comparison_pipeline()

