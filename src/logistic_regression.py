"""
src/logistic_regression.py
--------------------------
Customer Churn Prediction - Phase 4: Logistic Regression Baseline Model.

Academic Context:
1. Implements a regularized Logistic Regression classifier as an interpretable baseline.
2. Incorporates balanced class weighting (w0=0.6806, w1=1.8843) to counter minority churn imbalance.
3. Fits model strictly on training data (X_train, y_train) with reproducible random_state.
4. Evaluates unseen generalization performance on X_test (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Extracts and visualizes standardized feature coefficients and log-odds associations.
"""

import os
import sys
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Define directories and file paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures", "models")

TRAIN_X_PATH = os.path.join(PROCESSED_DATA_DIR, "X_train.csv")
TEST_X_PATH = os.path.join(PROCESSED_DATA_DIR, "X_test.csv")
TRAIN_Y_PATH = os.path.join(PROCESSED_DATA_DIR, "y_train.csv")
TEST_Y_PATH = os.path.join(PROCESSED_DATA_DIR, "y_test.csv")

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "logistic_regression.joblib")
METRICS_OUTPUT_PATH = os.path.join(METRICS_DIR, "logistic_regression_metrics.json")
PREDICTIONS_OUTPUT_PATH = os.path.join(METRICS_DIR, "logistic_regression_predictions.csv")
COEFFICIENTS_OUTPUT_PATH = os.path.join(METRICS_DIR, "logistic_regression_coefficients.csv")
CM_FIGURE_PATH = os.path.join(FIGURES_DIR, "logistic_regression_confusion_matrix.png")
CM_NORM_FIGURE_PATH = os.path.join(FIGURES_DIR, "logistic_regression_confusion_matrix_normalized.png")
COEF_FIGURE_PATH = os.path.join(FIGURES_DIR, "logistic_regression_coefficients.png")


def load_datasets() -> tuple:
    """Loads processed training and testing datasets and verifies feature alignment."""
    print("[1/5] Loading processed datasets...")
    X_train = pd.read_csv(TRAIN_X_PATH)
    X_test = pd.read_csv(TEST_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH)["Churn"].values
    y_test = pd.read_csv(TEST_Y_PATH)["Churn"].values

    assert list(X_train.columns) == list(X_test.columns), "Feature columns mismatch between train and test!"
    print(f"      X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"      X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    print(f"      Features count: {len(X_train.columns)} (Identical ordering confirmed)")
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray) -> LogisticRegression:
    """
    Trains LogisticRegression with balanced class weights and reproducible random state.
    
    Academic Rationale:
    - class_weight='balanced': Inversely weights instances inversely proportional to class frequencies.
      Minority class 1 (Churn) receives a penalty weight ~1.884 vs ~0.681 for majority class 0.
      This shifts the decision boundary to prioritize churn recall.
    - C=1.0: L2 ridge penalty controls parameter magnitude and stabilizes multicollinear features.
    - solver='lbfgs': Robust, second-order quasi-Newton optimizer.
    """
    print("[2/5] Training Logistic Regression model with balanced class weighting...")
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    print("      Model training complete.")
    print(f"      Optimization iterations taken: {model.n_iter_[0]}")
    print(f"      Fitted Intercept (bias): {model.intercept_[0]:.4f}")
    return model


def evaluate_model(model: LogisticRegression, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray) -> dict:
    """Evaluates model performance on both training and test sets and returns detailed metrics."""
    print("[3/5] Evaluating model performance on unseen test data...")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Confusion matrix breakdown (Test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()

    # Test metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, pos_label=1)
    test_rec = recall_score(y_test, y_pred_test, pos_label=1)
    test_f1 = f1_score(y_test, y_pred_test, pos_label=1)
    test_auc = roc_auc_score(y_test, y_prob_test)

    # Train metrics for overfitting check
    train_acc = accuracy_score(y_train, y_pred_train)
    train_prec = precision_score(y_train, y_pred_train, pos_label=1)
    train_rec = recall_score(y_train, y_pred_train, pos_label=1)
    train_f1 = f1_score(y_train, y_pred_train, pos_label=1)
    train_auc = roc_auc_score(y_train, y_prob_train)

    report_dict = classification_report(y_test, y_pred_test, output_dict=True)

    print("\n" + "=" * 55)
    print("        LOGISTIC REGRESSION TEST EVALUATION")
    print("=" * 55)
    print(f"  Accuracy:         {test_acc:.4f} (Train: {train_acc:.4f})")
    print(f"  Precision (Churn): {test_prec:.4f} (Train: {train_prec:.4f})")
    print(f"  Recall (Churn):    {test_rec:.4f} (Train: {train_rec:.4f})")
    print(f"  F1-Score (Churn):  {test_f1:.4f} (Train: {train_f1:.4f})")
    print(f"  ROC-AUC Score:    {test_auc:.4f} (Train: {train_auc:.4f})")
    print("-" * 55)
    print("  Confusion Matrix:")
    print(f"    True Negatives (TN):  {tn:5d}  |  False Positives (FP): {fp:5d}")
    print(f"    False Negatives (FN): {fn:5d}  |  True Positives (TP):  {tp:5d}")
    print("=" * 55 + "\n")

    metrics = {
        "model_name": "Logistic Regression (Balanced)",
        "hyperparameters": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_auc), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "train_metrics": {
            "accuracy": round(float(train_acc), 4),
            "precision": round(float(train_prec), 4),
            "recall": round(float(train_rec), 4),
            "f1_score": round(float(train_f1), 4),
            "roc_auc": round(float(train_auc), 4),
        },
        "classification_report": report_dict
    }

    # Save predictions dataframe
    predictions_df = pd.DataFrame({
        "Actual_Churn": y_test,
        "Predicted_Churn": y_pred_test,
        "Churn_Probability": np.round(y_prob_test, 4)
    })

    return metrics, predictions_df, cm


def extract_and_save_coefficients(model: LogisticRegression, feature_names: list) -> pd.DataFrame:
    """Extracts feature coefficients, calculates odds ratios, and ranks them by absolute impact."""
    print("[4/5] Extracting and interpreting model coefficients...")
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "Odds_Ratio": odds_ratios,
        "Abs_Impact": np.abs(coefs)
    }).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)

    print("\nTop 5 Positive Coefficients (Highest Churn Risk Association):")
    for _, row in coef_df.head(5).iterrows():
        print(f"  + {row['Feature']:<35} : Coef = {row['Coefficient']:+.4f} (Odds Ratio = {row['Odds_Ratio']:.3f})")

    print("\nTop 5 Negative Coefficients (Highest Retention Association):")
    for _, row in coef_df.tail(5).iloc[::-1].iterrows():
        print(f"  - {row['Feature']:<35} : Coef = {row['Coefficient']:+.4f} (Odds Ratio = {row['Odds_Ratio']:.3f})")

    return coef_df


def generate_visualizations(cm: np.ndarray, coef_df: pd.DataFrame = None):
    """
    Generates confusion matrix heatmaps (overall & row-normalized) and coefficient bar plot.
    
    Visualization A - Overall test-set percentages:
      (cell count / total_samples) * 100
      TN = 747 (53.02%), FP = 288 (20.44%), FN = 81 (5.75%), TP = 293 (20.79%)
      
    Visualization B - Row-normalized percentages:
      Normalized within each actual class row:
      Retained (0): TN = 747/1035 = 72.17%, FP = 288/1035 = 27.83%
      Churned (1):  FN = 81/374  = 21.66%, TP = 293/374  = 78.34%
    """
    print("[5/5] Generating visualizations...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    total_samples = int(cm.sum())
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # --------------------------------------------------------------------------
    # 1. Visualization A: Overall test-set percentages ((count / 1409) * 100)
    # --------------------------------------------------------------------------
    fig_a, ax_a = plt.subplots(figsize=(6.5, 5.2))
    pct_tn = (tn / total_samples) * 100
    pct_fp = (fp / total_samples) * 100
    pct_fn = (fn / total_samples) * 100
    pct_tp = (tp / total_samples) * 100

    labels_a = [
        [f"True Negative (TN)\n{tn:,}\n({pct_tn:.2f}%)", f"False Positive (FP)\n{fp:,}\n({pct_fp:.2f}%)"],
        [f"False Negative (FN)\n{fn:,}\n({pct_fn:.2f}%)", f"True Positive (TP)\n{tp:,}\n({pct_tp:.2f}%)"]
    ]
    sns.heatmap(cm, annot=labels_a, fmt="", cmap="Blues", cbar=True, ax=ax_a,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_a.set_title(f"Logistic Regression Confusion Matrix\n(Overall % of Total Test Set, N = {total_samples:,})", pad=15, fontweight="bold")
    ax_a.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontweight="bold")
    ax_a.set_ylabel("Actual Label (0 = Retained, 1 = Churned)", fontweight="bold")
    fig_a.savefig(CM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_a)
    print(f"      Saved overall confusion matrix -> {CM_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 2. Visualization B: Row-normalized percentages ((count / row_total) * 100)
    # --------------------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(6.5, 5.2))
    row_retained = tn + fp  # Actual Retained total (1035)
    row_churned = fn + tp   # Actual Churned total (374)
    norm_tn = (tn / row_retained) * 100
    norm_fp = (fp / row_retained) * 100
    norm_fn = (fn / row_churned) * 100
    norm_tp = (tp / row_churned) * 100

    labels_b = [
        [f"True Negative (TN)\n{tn:,}\n({norm_tn:.2f}%)", f"False Positive (FP)\n{fp:,}\n({norm_fp:.2f}%)"],
        [f"False Negative (FN)\n{fn:,}\n({norm_fn:.2f}%)", f"True Positive (TP)\n{tp:,}\n({norm_tp:.2f}%)"]
    ]
    cm_norm_pct = np.array([
        [norm_tn, norm_fp],
        [norm_fn, norm_tp]
    ])
    sns.heatmap(cm_norm_pct, annot=labels_b, fmt="", cmap="Blues", cbar=True, ax=ax_b, vmin=0, vmax=100,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_b.set_title(f"Logistic Regression Confusion Matrix\n(Row-Normalized: % within Actual Class, N = {total_samples:,})", pad=15, fontweight="bold")
    ax_b.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontweight="bold")
    ax_b.set_ylabel("Actual Label (0 = Retained, 1 = Churned)", fontweight="bold")
    fig_b.savefig(CM_NORM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_b)
    print(f"      Saved row-normalized confusion matrix -> {CM_NORM_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 3. Coefficients Bar Plot (if coef_df provided)
    # --------------------------------------------------------------------------
    if coef_df is not None:
        fig_c, ax_c = plt.subplots(figsize=(10, 11))
        sorted_df = coef_df.sort_values(by="Coefficient", ascending=True)
        colors = ["#2ca02c" if c < 0 else "#d62728" for c in sorted_df["Coefficient"]]

        bars = ax_c.barh(sorted_df["Feature"], sorted_df["Coefficient"], color=colors, edgecolor="black", linewidth=0.6, height=0.65)
        ax_c.axvline(0, color="black", linestyle="-", linewidth=1.0)
        ax_c.set_title("Logistic Regression Standardized Feature Coefficients (Log-Odds Impact)\nGreen = Protective (Promotes Retention), Red = Risk Factor (Promotes Churn)",
                       pad=15, fontweight="bold", fontsize=11)
        ax_c.set_xlabel("Model Coefficient (Beta Weight)", fontweight="bold")
        ax_c.set_ylabel("Feature Name", fontweight="bold")
        ax_c.set_xlim(sorted_df["Coefficient"].min() * 1.25, sorted_df["Coefficient"].max() * 1.25)

        for bar in bars:
            w = bar.get_width()
            offset = 0.03 if w >= 0 else -0.03
            ha = "left" if w >= 0 else "right"
            ax_c.annotate(f"{w:+.2f}", xy=(w + offset, bar.get_y() + bar.get_height() / 2),
                          va="center", ha=ha, fontsize=8.5, fontweight="bold")

        fig_c.savefig(COEF_FIGURE_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig_c)
        print(f"      Saved coefficient plot -> {COEF_FIGURE_PATH}")


def save_all_artifacts(model: LogisticRegression, metrics: dict, predictions_df: pd.DataFrame, coef_df: pd.DataFrame):
    """Saves serialized model, metrics JSON, predictions CSV, and coefficients CSV."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"      Saved trained model -> {MODEL_OUTPUT_PATH}")

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Saved evaluation metrics -> {METRICS_OUTPUT_PATH}")

    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"      Saved predictions -> {PREDICTIONS_OUTPUT_PATH}")

    coef_df.to_csv(COEFFICIENTS_OUTPUT_PATH, index=False)
    print(f"      Saved coefficients table -> {COEFFICIENTS_OUTPUT_PATH}")


def run_logistic_regression_pipeline():
    """Main execution function for Phase 4."""
    print("=" * 60)
    print("   CUSTOMER CHURN PREDICTION - PHASE 4: LOGISTIC REGRESSION")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_datasets()

    # 2. Train model
    model = train_logistic_regression(X_train, y_train)

    # 3. Evaluate model
    metrics, predictions_df, cm = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 4. Coefficients interpretation
    coef_df = extract_and_save_coefficients(model, list(X_train.columns))

    # 5. Visualizations
    generate_visualizations(cm, coef_df)

    # 6. Save artifacts
    save_all_artifacts(model, metrics, predictions_df, coef_df)

    print("\n" + "=" * 60)
    print("   PHASE 4: LOGISTIC REGRESSION COMPLETED SUCCESSFULLY")
    print("=" * 60)


def run_visualizations_only():
    """Regenerates only confusion matrix visualizations from existing predictions without retraining."""
    print("=" * 60)
    print("   CONFUSION MATRIX VISUALIZATION UPDATE (NO RETRAINING)")
    print("=" * 60)
    pred_df = pd.read_csv(PREDICTIONS_OUTPUT_PATH)
    cm = confusion_matrix(pred_df["Actual_Churn"], pred_df["Predicted_Churn"])
    print(f"Loaded existing predictions from: {PREDICTIONS_OUTPUT_PATH}")
    print(f"Verified Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]} (Total: {cm.sum()})")
    assert cm[0,0] == 747 and cm[0,1] == 288 and cm[1,0] == 81 and cm[1,1] == 293
    generate_visualizations(cm, coef_df=None)
    print("=" * 60)
    print("   VISUALIZATION UPDATE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--vis-only":
        run_visualizations_only()
    else:
        run_logistic_regression_pipeline()
