"""
src/random_forest.py
--------------------
Customer Churn Prediction - Phase 5: Random Forest Classifier.

Academic Context:
1. Implements a nonlinear ensemble classifier using Bootstrap Aggregation (Bagging) and
   random feature subspacing (m = sqrt(p)).
2. Incorporates balanced class weighting (w0=0.6806, w1=1.8843) to account for class imbalance.
3. Uses regularized tree constraints (max_depth=10, min_samples_leaf=4) to prevent overfitting.
4. Evaluates unseen generalization performance on X_test (Accuracy, Precision, Recall, F1, ROC-AUC).
5. Extracts and plots Gini-impurity feature importances (Mean Decrease in Impurity - MDI).
6. Compares predictive behavior with the baseline Logistic Regression model.
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

from sklearn.ensemble import RandomForestClassifier
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
LOGISTIC_METRICS_PATH = os.path.join(METRICS_DIR, "logistic_regression_metrics.json")

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "random_forest.joblib")
METRICS_OUTPUT_PATH = os.path.join(METRICS_DIR, "random_forest_metrics.json")
PREDICTIONS_OUTPUT_PATH = os.path.join(METRICS_DIR, "random_forest_predictions.csv")
IMPORTANCE_OUTPUT_PATH = os.path.join(METRICS_DIR, "random_forest_feature_importance.csv")
CM_FIGURE_PATH = os.path.join(FIGURES_DIR, "random_forest_confusion_matrix.png")
CM_NORM_FIGURE_PATH = os.path.join(FIGURES_DIR, "random_forest_confusion_matrix_normalized.png")
IMPORTANCE_FIGURE_PATH = os.path.join(FIGURES_DIR, "random_forest_feature_importance.png")


def load_datasets() -> tuple:
    """Loads processed training and testing datasets and verifies feature alignment."""
    print("[1/5] Loading processed datasets...")
    X_train = pd.read_csv(TRAIN_X_PATH)
    X_test = pd.read_csv(TEST_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH)["Churn"].values
    y_test = pd.read_csv(TEST_Y_PATH)["Churn"].values

    assert list(X_train.columns) == list(X_test.columns), "Feature mismatch between train and test!"
    print(f"      X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"      X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    print(f"      Features count: {len(X_train.columns)} (Identical ordering confirmed)")
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier with balanced class weighting and regularized depth.
    
    Academic Rationale:
    - Bootstrap Aggregation (Bagging): Builds 100 decision trees on independent bootstrap
      subsamples drawn with replacement from X_train, reducing model variance without increasing bias.
    - Feature Randomness: At each candidate split, considers sqrt(30) ~ 5 random features.
      This decorrelates individual trees, preventing dominant features (e.g. tenure) from dictating every tree.
    - Regularized Depth: max_depth=10, min_samples_leaf=4, min_samples_split=10 prevent unconstrained leaf
      memorization, maintaining generalization.
    - class_weight='balanced': Re-weights the Gini criterion inversely to class frequencies.
    """
    print("[2/5] Training Random Forest ensemble with balanced class weighting...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("      Random Forest training complete.")
    print(f"      Number of estimators (trees): {len(model.estimators_)}")
    print(f"      Max depth constraint: {model.max_depth}")
    return model


def evaluate_model(model: RandomForestClassifier, X_train: pd.DataFrame, y_train: np.ndarray,
                   X_test: pd.DataFrame, y_test: np.ndarray) -> tuple:
    """Evaluates model performance on both training and test sets and returns detailed metrics."""
    print("[3/5] Evaluating model performance on unseen test data...")

    # Predictions
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]

    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Confusion matrix breakdown (Test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

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
    print("         RANDOM FOREST TEST EVALUATION")
    print("=" * 55)
    print(f"  Accuracy:          {test_acc:.4f} (Train: {train_acc:.4f})")
    print(f"  Precision (Churn): {test_prec:.4f} (Train: {train_prec:.4f})")
    print(f"  Recall (Churn):     {test_rec:.4f} (Train: {train_rec:.4f})")
    print(f"  F1-Score (Churn):   {test_f1:.4f} (Train: {train_f1:.4f})")
    print(f"  ROC-AUC Score:     {test_auc:.4f} (Train: {train_auc:.4f})")
    print("-" * 55)
    print("  Confusion Matrix:")
    print(f"    True Negatives (TN):  {tn:5d}  |  False Positives (FP): {fp:5d}")
    print(f"    False Negatives (FN): {fn:5d}  |  True Positives (TP):  {tp:5d}")
    print("=" * 55 + "\n")

    metrics = {
        "model_name": "Random Forest (Balanced)",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "random_state": 42
        },
        "test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "precision": round(float(test_prec), 4),
            "recall": round(float(test_rec), 4),
            "f1_score": round(float(test_f1), 4),
            "roc_auc": round(float(test_auc), 4),
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
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

    predictions_df = pd.DataFrame({
        "Actual_Churn": y_test,
        "Predicted_Churn": y_pred_test,
        "Churn_Probability": np.round(y_prob_test, 4)
    })

    return metrics, predictions_df, cm


def extract_and_save_feature_importance(model: RandomForestClassifier, feature_names: list) -> pd.DataFrame:
    """Extracts Gini-impurity feature importances (MDI) and ranks them."""
    print("[4/5] Extracting Random Forest feature importances...")
    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
        "Importance_Pct": importances * 100
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Most Important Features (Gini Impurity Reduction):")
    for rank, row in importance_df.head(10).iterrows():
        print(f"  {rank+1:2d}. {row['Feature']:<35} : {row['Importance']:.4f} ({row['Importance_Pct']:.2f}%)")

    return importance_df


def generate_visualizations(cm: np.ndarray, importance_df: pd.DataFrame):
    """Generates overall confusion matrix, row-normalized confusion matrix, and feature importance chart."""
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
    sns.heatmap(cm, annot=labels_a, fmt="", cmap="Greens", cbar=True, ax=ax_a,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_a.set_title(f"Random Forest Confusion Matrix\n(Overall % of Total Test Set, N = {total_samples:,})", pad=15, fontweight="bold")
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
    sns.heatmap(cm_norm_pct, annot=labels_b, fmt="", cmap="Greens", cbar=True, ax=ax_b, vmin=0, vmax=100,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_b.set_title(f"Random Forest Confusion Matrix\n(Row-Normalized: % within Actual Class, N = {total_samples:,})", pad=15, fontweight="bold")
    ax_b.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontweight="bold")
    ax_b.set_ylabel("Actual Label (0 = Retained, 1 = Churned)", fontweight="bold")
    fig_b.savefig(CM_NORM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_b)
    print(f"      Saved row-normalized confusion matrix -> {CM_NORM_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 3. Top 15 Feature Importance Chart
    # --------------------------------------------------------------------------
    fig_c, ax_c = plt.subplots(figsize=(10, 7.5))
    top15_df = importance_df.head(15).sort_values(by="Importance", ascending=True)

    bars = ax_c.barh(top15_df["Feature"], top15_df["Importance_Pct"], color="#2b5c8f", edgecolor="black", linewidth=0.6, height=0.65)
    ax_c.set_title("Top 15 Most Important Features - Random Forest Classifier\n(Mean Decrease in Gini Impurity - MDI)",
                   pad=15, fontweight="bold", fontsize=11)
    ax_c.set_xlabel("Relative Importance (%)", fontweight="bold")
    ax_c.set_ylabel("Feature Name", fontweight="bold")
    ax_c.set_xlim(0, top15_df["Importance_Pct"].max() * 1.18)

    for bar in bars:
        w = bar.get_width()
        ax_c.annotate(f"{w:.2f}%", xy=(w + 0.3, bar.get_y() + bar.get_height() / 2),
                      va="center", ha="left", fontsize=9, fontweight="bold")

    fig_c.savefig(IMPORTANCE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_c)
    print(f"      Saved feature importance chart -> {IMPORTANCE_FIGURE_PATH}")


def save_all_artifacts(model: RandomForestClassifier, metrics: dict, predictions_df: pd.DataFrame, importance_df: pd.DataFrame):
    """Saves serialized model, metrics JSON, predictions CSV, and feature importance CSV."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"      Saved trained model -> {MODEL_OUTPUT_PATH}")

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Saved evaluation metrics -> {METRICS_OUTPUT_PATH}")

    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"      Saved predictions -> {PREDICTIONS_OUTPUT_PATH}")

    importance_df.to_csv(IMPORTANCE_OUTPUT_PATH, index=False)
    print(f"      Saved feature importance table -> {IMPORTANCE_OUTPUT_PATH}")


def compare_with_logistic_regression(rf_metrics: dict):
    """Compares actual Logistic Regression and Random Forest evaluation metrics side-by-side."""
    print("\n" + "=" * 65)
    print("      PRELIMINARY MODEL COMPARISON (TEST SET EVALUATION)")
    print("=" * 65)

    if not os.path.exists(LOGISTIC_METRICS_PATH):
        print("Logistic Regression metrics file not found for comparison.")
        return

    with open(LOGISTIC_METRICS_PATH, "r", encoding="utf-8") as f:
        lr_metrics = json.load(f)

    lr_t = lr_metrics["test_metrics"]
    rf_t = rf_metrics["test_metrics"]

    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC-AUC Score",
                   "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)"],
        "Logistic Regression": [
            f"{lr_t['accuracy']:.4f}",
            f"{lr_t['precision']:.4f}",
            f"{lr_t['recall']:.4f}",
            f"{lr_t['f1_score']:.4f}",
            f"{lr_t['roc_auc']:.4f}",
            f"{lr_t['true_negatives']:,}",
            f"{lr_t['false_positives']:,}",
            f"{lr_t['false_negatives']:,}",
            f"{lr_t['true_positives']:,}"
        ],
        "Random Forest": [
            f"{rf_t['accuracy']:.4f}",
            f"{rf_t['precision']:.4f}",
            f"{rf_t['recall']:.4f}",
            f"{rf_t['f1_score']:.4f}",
            f"{rf_t['roc_auc']:.4f}",
            f"{rf_t['true_negatives']:,}",
            f"{rf_t['false_positives']:,}",
            f"{rf_t['false_negatives']:,}",
            f"{rf_t['true_positives']:,}"
        ]
    })

    print(comparison.to_string(index=False))
    print("=" * 65)


def run_random_forest_pipeline():
    """Main execution function for Phase 5."""
    print("=" * 60)
    print("   CUSTOMER CHURN PREDICTION - PHASE 5: RANDOM FOREST")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_datasets()

    # 2. Train model
    model = train_random_forest(X_train, y_train)

    # 3. Evaluate model
    metrics, predictions_df, cm = evaluate_model(model, X_train, y_train, X_test, y_test)

    # 4. Feature importance
    importance_df = extract_and_save_feature_importance(model, list(X_train.columns))

    # 5. Visualizations
    generate_visualizations(cm, importance_df)

    # 6. Save artifacts
    save_all_artifacts(model, metrics, predictions_df, importance_df)

    # 7. Comparison with Logistic Regression
    compare_with_logistic_regression(metrics)

    print("\n" + "=" * 60)
    print("   PHASE 5: RANDOM FOREST COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_random_forest_pipeline()

