"""
src/neural_network.py
---------------------
Customer Churn Prediction - Phase 6: Deep Learning Neural Network (MLP).

Academic Context:
1. Implements a Multi-Layer Perceptron (MLP) binary classifier using TensorFlow/Keras.
2. Architecture: 30 Inputs -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid).
3. Utilizes Adam optimizer (lr=0.001), binary crossentropy loss, and balanced class weights.
4. Employs EarlyStopping with restore_best_weights=True on a 20% validation split from training data.
5. Strictly avoids data leakage: X_test is used solely for final unseen evaluation.
6. Generates loss/accuracy training curves, confusion matrices, and ROC curves.
7. Produces a preliminary 3-model comparative benchmark across all evaluated models.
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

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

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
RF_METRICS_PATH = os.path.join(METRICS_DIR, "random_forest_metrics.json")

MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "neural_network.keras")
METRICS_OUTPUT_PATH = os.path.join(METRICS_DIR, "neural_network_metrics.json")
PREDICTIONS_OUTPUT_PATH = os.path.join(METRICS_DIR, "neural_network_predictions.csv")

HISTORY_FIGURE_PATH = os.path.join(FIGURES_DIR, "neural_network_training_history.png")
CM_FIGURE_PATH = os.path.join(FIGURES_DIR, "neural_network_confusion_matrix.png")
CM_NORM_FIGURE_PATH = os.path.join(FIGURES_DIR, "neural_network_confusion_matrix_normalized.png")
ROC_FIGURE_PATH = os.path.join(FIGURES_DIR, "neural_network_roc_curve.png")


def load_datasets() -> tuple:
    """Loads processed training and testing datasets and verifies feature alignment."""
    print("[1/5] Loading processed datasets...")
    X_train_df = pd.read_csv(TRAIN_X_PATH)
    X_test_df = pd.read_csv(TEST_X_PATH)
    y_train = pd.read_csv(TRAIN_Y_PATH)["Churn"].values
    y_test = pd.read_csv(TEST_Y_PATH)["Churn"].values

    assert list(X_train_df.columns) == list(X_test_df.columns), "Feature mismatch between train and test!"
    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    print(f"      X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"      X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")
    print(f"      Features count: {X_train.shape[1]} (Identical ordering confirmed)")
    return X_train, X_test, y_train, y_test, list(X_train_df.columns)


def build_neural_network(input_dim: int = 30) -> tf.keras.Model:
    """
    Constructs the Multi-Layer Perceptron (MLP) architecture.
    
    Academic Architecture:
    - Input Layer: 30 normalized & encoded features.
    - Hidden Layer 1: Dense(64, ReLU) captures first-order nonlinear interactions across tabular inputs.
    - Dropout(0.3): Randomly zeros 30% of unit activations during training to discourage co-adaptation.
    - Hidden Layer 2: Dense(32, ReLU) condenses representation to 32 latent features.
    - Dropout(0.2): Additional regularization layer.
    - Output Layer: Dense(1, Sigmoid) outputs predicted probability p(y = 1 | x).
    """
    print("[2/5] Building Neural Network architecture...")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="Input_Features"),
        tf.keras.layers.Dense(64, activation="relu", name="Dense_Hidden_1"),
        tf.keras.layers.Dropout(0.3, name="Dropout_1"),
        tf.keras.layers.Dense(32, activation="relu", name="Dense_Hidden_2"),
        tf.keras.layers.Dropout(0.2, name="Dropout_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="Output_Sigmoid")
    ], name="Customer_Churn_MLP")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    print(f"      Total trainable parameters: {model.count_params():,}")
    model.summary(print_fn=lambda x: print(f"      {x}"))
    return model


def train_neural_network(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Trains the neural network with balanced class weights and early stopping.
    Validation split (20%) is taken exclusively from training data.
    """
    print("\n[3/5] Training Neural Network with EarlyStopping and class weighting...")
    
    # Calculate balanced class weights
    n_samples = len(y_train)
    n_class0 = (y_train == 0).sum()
    n_class1 = (y_train == 1).sum()
    w0 = n_samples / (2.0 * n_class0)
    w1 = n_samples / (2.0 * n_class1)
    class_weight = {0: float(w0), 1: float(w1)}
    print(f"      Applied Class Weights: Class 0 = {w0:.4f}, Class 1 = {w1:.4f}")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=1
    )

    best_epoch = early_stopping.best_epoch + 1 if hasattr(early_stopping, "best_epoch") else len(history.history["loss"])
    total_epochs = len(history.history["loss"])
    print(f"      Training stopped at epoch: {total_epochs}")
    print(f"      Best weights restored from epoch: {best_epoch} (Val Loss: {min(history.history['val_loss']):.4f})")
    return model, history, best_epoch


def evaluate_model(model: tf.keras.Model, history, best_epoch: int,
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """Evaluates the trained neural network on unseen test data."""
    print("\n[4/5] Evaluating Neural Network on unseen test data...")

    # Predictions
    y_prob_train = model.predict(X_train, verbose=0).ravel()
    y_pred_train = (y_prob_train >= 0.5).astype(int)

    y_prob_test = model.predict(X_test, verbose=0).ravel()
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    # Confusion matrix (Test)
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
    print("        NEURAL NETWORK TEST EVALUATION")
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
        "model_name": "Neural Network (MLP Balanced)",
        "architecture": {
            "layers": [
                {"type": "Dense", "units": 64, "activation": "relu"},
                {"type": "Dropout", "rate": 0.3},
                {"type": "Dense", "units": 32, "activation": "relu"},
                {"type": "Dropout", "rate": 0.2},
                {"type": "Dense", "units": 1, "activation": "sigmoid"}
            ],
            "total_parameters": model.count_params(),
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "binary_crossentropy",
            "epochs_trained": len(history.history["loss"]),
            "best_epoch": int(best_epoch)
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

    return metrics, predictions_df, cm, y_prob_test


def generate_visualizations(history, best_epoch: int, cm: np.ndarray, y_test: np.ndarray, y_prob_test: np.ndarray):
    """Generates training history curves, confusion matrices, and ROC curve."""
    print("[5/5] Generating visualizations...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    total_samples = int(cm.sum())
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # --------------------------------------------------------------------------
    # 1. Training vs Validation History (Accuracy & Loss)
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history.history["loss"]) + 1)

    # Accuracy
    ax1.plot(epochs_range, history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    ax1.plot(epochs_range, history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2, linestyle="--")
    ax1.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, label=f"Best Epoch ({best_epoch})")
    ax1.set_title("Neural Network Accuracy vs. Epochs", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss
    ax2.plot(epochs_range, history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    ax2.axvline(best_epoch, color="green", linestyle=":", linewidth=1.5, label=f"Best Epoch ({best_epoch})")
    ax2.set_title("Neural Network Binary Crossentropy Loss vs. Epochs", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    fig.suptitle("Neural Network (MLP) Learning Dynamics (Early Stopping with Patience=15)", fontsize=13, fontweight="bold")
    fig.savefig(HISTORY_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Saved training history plot -> {HISTORY_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 2. Confusion Matrix - Overall Percentages ((count / 1409) * 100)
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
    sns.heatmap(cm, annot=labels_a, fmt="", cmap="Purples", cbar=True, ax=ax_a,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_a.set_title(f"Neural Network Confusion Matrix\n(Overall % of Total Test Set, N = {total_samples:,})", pad=15, fontweight="bold")
    ax_a.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontweight="bold")
    ax_a.set_ylabel("Actual Label (0 = Retained, 1 = Churned)", fontweight="bold")
    fig_a.savefig(CM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_a)
    print(f"      Saved overall confusion matrix -> {CM_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 3. Confusion Matrix - Row-Normalized Percentages ((count / row_total) * 100)
    # --------------------------------------------------------------------------
    fig_b, ax_b = plt.subplots(figsize=(6.5, 5.2))
    row_retained = tn + fp  # 1035
    row_churned = fn + tp   # 374
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
    sns.heatmap(cm_norm_pct, annot=labels_b, fmt="", cmap="Purples", cbar=True, ax=ax_b, vmin=0, vmax=100,
                xticklabels=["Retained (0)", "Churned (1)"],
                yticklabels=["Retained (0)", "Churned (1)"],
                annot_kws={"fontsize": 10.5, "fontweight": "bold"})
    ax_b.set_title(f"Neural Network Confusion Matrix\n(Row-Normalized: % within Actual Class, N = {total_samples:,})", pad=15, fontweight="bold")
    ax_b.set_xlabel("Predicted Label (0 = Retained, 1 = Churned)", fontweight="bold")
    ax_b.set_ylabel("Actual Label (0 = Retained, 1 = Churned)", fontweight="bold")
    fig_b.savefig(CM_NORM_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_b)
    print(f"      Saved row-normalized confusion matrix -> {CM_NORM_FIGURE_PATH}")

    # --------------------------------------------------------------------------
    # 4. ROC Curve
    # --------------------------------------------------------------------------
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    auc_score = roc_auc_score(y_test, y_prob_test)

    ax_roc.plot(fpr, tpr, color="#6a3d9a", linewidth=2.5, label=f"Neural Network MLP (AUC = {auc_score:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="navy", linestyle="--", linewidth=1.5, label="Random Chance Baseline (AUC = 0.5000)")
    ax_roc.set_title("Neural Network Receiver Operating Characteristic (ROC) Curve\n(Test Set Evaluation, N = 1,409)",
                     pad=15, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.legend(loc="lower right")

    fig_roc.savefig(ROC_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig_roc)
    print(f"      Saved ROC curve -> {ROC_FIGURE_PATH}")


def save_artifacts(model: tf.keras.Model, metrics: dict, predictions_df: pd.DataFrame):
    """Saves serialized model, metrics JSON, and predictions CSV."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Save model in modern native Keras format
    model.save(MODEL_OUTPUT_PATH)
    print(f"      Saved trained model -> {MODEL_OUTPUT_PATH}")

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Saved evaluation metrics -> {METRICS_OUTPUT_PATH}")

    predictions_df.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print(f"      Saved predictions -> {PREDICTIONS_OUTPUT_PATH}")


def compare_all_models(nn_metrics: dict):
    """Prints empirical side-by-side comparison across all 3 trained models."""
    print("\n" + "=" * 80)
    print("      TRI-MODEL COMPARATIVE BENCHMARK (TEST SET EVALUATION)")
    print("=" * 80)

    lr_t = {}
    rf_t = {}

    if os.path.exists(LOGISTIC_METRICS_PATH):
        with open(LOGISTIC_METRICS_PATH, "r", encoding="utf-8") as f:
            lr_t = json.load(f)["test_metrics"]

    if os.path.exists(RF_METRICS_PATH):
        with open(RF_METRICS_PATH, "r", encoding="utf-8") as f:
            rf_t = json.load(f)["test_metrics"]

    nn_t = nn_metrics["test_metrics"]

    metrics_list = [
        "Accuracy", "Precision (Churn)", "Recall (Churn)", "F1-Score (Churn)", "ROC-AUC Score",
        "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)", "True Positives (TP)"
    ]

    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc",
            "true_negatives", "false_positives", "false_negatives", "true_positives"]

    comparison = pd.DataFrame({
        "Evaluation Metric": metrics_list,
        "Logistic Regression": [
            f"{lr_t.get(k, 0):.4f}" if "score" in k or "accuracy" in k or "precision" in k or "recall" in k or "auc" in k
            else f"{int(lr_t.get(k, 0)):,}" for k in keys
        ],
        "Random Forest": [
            f"{rf_t.get(k, 0):.4f}" if "score" in k or "accuracy" in k or "precision" in k or "recall" in k or "auc" in k
            else f"{int(rf_t.get(k, 0)):,}" for k in keys
        ],
        "Neural Network (MLP)": [
            f"{nn_t.get(k, 0):.4f}" if "score" in k or "accuracy" in k or "precision" in k or "recall" in k or "auc" in k
            else f"{int(nn_t.get(k, 0)):,}" for k in keys
        ]
    })

    print(comparison.to_string(index=False))
    print("=" * 80)


def run_neural_network_pipeline():
    """Main execution function for Phase 6."""
    print("=" * 60)
    print("   CUSTOMER CHURN PREDICTION - PHASE 6: NEURAL NETWORK (MLP)")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_datasets()

    # 2. Build model
    model = build_neural_network(input_dim=X_train.shape[1])

    # 3. Train model
    model, history, best_epoch = train_neural_network(model, X_train, y_train)

    # 4. Evaluate model
    metrics, predictions_df, cm, y_prob_test = evaluate_model(
        model, history, best_epoch, X_train, y_train, X_test, y_test
    )

    # 5. Visualizations
    generate_visualizations(history, best_epoch, cm, y_test, y_prob_test)

    # 6. Save artifacts
    save_artifacts(model, metrics, predictions_df)

    # 7. Tri-Model Comparative Benchmark
    compare_all_models(metrics)

    print("\n" + "=" * 60)
    print("   PHASE 6: NEURAL NETWORK COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_neural_network_pipeline()

