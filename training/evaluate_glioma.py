import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Paths
BASE_DIR = Path(r"D:\Model Train")
TEST_DIR = BASE_DIR / "dataset" / "test"
MODEL_PATH = BASE_DIR / "models" / "glioma_model.keras"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["No_Tumor", "Glioma_Tumor"]

def load_test_dataset():
    print("=" * 60)
    print("Loading Test Dataset for Glioma Model...")
    print("=" * 60)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )
    return test_ds

def plot_confusion_matrix(cm, accuracy):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["No Tumor (0)", "Glioma Tumor (1)"],
        yticklabels=["No Tumor (0)", "Glioma Tumor (1)"],
        title=f"Glioma Confusion Matrix (Accuracy: {accuracy:.2%})",
        ylabel="Actual Label",
        xlabel="Predicted Label"
    )

    # Annotate numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold"
            )

    plt.tight_layout()
    cm_path = OUTPUTS_DIR / "glioma_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved: {cm_path}")

def plot_roc_curve(y_true, y_pred_prob, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="forestgreen", lw=2.5, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Glioma Model ROC Curve (Test Set)", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    roc_path = OUTPUTS_DIR / "glioma_roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved: {roc_path}")

def evaluate():
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    test_ds = load_test_dataset()
    print(f"Loading trained Glioma model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Extract all true labels and predictions
    y_true = []
    y_pred_prob = []

    print("\nRunning inference on test dataset...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy().flatten())
        y_pred_prob.extend(preds.flatten())

    y_true = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("           GLIOMA TEST EVALUATION RESULTS           ")
    print("=" * 60)
    print(f"  Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision:  {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall:     {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1-Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No_Tumor (0)", "Glioma_Tumor (1)"]))

    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  TN (True No-Tumor): {cm[0, 0]} | FP (False Glioma): {cm[0, 1]}")
    print(f"  FN (Missed Glioma): {cm[1, 0]} | TP (Detected Glioma): {cm[1, 1]}")
    print("=" * 60)

    # Generate and save plots
    plot_confusion_matrix(cm, acc)
    plot_roc_curve(y_true, y_pred_prob, auc)

if __name__ == "__main__":
    evaluate()
