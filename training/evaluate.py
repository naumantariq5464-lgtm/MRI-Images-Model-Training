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
MODEL_PATH = BASE_DIR / "models" / "pituitary_model.keras"
OUTPUTS_DIR = BASE_DIR / "outputs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["No_Tumor", "Pituitary_Tumor"]

def load_test_dataset():
    print("=" * 60)
    print("Loading Test Dataset (Unseen Images)...")
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
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=["No Tumor (0)", "Pituitary Tumor (1)"],
        yticklabels=["No Tumor (0)", "Pituitary Tumor (1)"],
        title=f"Confusion Matrix (Test Accuracy: {accuracy:.2%})",
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
    cm_path = OUTPUTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion Matrix saved to: {cm_path}")

def plot_roc_curve(y_true, y_probs, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate (Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    roc_path = OUTPUTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"ROC Curve saved to: {roc_path}")

def evaluate():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("=" * 60)
    print("Loading Trained Model for Evaluation...")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 60)
    model = tf.keras.models.load_model(str(MODEL_PATH))

    test_ds = load_test_dataset()

    # Gather true labels and predictions
    y_true = []
    y_probs = []

    print("\nRunning Inference on Test Dataset...")
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_probs.extend(preds.flatten())
        y_true.extend(labels.numpy().flatten())

    y_true = np.array(y_true, dtype=int)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= 0.5).astype(int)

    # Metrics calculation
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 60)
    print("           FINAL TEST EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Test Samples Evaluated: {len(y_true)}")
    print(f"  - No Tumor Samples:        {np.sum(y_true == 0)}")
    print(f"  - Pituitary Tumor Samples: {np.sum(y_true == 1)}")
    print("-" * 60)
    print(f"  Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"  Recall:    {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  F1-Score:  {f1:.4f} ({f1 * 100:.2f}%)")
    print(f"  ROC-AUC:   {auc:.4f} ({auc * 100:.2f}%)")
    print("-" * 60)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["No_Tumor (0)", "Pituitary_Tumor (1)"], digits=4))

    # Save visual outputs
    plot_confusion_matrix(cm, acc)
    plot_roc_curve(y_true, y_probs, auc)

    print("\n" + "=" * 60)
    print(">>> STEP 3 COMPLETE! Evaluation artifacts saved in 'outputs/' <<<")
    print("=" * 60)

if __name__ == "__main__":
    evaluate()
