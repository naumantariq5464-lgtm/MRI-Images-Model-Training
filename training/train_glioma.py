import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2

# -------------------------------------------------------------
# Configuration & Paths
# -------------------------------------------------------------
BASE_DIR = Path(r"D:\Model Train")
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 1e-4

CLASS_NAMES = ["No_Tumor", "Glioma_Tumor"]
# No_Tumor = 0, Glioma_Tumor = 1

def load_datasets():
    print("=" * 60)
    print("Loading Training and Validation Datasets for Glioma Model...")
    print(f"Classes: {CLASS_NAMES}")
    print("=" * 60)
    
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )

    # Performance optimization (caching and prefetching)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def calculate_class_weights():
    """Calculate balanced class weights to handle No_Tumor vs Glioma_Tumor imbalance"""
    no_tumor_count = len(list((TRAIN_DIR / "No_Tumor").glob("*.*")))
    glioma_count = len(list((TRAIN_DIR / "Glioma_Tumor").glob("*.*")))
    total_samples = no_tumor_count + glioma_count
    
    # Class 0: No_Tumor, Class 1: Glioma_Tumor
    y_train = np.array([0] * no_tumor_count + [1] * glioma_count)
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {cls: float(weight) for cls, weight in zip(classes, weights)}
    
    print("\n" + "-" * 50)
    print(f"Dataset Balance in Training Set:")
    print(f"  - No_Tumor (Class 0):     {no_tumor_count} images -> Weight: {class_weight_dict[0]:.4f}")
    print(f"  - Glioma_Tumor (Class 1): {glioma_count} images -> Weight: {class_weight_dict[1]:.4f}")
    print("-" * 50 + "\n")
    return class_weight_dict

def build_model():
    print("=" * 60)
    print("Building MobileNetV2 Transfer Learning Model for Glioma...")
    print("=" * 60)

    # 1. Data Augmentation Pipeline
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1)
    ], name="data_augmentation")

    # 2. Input Layer
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="mri_input")
    x = data_augmentation(inputs)

    # 3. MobileNetV2 Preprocessing (scales pixel values to [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # 4. Pretrained MobileNetV2 Base (ImageNet)
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained backbone
    
    x = base_model(x, training=False)

    # 5. Custom Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="GliomaTumor_MobileNetV2")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model.summary()
    return model

def plot_training_history(history):
    print("\nSaving training curves to outputs/glioma_training_curves.png...")
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs_range, acc, "b-o", label="Training Accuracy")
    ax1.plot(epochs_range, val_acc, "r--s", label="Validation Accuracy")
    ax1.set_title("Glioma Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Loss Plot
    ax2.plot(epochs_range, loss, "b-o", label="Training Loss")
    ax2.plot(epochs_range, val_loss, "r--s", label="Validation Loss")
    ax2.set_title("Glioma Training & Validation Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = OUTPUTS_DIR / "glioma_training_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved: {plot_path}")

def train():
    train_ds, val_ds = load_datasets()
    class_weights = calculate_class_weights()
    model = build_model()

    saved_model_path = MODELS_DIR / "glioma_model.keras"

    # Callbacks for robust training
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(saved_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    early_stopping_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("=" * 60)
    print(f"Starting Training for Glioma Model ({EPOCHS} Epochs max with EarlyStopping)...")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    plot_training_history(history)

    print("\n" + "=" * 60)
    print(">>> GLIOMA TRAINING COMPLETE! Best model saved to:")
    print(f"    {saved_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    train()
