"""
predictor.py - Core AI Engine
Handles model loading, MRI prediction, and Grad-CAM heatmap generation.
"""

import io
import base64
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import tensorflow as tf

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pituitary_model.keras"
IMG_SIZE = (224, 224)

# -----------------------------------------------------------
# Global State (loaded once at startup)
# -----------------------------------------------------------
_model = None
_gradcam_model = None


def load_model():
    """Load trained Keras model and build Grad-CAM model. Called once at startup."""
    global _model, _gradcam_model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    print(f"[Predictor] Loading model from {MODEL_PATH}...")
    _model = tf.keras.models.load_model(str(MODEL_PATH))

    # Build Grad-CAM model
    _gradcam_model = _build_gradcam_model(_model)
    print("[Predictor] Model and Grad-CAM engine loaded successfully.")


def _build_gradcam_model(model):
    """
    Constructs a Grad-CAM model that outputs:
      1. Last conv layer feature maps (from MobileNetV2 backbone)
      2. Final sigmoid prediction score
    """
    backbone = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = backbone.get_layer("out_relu")

    # Sub-model: backbone input -> (last conv output, backbone final output)
    backbone_conv_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output]
    )

    # Rebuild forward pass (bypassing data augmentation for deterministic inference)
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="gradcam_input")
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    conv_output, backbone_out = backbone_conv_model(x)

    # Pass through classifier head
    x = model.get_layer("global_avg_pool")(backbone_out)
    x = model.get_layer("batch_normalization")(x)
    x = model.get_layer("dense_features")(x)
    x = model.get_layer("dropout")(x, training=False)
    predictions = model.get_layer("prediction")(x)

    return tf.keras.Model(inputs=inputs, outputs=[conv_output, predictions])


def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Converts raw uploaded image bytes into:
      - img_tensor: (1, 224, 224, 3) float32 tensor for model input
      - img_rgb: original RGB numpy array for visualization
    """
    # Decode image bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode the uploaded image. Please upload a valid MRI image (JPG/PNG).")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_tensor = tf.expand_dims(tf.cast(img_resized, tf.float32), axis=0)

    return img_tensor, img_bgr, img_rgb


def predict(image_bytes: bytes) -> dict:
    """
    Main prediction pipeline:
      1. Preprocess uploaded MRI image
      2. Run model inference
      3. Generate Grad-CAM heatmap
      4. Create side-by-side visualization
      5. Return prediction result with base64-encoded Grad-CAM image
    """
    if _model is None or _gradcam_model is None:
        raise RuntimeError("Model not loaded. Server may still be starting up.")

    # 1. Preprocess
    img_tensor, img_bgr, img_rgb = preprocess_image(image_bytes)

    # 2. Grad-CAM + Prediction (single forward pass with gradient tape)
    heatmap, pred_score = _compute_gradcam(img_tensor)

    # 3. Determine label and confidence
    is_tumor = float(pred_score) >= 0.5
    confidence = float(pred_score) if is_tumor else float(1.0 - pred_score)
    label = "Pituitary Tumor" if is_tumor else "No Tumor"

    # 4. Generate visualization
    gradcam_base64 = _create_visualization(img_bgr, img_rgb, heatmap, label, confidence)

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{confidence * 100:.2f}%",
        "raw_score": round(float(pred_score), 4),
        "gradcam_image": gradcam_base64
    }


def _compute_gradcam(img_tensor) -> tuple:
    """Computes Grad-CAM heatmap using gradient tape."""
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = _gradcam_model(img_tensor)
        loss = predictions[:, 0]

    # Gradient of prediction w.r.t. last conv layer feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients -> channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU: keep only positive influence
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize to [0, 1]
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), float(predictions[0][0])


def _create_visualization(img_bgr, img_rgb, heatmap, label, confidence) -> str:
    """
    Creates a 3-panel side-by-side visualization:
      Panel 1: Original MRI
      Panel 2: Grad-CAM Heatmap
      Panel 3: Superimposed Overlay
    Returns base64-encoded PNG string.
    """
    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply JET colormap
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # Superimpose heatmap onto original
    superimposed = cv2.addWeighted(img_bgr, 0.5, colored_heatmap, 0.5, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    # Determine color scheme
    color = "red" if label == "Pituitary Tumor" else "green"

    # Draw 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original MRI", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(colored_heatmap_rgb)
    axes[1].set_title("Grad-CAM Heatmap\n(Red = High Influence)", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(superimposed_rgb)
    axes[2].set_title(f"Overlay: {label}\nConfidence: {confidence * 100:.1f}%",
                      fontsize=13, fontweight="bold", color=color)
    axes[2].axis("off")

    plt.suptitle(
        f"AI Diagnosis: {label} | Confidence: {confidence * 100:.2f}%",
        fontsize=16, fontweight="bold", y=0.98, color=color
    )
    plt.tight_layout()

    # Convert plot to base64 PNG
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    b64_string = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    return b64_string
