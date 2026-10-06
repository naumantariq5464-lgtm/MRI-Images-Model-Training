"""
predictor.py - Core AI Engine (Unified Multi-Model)
Handles loading of both Pituitary and Glioma models,
runs unified prediction, and generates Grad-CAM heatmaps.
"""

import io
import base64
from pathlib import Path
import numpy as np
import cv2
import tensorflow as tf

# -----------------------------------------------------------
# Configuration
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
PITUITARY_MODEL_PATH = BASE_DIR / "models" / "pituitary_model.keras"
GLIOMA_MODEL_PATH = BASE_DIR / "models" / "glioma_model.keras"
IMG_SIZE = (224, 224)

# -----------------------------------------------------------
# Global State (loaded once at startup)
# -----------------------------------------------------------
_pituitary_model = None
_glioma_model = None
_pituitary_gradcam = None
_glioma_gradcam = None


def load_model():
    """Load both trained Keras models and build Grad-CAM models. Called once at startup."""
    global _pituitary_model, _glioma_model, _pituitary_gradcam, _glioma_gradcam

    # Load Pituitary Model
    if PITUITARY_MODEL_PATH.exists():
        print(f"[Predictor] Loading Pituitary model from {PITUITARY_MODEL_PATH}...")
        _pituitary_model = tf.keras.models.load_model(str(PITUITARY_MODEL_PATH))
        _pituitary_gradcam = _build_gradcam_model(_pituitary_model)
        print("[Predictor] Pituitary model + Grad-CAM loaded.")
    else:
        print(f"[Predictor] WARNING: Pituitary model not found at {PITUITARY_MODEL_PATH}")

    # Load Glioma Model
    if GLIOMA_MODEL_PATH.exists():
        print(f"[Predictor] Loading Glioma model from {GLIOMA_MODEL_PATH}...")
        _glioma_model = tf.keras.models.load_model(str(GLIOMA_MODEL_PATH))
        _glioma_gradcam = _build_gradcam_model(_glioma_model)
        print("[Predictor] Glioma model + Grad-CAM loaded.")
    else:
        print(f"[Predictor] WARNING: Glioma model not found at {GLIOMA_MODEL_PATH}")

    if _pituitary_model is None and _glioma_model is None:
        raise FileNotFoundError("No models found! At least one model is required.")

    print("[Predictor] All available models loaded successfully.")


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


def _validate_mri_image(img_bgr: np.ndarray):
    """
    Validates if an uploaded image is a valid Brain MRI scan vs a non-MRI photo.
    Checks:
      1. Grayscale / Color Saturation: MRI scans are monochrome (low saturation & channel variance).
      2. Background Darkness: MRI scans have dark background borders.
    Raises ValueError with descriptive message if non-MRI image detected.
    """
    if img_bgr is None:
        raise ValueError("Could not decode the uploaded image. Please upload a valid image file (JPG/PNG).")

    # 1. Color Saturation & Channel Difference Check (MRI is grayscale)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_saturation = float(np.mean(hsv[:, :, 1]))

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    channel_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(b - r)))

    if mean_saturation > 25.0 or channel_diff > 15.0:
        raise ValueError("Invalid Image: Uploaded file is not a valid Brain MRI scan. Brain MRI scans must be grayscale images.")

    # 2. Dark Background Border Check (MRI scans have dark border surroundings)
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    corner_top_left = gray[0:int(h * 0.08), 0:int(w * 0.08)]
    corner_top_right = gray[0:int(h * 0.08), int(w * 0.92):w]
    corner_bot_left = gray[int(h * 0.92):h, 0:int(w * 0.08)]
    corner_bot_right = gray[int(h * 0.92):h, int(w * 0.92):w]

    corner_avg = float(np.mean([
        np.mean(corner_top_left),
        np.mean(corner_top_right),
        np.mean(corner_bot_left),
        np.mean(corner_bot_right)
    ]))

    if corner_avg > 90.0:
        raise ValueError("Invalid Image: Uploaded file does not match Brain MRI characteristics (bright background detected). Please upload a valid Brain MRI scan.")


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

    # Validate image is a valid Brain MRI scan
    _validate_mri_image(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_tensor = tf.expand_dims(tf.cast(img_resized, tf.float32), axis=0)

    return img_tensor, img_bgr, img_rgb



def predict(image_bytes: bytes) -> dict:
    """
    Unified prediction pipeline:
      1. Preprocess uploaded MRI image
      2. Run BOTH models (Pituitary & Glioma)
      3. Determine the best prediction across all models
      4. Generate Grad-CAM heatmap from the winning model
      5. Return unified result
    """
    if _pituitary_model is None and _glioma_model is None:
        raise RuntimeError("No models loaded. Server may still be starting up.")

    # 1. Preprocess
    img_tensor, img_bgr, img_rgb = preprocess_image(image_bytes)

    # 2. Run both models and collect tumor scores
    results = []

    # Pituitary Model: score >= 0.5 means Pituitary Tumor
    if _pituitary_model is not None and _pituitary_gradcam is not None:
        pit_heatmap, pit_score = _compute_gradcam(img_tensor, _pituitary_gradcam)
        pit_tumor_conf = float(pit_score)  # Higher = more likely Pituitary Tumor
        results.append({
            "label": "Pituitary Tumor",
            "tumor_confidence": pit_tumor_conf,
            "heatmap": pit_heatmap,
            "gradcam_model": _pituitary_gradcam
        })

    # Glioma Model: score >= 0.5 means Glioma Tumor
    if _glioma_model is not None and _glioma_gradcam is not None:
        gli_heatmap, gli_score = _compute_gradcam(img_tensor, _glioma_gradcam)
        gli_tumor_conf = float(gli_score)  # Higher = more likely Glioma Tumor
        results.append({
            "label": "Glioma Tumor",
            "tumor_confidence": gli_tumor_conf,
            "heatmap": gli_heatmap,
            "gradcam_model": _glioma_gradcam
        })

    # 3. Decision Logic:
    #    - If BOTH models say "No Tumor" (scores < 0.5), result = "No Tumor"
    #    - If one or both models say "Tumor" (score >= 0.5), pick the one with highest confidence
    tumor_detections = [r for r in results if r["tumor_confidence"] >= 0.5]

    if len(tumor_detections) == 0:
        # No tumor detected by any model
        # Use the model with the LOWEST tumor score (most confident "No Tumor")
        best = min(results, key=lambda r: r["tumor_confidence"])
        label = "No Tumor"
        confidence = float(1.0 - best["tumor_confidence"])
        raw_score = best["tumor_confidence"]
        heatmap = best["heatmap"]
    else:
        # One or more models detected tumor — pick the most confident one
        best = max(tumor_detections, key=lambda r: r["tumor_confidence"])
        label = best["label"]
        confidence = best["tumor_confidence"]
        raw_score = best["tumor_confidence"]
        heatmap = best["heatmap"]

    # 4. Generate visualization using the winning model's heatmap
    gradcam_base64 = _create_visualization(img_bgr, img_rgb, heatmap, label, confidence)

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "confidence_percent": f"{confidence * 100:.2f}%",
        "raw_score": round(raw_score, 4),
        "gradcam_image": gradcam_base64
    }


def _compute_gradcam(img_tensor, gradcam_model) -> tuple:
    """Computes Grad-CAM heatmap using gradient tape for a given model."""
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = gradcam_model(img_tensor)
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
    Creates a 3-panel side-by-side visualization using OpenCV:
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

    # Superimpose heatmap onto original
    superimposed = cv2.addWeighted(img_bgr, 0.5, colored_heatmap, 0.5, 0)

    # Create a simple concatenated image (Panel 1 | Panel 2 | Panel 3)
    # Using BGR for cv2.imencode
    border_size = 5
    border_color = [255, 255, 255] # White border
    
    p1 = cv2.copyMakeBorder(img_bgr, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    p2 = cv2.copyMakeBorder(colored_heatmap, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)
    p3 = cv2.copyMakeBorder(superimposed, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

    combined = cv2.hconcat([p1, p2, p3])
    
    # Add a top bar for title
    top_bar = 50
    final_img = cv2.copyMakeBorder(combined, top_bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Color based on tumor type
    if label == "No Tumor":
        color = (0, 255, 0)   # Green (BGR)
    elif label == "Pituitary Tumor":
        color = (0, 0, 255)   # Red (BGR)
    elif label == "Glioma Tumor":
        color = (0, 140, 255) # Orange (BGR)
    else:
        color = (255, 255, 255)

    title_text = f"AI Diagnosis: {label} | Confidence: {confidence * 100:.2f}%"
    cv2.putText(final_img, title_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    # Encode to base64
    _, buffer = cv2.imencode('.png', final_img)
    b64_string = base64.b64encode(buffer).decode("utf-8")

    return b64_string
