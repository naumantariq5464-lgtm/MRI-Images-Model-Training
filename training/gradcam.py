import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# Paths
BASE_DIR = Path(r"D:\Model Train")
MODEL_PATH = BASE_DIR / "models" / "pituitary_model.keras"
OUTPUTS_DIR = BASE_DIR / "outputs" / "gradcam"
TEST_DIR = BASE_DIR / "dataset" / "test"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (224, 224)

def get_gradcam_model(model, backbone_layer_name="mobilenetv2_1.00_224", last_conv_name="out_relu"):
    """
    Builds a Grad-CAM model that outputs:
    1. The feature maps from the backbone's last conv layer
    2. The final prediction score of the model
    """
    backbone = model.get_layer(backbone_layer_name)
    last_conv_layer = backbone.get_layer(last_conv_name)
    
    # Model that maps backbone inputs -> backbone last conv output
    backbone_conv_model = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[last_conv_layer.output, backbone.output]
    )

    # Input to whole model
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="gradcam_input")
    
    # 1. Bypass random data augmentation for deterministic test-time Grad-CAM
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # 2. Get conv output and backbone pooled output
    conv_output, backbone_out = backbone_conv_model(x)
    
    # 3. Pass through remaining top classifier layers
    x = model.get_layer("global_avg_pool")(backbone_out)
    x = model.get_layer("batch_normalization")(x)
    x = model.get_layer("dense_features")(x)
    x = model.get_layer("dropout")(x, training=False)
    predictions = model.get_layer("prediction")(x)

    gradcam_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, predictions])
    return gradcam_model

def compute_gradcam_heatmap(img_array, gradcam_model):
    """
    Generates Grad-CAM heatmap array of shape (224, 224)
    """
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = gradcam_model(img_array)
        loss = predictions[:, 0]  # Score for Pituitary Tumor (class 1)

    # Gradient of the class score with respect to the feature map activations
    grads = tape.gradient(loss, conv_outputs)

    # Global Average Pooling of the gradients (importance weights)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the channels by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to keep only features that have a positive influence on prediction
    heatmap = tf.maximum(heatmap, 0.0)
    
    # Normalize between 0 and 1
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy(), float(predictions[0][0])

def overlay_gradcam(original_img_bgr, heatmap, alpha=0.55):
    """
    Overlays colored heatmap on original image using OpenCV JET colormap
    """
    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (original_img_bgr.shape[1], original_img_bgr.shape[0]))
    
    # Convert heatmap to uint8 (0 to 255)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply JET colormap (Red = high activation, Blue = low activation)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    # Superimpose heatmap onto original image
    superimposed = cv2.addWeighted(original_img_bgr, 1.0 - alpha, colored_heatmap, alpha, 0)
    return colored_heatmap, superimposed

def process_and_save_gradcam(image_path, model, gradcam_model, output_save_path):
    """
    Processes single MRI, runs inference + Grad-CAM, and saves 3-panel visualization
    """
    # 1. Load image using OpenCV
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"Error loading {image_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_tensor = tf.expand_dims(tf.cast(img_resized, tf.float32), axis=0)

    # 2. Compute Grad-CAM
    heatmap, pred_score = compute_gradcam_heatmap(img_tensor, gradcam_model)

    # 3. Label & Confidence
    is_tumor = pred_score >= 0.5
    confidence = pred_score if is_tumor else (1.0 - pred_score)
    label = "Pituitary Tumor" if is_tumor else "No Tumor"
    color = "red" if is_tumor else "green"

    # 4. Generate overlays
    colored_heatmap, superimposed_bgr = overlay_gradcam(img_bgr, heatmap, alpha=0.5)
    superimposed_rgb = cv2.cvtColor(superimposed_bgr, cv2.COLOR_BGR2RGB)
    colored_heatmap_rgb = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

    # 5. Plot 3-panel side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original MRI
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original MRI Image\n({Path(image_path).name})", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im_heat = axes[1].imshow(colored_heatmap_rgb)
    axes[1].set_title("Grad-CAM Activation Heatmap\n(Red: High Influence Area)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Superimposed Result
    axes[2].imshow(superimposed_rgb)
    axes[2].set_title(f"Grad-CAM Overlay\nPrediction: {label} ({confidence * 100:.1f}%)", 
                      fontsize=12, fontweight="bold", color=color)
    axes[2].axis("off")

    plt.suptitle(
        f"AI Diagnosis: {label} | Confidence: {confidence * 100:.2f}%", 
        fontsize=16, fontweight="heavy", y=0.98, color=color
    )
    plt.tight_layout()
    plt.savefig(output_save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated Grad-CAM: {output_save_path.name} -> {label} ({confidence * 100:.2f}%)")

def run_gradcam_demo():
    print("=" * 60)
    print("STEP 4: Initializing Grad-CAM Explainable AI Engine...")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(str(MODEL_PATH))
    gradcam_model = get_gradcam_model(model)

    # Select sample images from test set
    pituitary_samples = list((TEST_DIR / "Pituitary_Tumor").glob("*.*"))[:4]
    no_tumor_samples = list((TEST_DIR / "No_Tumor").glob("*.*"))[:2]
    all_samples = pituitary_samples + no_tumor_samples

    print(f"Generating Grad-CAM for {len(all_samples)} sample test MRI images...")

    for idx, sample in enumerate(all_samples):
        prefix = "pituitary_tumor" if "Pituitary" in str(sample) else "no_tumor"
        out_file = OUTPUTS_DIR / f"gradcam_sample_{idx + 1}_{prefix}.png"
        process_and_save_gradcam(sample, model, gradcam_model, out_file)

    print("\n" + "=" * 60)
    print(f">>> STEP 4 COMPLETE! Grad-CAM visualizations saved to:")
    print(f"    {OUTPUTS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    run_gradcam_demo()
