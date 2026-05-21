

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import cv2


def predict_with_ensemble(ensemble, image_path):
    """
    Run ensemble prediction on a single image with detailed output.

    Args:
        ensemble: EnsembleClassifier instance
        image_path: Path to MRI image
    """
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE PREDICTION")
    print(f"{'='*60}")
    print(f"  Image: {os.path.basename(image_path)}")

    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return

    ensemble_pred, individual = ensemble.predict_image(image_path)
    if ensemble_pred is None:
        return

    _plot_prediction_comparison(individual, ensemble_pred, image_path)


def _plot_prediction_comparison(individual_preds, ensemble_pred, image_path):
    """Plot side-by-side comparison: original image + prediction bars"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (224, 224))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input MRI Scan", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    names = list(individual_preds.keys()) + ["ENSEMBLE"]
    values = list(individual_preds.values()) + [ensemble_pred]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]

    bars = axes[1].barh(names, values, color=colors[:len(names)], edgecolor="black")
    axes[1].axvline(x=0.5, color="gray", linestyle="--", linewidth=1.5, label="Threshold")
    for bar, val in zip(bars, values):
        label = "Tumor" if val > 0.5 else "No Tumor"
        axes[1].text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f} ({label})", va="center", fontweight="bold", fontsize=10)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("Model Predictions (Soft Voting)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Prediction Probability")
    axes[1].legend()
    axes[1].grid(axis="x", alpha=0.3)

    plt.suptitle("Ensemble Brain Tumor Prediction", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.show()


def ensemble_gradcam_all_models(ensemble, image_path):
    """
    Generate Grad-CAM++ heatmaps from each model in the ensemble
    and display them side by side for comparison.

    Uses the professional grad_cam_visualization module for consistent,
    high-quality, noise-free heatmaps across all architectures.

    Args:
        ensemble: EnsembleClassifier instance
        image_path: Path to MRI image
    """
    from grad_cam_visualization import generate_gradcam_for_model

    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        return

    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    original_img = cv2.imread(image_path)
    original_img = cv2.resize(original_img, (224, 224))

    n_models = len(ensemble.models)
    fig, axes = plt.subplots(2, n_models + 1, figsize=(5 * (n_models + 1), 9))
    fig.patch.set_facecolor('#1a1a2e')

    axes[0][0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axes[0][0].set_title("Original MRI", fontsize=11, fontweight="bold", color="white")
    axes[0][0].axis("off")
    axes[1][0].axis("off")

    for i, (name, model, _) in enumerate(ensemble.models):
        col = i + 1
        try:
            heatmap_norm, heatmap_uint8, overlay, pred_val, layer_used = \
                generate_gradcam_for_model(model, img_array, original_img)

            if heatmap_uint8 is None:
                axes[0][col].text(0.5, 0.5, "Grad-CAM failed",
                                  ha="center", color="white",
                                  transform=axes[0][col].transAxes)
                axes[0][col].set_facecolor('#1a1a2e')
                axes[1][col].set_facecolor('#1a1a2e')
                axes[0][col].axis("off")
                axes[1][col].axis("off")
                continue

            axes[0][col].imshow(heatmap_uint8, cmap="jet", vmin=0, vmax=255)
            axes[0][col].set_title(f"{name}\nPred: {pred_val:.3f}",
                                    fontsize=10, fontweight="bold", color="white")
            axes[0][col].axis("off")

            axes[1][col].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            axes[1][col].set_title(f"Overlay ({layer_used})",
                                    fontsize=9, color="white")
            axes[1][col].axis("off")

        except Exception as e:
            axes[0][col].text(0.5, 0.5, f"Error:\n{str(e)[:40]}",
                              ha="center", fontsize=8, color="white",
                              transform=axes[0][col].transAxes)
            axes[0][col].axis("off")
            axes[1][col].axis("off")

    plt.suptitle("Grad-CAM++ Comparison Across Ensemble Models",
                  fontsize=14, fontweight="bold", color="white")
    plt.tight_layout()
    plt.show()


def demonstrate_ensemble_architecture():
    """Print the ensemble architecture diagram for academic presentation"""
    print("\n" + "="*60)
    print("  ENSEMBLE ARCHITECTURE")
    print("="*60)
    print("""
    INPUT MRI IMAGE (224x224x3)
           |
    +------+------+----------------+
    |             |                |
    v             v                v
  VGG16       MobileNetV2     ResNet50
  (Hybrid)    (Hybrid)        (Hybrid)
    |             |                |
    v             v                v
  P1(x)        P2(x)           P3(x)
  w=0.4        w=0.3           w=0.3
    |             |                |
    +------+------+----------------+
           |
           v
    WEIGHTED SOFT VOTING
    P(x) = (0.4*P1 + 0.3*P2 + 0.3*P3)
           |
           v
    FINAL PREDICTION
    Tumor / No Tumor
    """)


if __name__ == "__main__":
    demonstrate_ensemble_architecture()
    print("  Ensemble prediction module ready.")