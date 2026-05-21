

import os
import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K

IMG_SIZE = 224
DATASET_PATH = "brain_tumor_dataset"
ENSEMBLE_DIR = "ensemble_models"


def focal_loss(alpha=0.6, gamma=2.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def load_and_preprocess(image_path):
    """Load and preprocess a single image for prediction."""
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr


def main():
    print("\n" + "=" * 70)
    print("  FULL DATASET VALIDATION — Tumor vs No-Tumor Detection")
    print("=" * 70)

    # ── Load ensemble models ──
    custom_objects = {"focal_loss_fixed": focal_loss()}
    model_configs = [
        ("VGG16",       "HybridCNN_VGG16.h5",       0.4),
        ("MobileNetV2", "HybridCNN_MobileNetV2.h5",  0.3),
        ("ResNet50",    "HybridCNN_ResNet50.h5",      0.3),
    ]

    models = {}
    weights = {}
    for short_name, filename, w in model_configs:
        path = os.path.join(ENSEMBLE_DIR, filename)
        if os.path.exists(path):
            models[short_name] = load_model(path, custom_objects=custom_objects)
            weights[short_name] = w
            print(f"  ✅ Loaded: {short_name} (weight={w})")
        else:
            print(f"  ❌ Missing: {path}")

    if not models:
        print("\n  No models found. Train first!")
        sys.exit(1)

    # ── Gather all test images ──
    yes_dir = os.path.join(DATASET_PATH, "yes")
    no_dir = os.path.join(DATASET_PATH, "no")

    test_images = []
    for fname in sorted(os.listdir(yes_dir)):
        fpath = os.path.join(yes_dir, fname)
        if os.path.isfile(fpath):
            test_images.append((fpath, 1, "TUMOR"))
    for fname in sorted(os.listdir(no_dir)):
        fpath = os.path.join(no_dir, fname)
        if os.path.isfile(fpath):
            test_images.append((fpath, 0, "NO TUMOR"))

    print(f"\n  Total images: {len(test_images)}  "
          f"(Tumor: {sum(1 for _,l,_ in test_images if l==1)}, "
          f"No-Tumor: {sum(1 for _,l,_ in test_images if l==0)})")

    # ── Predict on every image ──
    THRESHOLD = 0.5

    # Storage: model_name -> list of (pred_value, true_label)
    all_preds = {name: [] for name in models}
    all_preds["ENSEMBLE"] = []

    misclassified = {name: [] for name in list(models.keys()) + ["ENSEMBLE"]}

    print(f"\n{'='*70}")
    print(f"  {'Image':<28} {'True':<10}", end="")
    for name in models:
        print(f" {name:<12}", end="")
    print(f" {'ENSEMBLE':<12}")
    print("-" * 70)

    for img_path, true_label, true_name in test_images:
        fname = os.path.basename(img_path)[:26]
        arr = load_and_preprocess(img_path)

        print(f"  {fname:<28} {true_name:<10}", end="")

        preds = {}
        for name, model in models.items():
            pred = float(model.predict(arr, verbose=0)[0][0])
            preds[name] = pred
            pred_label = 1 if pred > THRESHOLD else 0
            correct = pred_label == true_label
            mark = "✅" if correct else "❌"
            print(f" {pred:.3f}{mark}   ", end="")

            all_preds[name].append((pred, true_label))
            if not correct:
                misclassified[name].append((fname, true_name, pred))

        # Ensemble weighted soft voting
        ens_sum = sum(preds[n] * weights[n] for n in preds)
        ens_w = sum(weights[n] for n in preds)
        ens_pred = ens_sum / ens_w
        ens_label = 1 if ens_pred > THRESHOLD else 0
        ens_correct = ens_label == true_label
        mark = "✅" if ens_correct else "❌"
        print(f" {ens_pred:.3f}{mark}")

        all_preds["ENSEMBLE"].append((ens_pred, true_label))
        if not ens_correct:
            misclassified["ENSEMBLE"].append((fname, true_name, ens_pred))

    # ── Metrics ──
    print("\n" + "=" * 70)
    print("  DETAILED METRICS")
    print("=" * 70)
    print(f"\n  {'Model':<16} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}  "
          f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}")
    print("  " + "-" * 72)

    for name in list(models.keys()) + ["ENSEMBLE"]:
        preds_list = all_preds[name]
        tp = sum(1 for p, t in preds_list if p > THRESHOLD and t == 1)
        tn = sum(1 for p, t in preds_list if p <= THRESHOLD and t == 0)
        fp = sum(1 for p, t in preds_list if p > THRESHOLD and t == 0)
        fn = sum(1 for p, t in preds_list if p <= THRESHOLD and t == 1)

        total = tp + tn + fp + fn
        acc = (tp + tn) / total * 100 if total else 0
        prec = tp / (tp + fp) * 100 if (tp + fp) else 0
        rec = tp / (tp + fn) * 100 if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

        marker = "⭐" if name == "ENSEMBLE" else "  "
        print(f"{marker} {name:<16} {acc:6.1f}% {prec:6.1f}% {rec:6.1f}% {f1:6.1f}%  "
              f"{tp:4d} {tn:4d} {fp:4d} {fn:4d}")

    # ── Misclassified images ──
    print("\n" + "=" * 70)
    print("  MISCLASSIFIED IMAGES (ENSEMBLE)")
    print("=" * 70)

    ens_miss = misclassified["ENSEMBLE"]
    if not ens_miss:
        print("  ✅ PERFECT — No misclassifications by the ensemble!")
    else:
        print(f"  Total misclassified: {len(ens_miss)}\n")
        fp_list = [(f, p) for f, t, p in ens_miss if t == "NO TUMOR"]
        fn_list = [(f, p) for f, t, p in ens_miss if t == "TUMOR"]

        if fp_list:
            print("  FALSE POSITIVES (predicted Tumor, actually No Tumor):")
            for fname, pred in fp_list:
                print(f"    ❌ {fname:<30} pred={pred:.4f}")

        if fn_list:
            print("\n  FALSE NEGATIVES (predicted No Tumor, actually Tumor):")
            for fname, pred in fn_list:
                print(f"    ❌ {fname:<30} pred={pred:.4f}")

    # ── Final verdict ──
    ens_data = all_preds["ENSEMBLE"]
    ens_acc = sum(1 for p, t in ens_data if (p > THRESHOLD) == (t == 1)) / len(ens_data) * 100

    print("\n" + "=" * 70)
    if ens_acc >= 95:
        print(f"  ✅ EXCELLENT — Ensemble accuracy: {ens_acc:.1f}%")
    elif ens_acc >= 85:
        print(f"  ✅ GOOD — Ensemble accuracy: {ens_acc:.1f}%")
    elif ens_acc >= 70:
        print(f"  ⚠️  MODERATE — Ensemble accuracy: {ens_acc:.1f}% (needs improvement)")
    else:
        print(f"  ❌ POOR — Ensemble accuracy: {ens_acc:.1f}% (model needs retraining)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
