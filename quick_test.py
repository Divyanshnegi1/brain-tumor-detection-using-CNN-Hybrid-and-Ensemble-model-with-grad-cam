
import os
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

def focal_loss(alpha=0.6, gamma=2.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

custom_objects = {"focal_loss_fixed": focal_loss()}

# ── Load all models ──
print("=" * 65)
print("  QUICK DETECTION TEST — Tumor vs No-Tumor")
print("=" * 65)

models = {}
for name, path in [
    ("Primary (VGG16)", "brain_tumor_model_balanced.h5"),
    ("Ensemble VGG16", "ensemble_models/HybridCNN_VGG16.h5"),
    ("Ensemble MobileNetV2", "ensemble_models/HybridCNN_MobileNetV2.h5"),
    ("Ensemble ResNet50", "ensemble_models/HybridCNN_ResNet50.h5"),
]:
    if os.path.exists(path):
        models[name] = load_model(path, custom_objects=custom_objects)
        print(f"  ✅ Loaded: {name}")
    else:
        print(f"  ❌ Missing: {path}")

# ── Test images ──
tumor_images = [
    ("brain_tumor_dataset/yes/Y1.jpg", "TUMOR"),
    ("brain_tumor_dataset/yes/Y2.jpg", "TUMOR"),
    ("brain_tumor_dataset/yes/Y10.jpg", "TUMOR"),
    ("brain_tumor_dataset/yes/Y3.jpg", "TUMOR"),
    ("brain_tumor_dataset/yes/Y4.jpg", "TUMOR"),
    ("brain_tumor_dataset/yes/Y25.jpg", "TUMOR"),
]

no_tumor_images = [
    ("brain_tumor_dataset/no/6 no.jpg", "NO TUMOR"),
    ("brain_tumor_dataset/no/1 no.jpeg", "NO TUMOR"),
    ("brain_tumor_dataset/no/10 no.jpg", "NO TUMOR"),
    ("brain_tumor_dataset/no/3 no.jpg", "NO TUMOR"),
    ("brain_tumor_dataset/no/N1.JPG", "NO TUMOR"),
    ("brain_tumor_dataset/no/No11.jpg", "NO TUMOR"),
]

all_tests = tumor_images + no_tumor_images

def predict(model, image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    pred = float(model.predict(arr, verbose=0)[0][0])
    return pred

# ── Run tests ──
print("\n" + "=" * 65)
print(f"  {'Image':<22} {'True':<12} ", end="")
for name in models:
    short = name.split("(")[-1].replace(")", "").strip() if "(" in name else name.split()[-1]
    print(f" {short:<10}", end="")
print("  Ensemble")
print("-" * 65)

correct_counts = {name: 0 for name in models}
correct_counts["Ensemble"] = 0
total = 0

ens_weights = {"Ensemble VGG16": 0.4, "Ensemble MobileNetV2": 0.3, "Ensemble ResNet50": 0.3}

for img_path, true_label in all_tests:
    if not os.path.exists(img_path):
        continue
    total += 1
    fname = os.path.basename(img_path)[:20]
    print(f"  {fname:<22} {true_label:<12} ", end="")

    preds = {}
    for name, model in models.items():
        pred = predict(model, img_path)
        preds[name] = pred
        predicted_label = "TUMOR" if pred > 0.55 else "NO TUMOR"
        is_correct = predicted_label == true_label
        if is_correct:
            correct_counts[name] += 1
        mark = "✅" if is_correct else "❌"
        print(f" {pred:.3f}{mark}  ", end="")

    # Ensemble prediction
    ens_models = ["Ensemble VGG16", "Ensemble MobileNetV2", "Ensemble ResNet50"]
    ens_sum = sum(preds.get(n, 0) * ens_weights.get(n, 0) for n in ens_models if n in preds)
    ens_total_w = sum(ens_weights.get(n, 0) for n in ens_models if n in preds)
    if ens_total_w > 0:
        ens_pred = ens_sum / ens_total_w
        ens_label = "TUMOR" if ens_pred > 0.55 else "NO TUMOR"
        ens_correct = ens_label == true_label
        if ens_correct:
            correct_counts["Ensemble"] += 1
        mark = "✅" if ens_correct else "❌"
        print(f" {ens_pred:.3f}{mark}")
    else:
        print("  N/A")

# ── Summary ──
print("\n" + "=" * 65)
print("  ACCURACY SUMMARY")
print("-" * 65)
for name in list(models.keys()) + ["Ensemble"]:
    acc = correct_counts[name] / total * 100 if total > 0 else 0
    status = "✅" if acc >= 80 else ("⚠️" if acc >= 60 else "❌")
    print(f"  {status} {name:<30} {correct_counts[name]}/{total} correct  ({acc:.0f}%)")
print("=" * 65)
