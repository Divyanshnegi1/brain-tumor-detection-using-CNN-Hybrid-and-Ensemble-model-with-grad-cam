

import streamlit as st
import numpy as np
import cv2
import os
import io
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE = 224
MODEL_PATH = "brain_tumor_model_balanced.h5"
ENSEMBLE_DIR = "ensemble_models"
DATASET_PATH = "brain_tumor_dataset"

# ── Loss function (needed to load saved models) ───────────────────────────
def focal_loss(alpha=0.6, gamma=2.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1.0 - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1.0 - pt_0))
    return focal_loss_fixed

# ── Model loading (cached) ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_primary_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss()})

@st.cache_resource(show_spinner=False)
def load_ensemble_models():
    configs = [
        ("VGG16", "HybridCNN_VGG16.h5", 0.4),
        ("MobileNetV2", "HybridCNN_MobileNetV2.h5", 0.3),
        ("ResNet50", "HybridCNN_ResNet50.h5", 0.3),
    ]
    models = []
    for name, fname, weight in configs:
        path = os.path.join(ENSEMBLE_DIR, fname)
        if os.path.exists(path):
            m = load_model(path, custom_objects={"focal_loss_fixed": focal_loss()})
            models.append((name, m, weight))
    return models if models else None

# ── Prediction helpers ────────────────────────────────────────────────────
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_single(model, img_array):
    pred = float(model.predict(img_array, verbose=0)[0][0])
    if pred > 0.55:
        return "Tumor Detected", pred * 100, True
    return "No Tumor Detected", (1 - pred) * 100, False

def predict_ensemble(models, img_array):
    results = {}
    weighted_sum = 0.0
    total_weight = 0.0
    for name, model, weight in models:
        pred = float(model.predict(img_array, verbose=0)[0][0])
        results[name] = pred
        weighted_sum += pred * weight
        total_weight += weight
    ensemble_pred = weighted_sum / total_weight
    return ensemble_pred, results

# ── Grad-CAM generator (uses professional module) ────────────────────────
from grad_cam_visualization import generate_gradcam_for_model

def generate_gradcam(model, img_array, original_bgr):
    """
    Generate medical-grade Grad-CAM++ heatmap and overlay.
    Delegates to the professional grad_cam_visualization module.

    Returns:
        heatmap_normalized (float32 0-1), heatmap_uint8, overlay_bgr
    """
    heatmap_norm, heatmap_uint8, overlay, pred_val, layer_used = \
        generate_gradcam_for_model(model, img_array, original_bgr)
    return heatmap_norm, heatmap_uint8, overlay

def pil_to_bgr(pil_img):
    img = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb = np.array(img)
    if len(rgb.shape) == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

# ── Matplotlib chart helper ───────────────────────────────────────────────
def ensemble_bar_chart(individual, ensemble_val):
    names = list(individual.keys()) + ["Ensemble"]
    values = list(individual.values()) + [ensemble_val]
    colors = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")

    bars = ax.barh(names, values, color=colors[: len(names)], edgecolor="#334155", height=0.55)
    ax.axvline(x=0.5, color="#94a3b8", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
    for bar, val in zip(bars, values):
        label = "Tumor" if val > 0.5 else "No Tumor"
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} ({label})", va="center", fontweight="bold",
                fontsize=9, color="#e2e8f0")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Prediction Probability", color="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    plt.tight_layout()
    return fig

def accuracy_comparison_chart(model_names, accuracies):
    """Vertical bar chart comparing model accuracies"""
    colors = ["#3b82f6", "#22c55e", "#ef4444", "#f59e0b"]
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    bars = ax.bar(model_names, accuracies, color=colors[:len(model_names)],
                  edgecolor="#334155", width=0.5)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=11, color="#e2e8f0")
    ax.set_ylabel("Accuracy (%)", color="#94a3b8", fontsize=11)
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.grid(axis="y", alpha=0.15, color="#94a3b8")
    ax.set_title("Model Accuracy Comparison", color="#e2e8f0", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig

def confusion_matrix_chart(cm, labels):
    """Confusion matrix heatmap"""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    import matplotlib.colors as mcolors
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_med", ["#1e293b", "#1e40af", "#3b82f6", "#93c5fd"])
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, color="#94a3b8", fontsize=10)
    ax.set_yticklabels(labels, color="#94a3b8", fontsize=10)
    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Actual", color="#94a3b8", fontsize=11)
    ax.set_title("Confusion Matrix", color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    return fig

def roc_curve_chart(fpr, tpr, auc_val):
    """ROC curve chart"""
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    ax.plot(fpr, tpr, color="#3b82f6", linewidth=2.5, label=f"AUC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], color="#475569", linestyle="--", linewidth=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3b82f6")
    ax.set_xlabel("False Positive Rate", color="#94a3b8", fontsize=10)
    ax.set_ylabel("True Positive Rate", color="#94a3b8", fontsize=10)
    ax.set_title("ROC Curve", color="#e2e8f0", fontsize=13, fontweight="bold", pad=10)
    ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8", fontsize=10)
    ax.tick_params(colors="#94a3b8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["left"].set_color("#334155")
    ax.grid(alpha=0.1, color="#94a3b8")
    plt.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def run_model_evaluation():
    """Evaluate primary model + ensemble on validation set and return metrics"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, roc_curve, auc

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)
    eval_gen = eval_datagen.flow_from_directory(
        DATASET_PATH, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32, class_mode="binary", shuffle=False, seed=42,
    )
    true_labels = eval_gen.classes
    class_labels = list(eval_gen.class_indices.keys())

    results = {}

    # Primary model
    model = load_primary_model()
    if model is not None:
        preds = model.predict(eval_gen, verbose=0).flatten()
        pred_cls = (preds > 0.5).astype(int)
        acc = accuracy_score(true_labels, pred_cls)
        prec = precision_score(true_labels, pred_cls, zero_division=0)
        rec = recall_score(true_labels, pred_cls, zero_division=0)
        f1 = f1_score(true_labels, pred_cls, zero_division=0)
        cm = confusion_matrix(true_labels, pred_cls)
        fpr, tpr, _ = roc_curve(true_labels, preds)
        auc_val = auc(fpr, tpr)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        results["primary"] = {
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "specificity": spec, "auc": auc_val,
            "cm": cm, "fpr": fpr, "tpr": tpr,
        }

    # Ensemble models
    ensemble_models = load_ensemble_models()
    model_accs = {}
    if ensemble_models:
        total_w = sum(w for _, _, w in ensemble_models)
        ens_preds_sum = None
        for name, m, weight in ensemble_models:
            eval_gen.reset()
            p = m.predict(eval_gen, verbose=0).flatten()
            p_cls = (p > 0.5).astype(int)
            a = accuracy_score(true_labels, p_cls)
            model_accs[name] = a * 100
            if ens_preds_sum is None:
                ens_preds_sum = p * weight
            else:
                ens_preds_sum += p * weight
        ens_preds = ens_preds_sum / total_w
        ens_cls = (ens_preds > 0.5).astype(int)
        ens_acc = accuracy_score(true_labels, ens_cls)
        model_accs["Ensemble"] = ens_acc * 100
        results["ensemble_accs"] = model_accs

    results["class_labels"] = class_labels
    return results

# ── Custom CSS ────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .main .block-container { max-width: 1100px; padding-top: 2rem; }
    
    .hero-title {
        text-align: center;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 2.4rem; font-weight: 800; margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center; color: #94a3b8; font-size: 1.05rem;
        margin-bottom: 2rem; font-weight: 400;
    }
    
    .result-card {
        border-radius: 16px; padding: 1.8rem 2rem; text-align: center;
        margin: 1rem 0; backdrop-filter: blur(12px);
    }
    .result-tumor {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(239,68,68,0.05));
        border: 1px solid rgba(239,68,68,0.3);
    }
    .result-no-tumor {
        background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(34,197,94,0.05));
        border: 1px solid rgba(34,197,94,0.3);
    }
    .result-label { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-conf { font-size: 1.1rem; font-weight: 400; color: #94a3b8; }
    .tumor-color { color: #f87171; }
    .safe-color { color: #4ade80; }
    
    .metric-box {
        background: rgba(30,41,59,0.7); border: 1px solid #334155;
        border-radius: 12px; padding: 1.2rem; text-align: center;
    }
    .metric-val { font-size: 1.5rem; font-weight: 700; color: #60a5fa; }
    .metric-lbl { font-size: 0.8rem; color: #94a3b8; margin-top: 0.2rem; }

    .section-hdr {
        font-size: 1.25rem; font-weight: 700; color: #e2e8f0;
        border-left: 4px solid #3b82f6; padding-left: 12px;
        margin: 2rem 0 1rem 0;
    }
    
    .img-caption {
        text-align: center; color: #94a3b8; font-size: 0.82rem;
        margin-top: 0.3rem; font-weight: 500;
    }

    .sidebar .sidebar-content { background-color: #0f172a; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white; border: none; border-radius: 10px;
        padding: 0.65rem 2.5rem; font-weight: 600; font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        box-shadow: 0 8px 25px rgba(37,99,235,0.35);
        transform: translateY(-1px);
    }
    
    div[data-testid="stFileUploader"] {
        border: 2px dashed #334155 !important; border-radius: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 Project Info")
        st.markdown("""
        **Brain Tumor Detection** using Hybrid CNN  
        with Ensemble Learning & Grad-CAM Explainability.
        """)
        st.divider()

        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        | Model | Weight |
        |-------|--------|
        | VGG16 (Hybrid) | 0.40 |
        | MobileNetV2 | 0.30 |
        | ResNet50 | 0.30 |
        """)
        st.divider()

        st.markdown("### 📊 Key Features")
        st.markdown("""
        - ✅ Transfer Learning  
        - ✅ Focal Loss (class imbalance)  
        - ✅ Grad-CAM Explainability  
        - ✅ Weighted Soft-Voting Ensemble  
        - ✅ Clinical-Grade Metrics  
        """)
        st.divider()

        st.markdown("### ⚙️ Technical Details")
        st.markdown("""
        - **Input**: 224×224 MRI (JPG/PNG)  
        - **Output**: Binary (Tumor / No Tumor)  
        - **Threshold**: 0.55  
        - **Framework**: TensorFlow / Keras  
        """)
        st.divider()
        st.caption("Final Year Major Project — B.Tech CSE")

# ── Sample images loader ─────────────────────────────────────────────────
def get_sample_images():
    samples = {"tumor": [], "no_tumor": []}
    yes_dir = os.path.join(DATASET_PATH, "yes")
    no_dir = os.path.join(DATASET_PATH, "no")
    if os.path.exists(yes_dir):
        files = sorted([f for f in os.listdir(yes_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:3]
        samples["tumor"] = [os.path.join(yes_dir, f) for f in files]
    if os.path.exists(no_dir):
        files = sorted([f for f in os.listdir(no_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])[:3]
        samples["no_tumor"] = [os.path.join(no_dir, f) for f in files]
    return samples

# ── Main App ──────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Brain Tumor Detection — AI System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    render_sidebar()

    # Hero
    st.markdown('<div class="hero-title">🧠 AI-Based Brain Tumor Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Hybrid CNN · Ensemble Learning · Grad-CAM Explainability</div>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading AI models..."):
        primary_model = load_primary_model()
        ensemble_models = load_ensemble_models()

    if primary_model is None:
        st.error(f"⚠️ Primary model not found at `{MODEL_PATH}`. Train the model first.")
        st.stop()

    # ── Image Input ───────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">📤 Upload MRI Scan</div>', unsafe_allow_html=True)

    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload a brain MRI image",
            type=["jpg", "jpeg", "png"],
            help="Supported: JPG, JPEG, PNG — max 10 MB",
        )

    with col_sample:
        st.markdown("**Or try a sample image:**")
        samples = get_sample_images()
        scol1, scol2 = st.columns(2)
        with scol1:
            if samples["tumor"] and st.button("🔴 Tumor Sample", use_container_width=True):
                st.session_state["sample_path"] = samples["tumor"][0]
        with scol2:
            if samples["no_tumor"] and st.button("🟢 No-Tumor Sample", use_container_width=True):
                st.session_state["sample_path"] = samples["no_tumor"][0]

    # Determine which image to use
    pil_image = None
    if uploaded is not None:
        pil_image = Image.open(uploaded).convert("RGB")
        st.session_state.pop("sample_path", None)
    elif "sample_path" in st.session_state:
        path = st.session_state["sample_path"]
        if os.path.exists(path):
            pil_image = Image.open(path).convert("RGB")

    if pil_image is None:
        st.info("👆 Upload an MRI image or select a sample to begin analysis.")
        st.stop()

    # Preview
    col_prev, col_btn = st.columns([2, 1])
    with col_prev:
        st.image(pil_image, caption="Uploaded MRI Scan", width=280)
    with col_btn:
        st.markdown("<br><br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍  Analyze Scan", use_container_width=True)

    if not predict_clicked and "last_result" not in st.session_state:
        st.stop()

    # ── Run Prediction ────────────────────────────────────────────────────
    if predict_clicked:
        img_array = preprocess_image(pil_image)
        original_bgr = pil_to_bgr(pil_image)

        progress = st.progress(0, text="Preparing image...")
        time.sleep(0.3)
        progress.progress(20, text="Running primary model...")

        label, confidence, is_tumor = predict_single(primary_model, img_array)
        progress.progress(40, text="Generating Grad-CAM...")

        heatmap_norm, heatmap_gray, overlay = generate_gradcam(primary_model, img_array, original_bgr)
        progress.progress(60, text="Running ensemble models...")

        ens_pred, ens_individual = None, None
        if ensemble_models:
            ens_pred, ens_individual = predict_ensemble(ensemble_models, img_array)
        progress.progress(100, text="Analysis complete!")
        time.sleep(0.3)
        progress.empty()

        st.session_state["last_result"] = {
            "label": label, "confidence": confidence, "is_tumor": is_tumor,
            "heatmap_gray": heatmap_gray, "heatmap_norm": heatmap_norm,
            "overlay": overlay, "original_bgr": original_bgr,
            "ens_pred": ens_pred, "ens_individual": ens_individual,
        }

    # ── Display Results ───────────────────────────────────────────────────
    r = st.session_state.get("last_result")
    if r is None:
        st.stop()

    st.markdown('<div class="section-hdr">📋 Prediction Result</div>', unsafe_allow_html=True)

    css_class = "result-tumor" if r["is_tumor"] else "result-no-tumor"
    color_class = "tumor-color" if r["is_tumor"] else "safe-color"
    icon = "⚠️" if r["is_tumor"] else "✅"

    st.markdown(f"""
    <div class="result-card {css_class}">
        <div class="result-label {color_class}">{icon} {r['label']}</div>
        <div class="result-conf">Confidence: {r['confidence']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(r["confidence"]), text=f"Confidence: {r['confidence']:.1f}%")

    # ── Grad-CAM Visualization ────────────────────────────────────────────
    st.markdown('<div class="section-hdr">🎨 Grad-CAM Visualization</div>', unsafe_allow_html=True)
    st.caption("Gradient-weighted Class Activation Mapping — highlights regions the model focuses on")

    if r["heatmap_gray"] is not None:
        g1, g2, g3 = st.columns(3)
        with g1:
            orig_rgb = cv2.cvtColor(r["original_bgr"], cv2.COLOR_BGR2RGB)
            st.image(orig_rgb, caption="Original MRI", use_container_width=True)
        with g2:
            fig_hm, ax_hm = plt.subplots(figsize=(3, 3))
            fig_hm.patch.set_facecolor("#1e293b")
            ax_hm.imshow(r["heatmap_gray"], cmap="jet")
            ax_hm.axis("off")
            plt.tight_layout(pad=0)
            st.pyplot(fig_hm, use_container_width=True)
            st.markdown('<div class="img-caption">Grad-CAM Heatmap</div>', unsafe_allow_html=True)
            plt.close(fig_hm)
        with g3:
            overlay_rgb = cv2.cvtColor(r["overlay"], cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Overlay (Model Focus)", use_container_width=True)
    else:
        st.warning("Grad-CAM could not be generated for this image.")

    # ── Ensemble Predictions ──────────────────────────────────────────────
    if r["ens_individual"]:
        st.markdown('<div class="section-hdr">🤖 Ensemble Model Predictions</div>', unsafe_allow_html=True)
        st.caption("Weighted soft-voting across VGG16, MobileNetV2, and ResNet50")

        mcols = st.columns(len(r["ens_individual"]) + 1)
        for i, (name, val) in enumerate(r["ens_individual"].items()):
            with mcols[i]:
                lbl = "Tumor" if val > 0.5 else "No Tumor"
                clr = "#f87171" if val > 0.5 else "#4ade80"
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-val" style="color:{clr}">{val:.3f}</div>
                    <div class="metric-lbl">{name}<br>{lbl}</div>
                </div>
                """, unsafe_allow_html=True)
        with mcols[-1]:
            ens_lbl = "Tumor" if r["ens_pred"] > 0.5 else "No Tumor"
            ens_clr = "#f87171" if r["ens_pred"] > 0.5 else "#4ade80"
            st.markdown(f"""
            <div class="metric-box" style="border-color:#f59e0b;">
                <div class="metric-val" style="color:{ens_clr}">{r['ens_pred']:.3f}</div>
                <div class="metric-lbl">ENSEMBLE<br>{ens_lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        fig_chart = ensemble_bar_chart(r["ens_individual"], r["ens_pred"])
        st.pyplot(fig_chart, use_container_width=True)
        plt.close(fig_chart)

    # ── Model Performance & Charts ─────────────────────────────────────────
    st.markdown('<div class="section-hdr">📊 Model Performance & Charts</div>', unsafe_allow_html=True)

    if st.button("📈  Run Full Model Evaluation", use_container_width=False):
        with st.spinner("Evaluating models on validation set... This may take a minute."):
            eval_results = run_model_evaluation()
        st.session_state["eval_results"] = eval_results

    ev = st.session_state.get("eval_results")
    if ev:
        # Metrics cards row
        if "primary" in ev:
            pm = ev["primary"]
            st.markdown("**Primary Model (VGG16 Hybrid) — Validation Metrics**")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            metric_items = [
                (m1, "Accuracy", pm["accuracy"]),
                (m2, "Precision", pm["precision"]),
                (m3, "Recall", pm["recall"]),
                (m4, "F1-Score", pm["f1"]),
                (m5, "Specificity", pm["specificity"]),
                (m6, "AUC-ROC", pm["auc"]),
            ]
            for col, lbl, val in metric_items:
                clr = "#4ade80" if val >= 0.8 else ("#fbbf24" if val >= 0.6 else "#f87171")
                with col:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-val" style="color:{clr};font-size:1.3rem;">{val*100:.1f}%</div>
                        <div class="metric-lbl">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Confusion Matrix + ROC curve
            st.markdown("")
            ch1, ch2 = st.columns(2)
            with ch1:
                fig_cm = confusion_matrix_chart(pm["cm"], ev["class_labels"])
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            with ch2:
                fig_roc = roc_curve_chart(pm["fpr"], pm["tpr"], pm["auc"])
                st.pyplot(fig_roc, use_container_width=True)
                plt.close(fig_roc)

        # Ensemble accuracy comparison bar chart
        if "ensemble_accs" in ev:
            st.markdown("")
            st.markdown("**Ensemble Accuracy Comparison**")
            ea = ev["ensemble_accs"]
            fig_acc = accuracy_comparison_chart(list(ea.keys()), list(ea.values()))
            st.pyplot(fig_acc, use_container_width=True)
            plt.close(fig_acc)
    else:
        st.caption("Click the button above to evaluate models and see accuracy charts, confusion matrix, and ROC curve.")

    # ── Model Architecture Info ───────────────────────────────────────────
    st.markdown('<div class="section-hdr">🏗️ Model Architecture</div>', unsafe_allow_html=True)

    a1, a2, a3 = st.columns(3)
    arch_data = [
        ("VGG16 (Primary)", "15.2M params", "Fine-grained texture features", "0.40"),
        ("MobileNetV2", "3.8M params", "Efficient spatial features", "0.30"),
        ("ResNet50", "25.9M params", "Deep hierarchical features", "0.30"),
    ]
    for col, (name, params, desc, w) in zip([a1, a2, a3], arch_data):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-val" style="font-size:1.1rem;">{name}</div>
                <div class="metric-lbl" style="margin-top:0.5rem;">
                    {params} · Weight: {w}<br>
                    <span style="color:#64748b;font-size:0.75rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align:center;color:#475569;font-size:0.8rem;padding:1rem 0;">
        Brain Tumor Detection using Hybrid CNN with Ensemble Learning<br>
        Final Year Major Project — B.Tech CSE
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
