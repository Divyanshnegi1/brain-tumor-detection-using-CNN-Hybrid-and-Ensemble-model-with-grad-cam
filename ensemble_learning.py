

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, MobileNetV2, ResNet50
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

IMG_SIZE = 224
BATCH_SIZE = 16
DATASET_PATH = "brain_tumor_dataset"
ENSEMBLE_DIR = "ensemble_models"

def focal_loss(alpha=0.6, gamma=2.5):
    """Focal loss to handle class imbalance in medical imaging"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


def _compile_model(model, learning_rate=1e-4):
    """Compile model with binary crossentropy for stable training"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_vgg16_model():
    """Hybrid CNN Model 1: VGG16 + Custom Classifier"""
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="vgg16_input")
    base = VGG16(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid", name="vgg16_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HybridCNN_VGG16")
    return _compile_model(model)


def build_mobilenet_model():
    """Hybrid CNN Model 2: MobileNetV2 + Custom Classifier"""
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="mobilenet_input")
    base = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid", name="mobilenet_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HybridCNN_MobileNetV2")
    return _compile_model(model)


def build_resnet_model():
    """Hybrid CNN Model 3: ResNet50 + Custom Classifier"""
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="resnet_input")
    base = ResNet50(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation="sigmoid", name="resnet_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="HybridCNN_ResNet50")
    return _compile_model(model)


def create_generators():
    """Create train / validation generators with augmentation"""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

    train_gen = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="training",
        shuffle=True,
        seed=42,
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    return train_gen, val_gen


def get_class_weights():
    """Calculate class weights for balanced training"""
    yes_path = os.path.join(DATASET_PATH, "yes")
    no_path = os.path.join(DATASET_PATH, "no")
    yes_count = len([f for f in os.listdir(yes_path) if os.path.isfile(os.path.join(yes_path, f))])
    no_count = len([f for f in os.listdir(no_path) if os.path.isfile(os.path.join(no_path, f))])
    total = yes_count + no_count
    w0 = (1 / no_count) * (total / 2.0)
    w1 = (1 / yes_count) * (total / 2.0)
    return {0: w0, 1: w1}


class EnsembleClassifier:
    """
    Weighted Soft-Voting Ensemble Classifier.
    Combines predictions from multiple trained CNN architectures
    using weighted averaging for improved accuracy and robustness.
    """

    def __init__(self):
        self.models = []
        self.histories = {}

    def add_model(self, name, model, weight=1.0):
        self.models.append((name, model, weight))
        print(f"  [+] Added {name}  (weight={weight:.2f}, params={model.count_params():,})")

    def train_all(self, train_gen, val_gen, class_weights, epochs=15):
        """
        Two-phase training for every model in the ensemble:
          Phase 1: Train classifier head with frozen base (warm-up)
          Phase 2: Fine-tune last layers of base with lower LR
        """
        print("\n" + "="*60)
        print("  ENSEMBLE TRAINING  -  {} models (2-phase)".format(len(self.models)))
        print("="*60)

        phase1_epochs = max(epochs // 2, 5)
        phase2_epochs = max(epochs - phase1_epochs, 10)

        for i, (name, model, _) in enumerate(self.models):
            print(f"\n--- [{i+1}/{len(self.models)}] Training {name} ---")

            # ── PHASE 1: Train head only (base frozen) ──
            print(f"\n  Phase 1: Training classifier head ({phase1_epochs} epochs)...")
            callbacks_p1 = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
            ]
            train_gen.reset()
            val_gen.reset()
            history1 = model.fit(
                train_gen,
                epochs=phase1_epochs,
                validation_data=val_gen,
                callbacks=callbacks_p1,
                class_weight=class_weights,
                verbose=1,
            )

            # ── PHASE 2: Fine-tune base model last layers ──
            print(f"\n  Phase 2: Fine-tuning base layers ({phase2_epochs} epochs)...")
            base_model = None
            for layer in model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 5:
                    base_model = layer
                    break
            if base_model is None:
                for layer in model.layers:
                    if 'vgg' in layer.name.lower() or 'mobilenet' in layer.name.lower() or 'resnet' in layer.name.lower():
                        base_model = layer
                        break

            if base_model is not None:
                base_model.trainable = True
                n_layers = len(base_model.layers)
                freeze_until = int(n_layers * 0.7)
                for layer in base_model.layers[:freeze_until]:
                    layer.trainable = False
                print(f"    Unfroze {n_layers - freeze_until}/{n_layers} layers of {base_model.name}")
            else:
                for layer in model.layers:
                    layer.trainable = True
                print("    Unfroze all model layers for fine-tuning")

            _compile_model(model, learning_rate=1e-5)

            callbacks_p2 = [
                EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
            ]
            train_gen.reset()
            val_gen.reset()
            history2 = model.fit(
                train_gen,
                epochs=phase2_epochs,
                validation_data=val_gen,
                callbacks=callbacks_p2,
                class_weight=class_weights,
                verbose=1,
            )

            combined_hist = {}
            for key in history1.history:
                combined_hist[key] = history1.history[key] + history2.history[key]

            class CombinedHistory:
                def __init__(self, h):
                    self.history = h
            self.histories[name] = CombinedHistory(combined_hist)

            os.makedirs(ENSEMBLE_DIR, exist_ok=True)
            save_path = os.path.join(ENSEMBLE_DIR, f"{name}.h5")
            model.save(save_path)
            print(f"  Saved -> {save_path}")

        print("\n" + "="*60)
        print("  ENSEMBLE TRAINING COMPLETE")
        print("="*60)

    def predict(self, x, verbose=True):
        """
        Weighted soft-voting prediction.

        Args:
            x: preprocessed image array (1, 224, 224, 3), scaled [0-1]

        Returns:
            ensemble_pred (float), individual_preds (dict)
        """
        individual = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, model, weight in self.models:
            pred = float(model.predict(x, verbose=0)[0][0])
            individual[name] = pred
            weighted_sum += pred * weight
            total_weight += weight

        ensemble_pred = weighted_sum / total_weight

        if verbose:
            print("\n  ENSEMBLE SOFT-VOTING PREDICTIONS:")
            print("  " + "-"*45)
            for name, pred in individual.items():
                w = [w for n, _, w in self.models if n == name][0]
                label = "Tumor" if pred > 0.5 else "No Tumor"
                print(f"  {name:<25} {pred:.4f}  (w={w:.2f})  -> {label}")
            print("  " + "-"*45)
            label = "Tumor Detected" if ensemble_pred > 0.5 else "No Tumor Detected"
            conf = ensemble_pred * 100 if ensemble_pred > 0.5 else (1 - ensemble_pred) * 100
            print(f"  ENSEMBLE RESULT:       {ensemble_pred:.4f}  -> {label} ({conf:.1f}% confidence)")

        return ensemble_pred, individual

    def predict_image(self, image_path):
        """Load an image from disk and run ensemble prediction"""
        if not os.path.exists(image_path):
            print(f"  Image not found: {image_path}")
            return None, None
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        return self.predict(img_array)

    def evaluate(self, val_gen):
        """Evaluate each model and the ensemble on the validation set"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        val_gen.reset()
        true_labels = val_gen.classes

        print("\n" + "="*60)
        print("  ENSEMBLE vs INDIVIDUAL MODEL EVALUATION")
        print("="*60)

        all_preds = {}
        for name, model, _ in self.models:
            val_gen.reset()
            preds = model.predict(val_gen, verbose=0).flatten()
            all_preds[name] = preds

            pred_classes = (preds > 0.5).astype(int)
            acc = accuracy_score(true_labels, pred_classes)
            prec = precision_score(true_labels, pred_classes, zero_division=0)
            rec = recall_score(true_labels, pred_classes, zero_division=0)
            f1 = f1_score(true_labels, pred_classes, zero_division=0)
            print(f"\n  {name}:")
            print(f"    Accuracy  = {acc*100:.2f}%")
            print(f"    Precision = {prec*100:.2f}%")
            print(f"    Recall    = {rec*100:.2f}%")
            print(f"    F1-Score  = {f1*100:.2f}%")

        total_w = sum(w for _, _, w in self.models)
        ensemble_preds = sum(
            all_preds[n] * w for n, _, w in self.models
        ) / total_w
        ens_classes = (ensemble_preds > 0.5).astype(int)
        acc = accuracy_score(true_labels, ens_classes)
        prec = precision_score(true_labels, ens_classes, zero_division=0)
        rec = recall_score(true_labels, ens_classes, zero_division=0)
        f1 = f1_score(true_labels, ens_classes, zero_division=0)

        print(f"\n  >>> ENSEMBLE (Soft Voting) <<<")
        print(f"    Accuracy  = {acc*100:.2f}%")
        print(f"    Precision = {prec*100:.2f}%")
        print(f"    Recall    = {rec*100:.2f}%")
        print(f"    F1-Score  = {f1*100:.2f}%")
        print("="*60)

        return {
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1_score": f1,
        }

    def plot_training_history(self):
        """Plot accuracy and loss curves for all models"""
        if not self.histories:
            print("  No training history available.")
            return

        n = len(self.histories)
        fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
        if n == 1:
            axes = [axes]

        for i, (name, hist) in enumerate(self.histories.items()):
            axes[i][0].plot(hist.history["accuracy"], label="Train")
            axes[i][0].plot(hist.history["val_accuracy"], label="Validation")
            axes[i][0].set_title(f"{name} - Accuracy")
            axes[i][0].set_xlabel("Epoch")
            axes[i][0].set_ylabel("Accuracy")
            axes[i][0].legend()
            axes[i][0].grid(True, alpha=0.3)

            axes[i][1].plot(hist.history["loss"], label="Train")
            axes[i][1].plot(hist.history["val_loss"], label="Validation")
            axes[i][1].set_title(f"{name} - Loss")
            axes[i][1].set_xlabel("Epoch")
            axes[i][1].set_ylabel("Loss")
            axes[i][1].legend()
            axes[i][1].grid(True, alpha=0.3)

        plt.suptitle("Ensemble Training History", fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_comparison(self, val_gen):
        """Bar chart comparing individual models vs ensemble"""
        from sklearn.metrics import accuracy_score

        true_labels = val_gen.classes
        names = []
        accuracies = []

        total_w = sum(w for _, _, w in self.models)
        ensemble_preds_sum = None

        for name, model, weight in self.models:
            val_gen.reset()
            preds = model.predict(val_gen, verbose=0).flatten()
            pred_classes = (preds > 0.5).astype(int)
            acc = accuracy_score(true_labels, pred_classes)
            names.append(name)
            accuracies.append(acc * 100)

            if ensemble_preds_sum is None:
                ensemble_preds_sum = preds * weight
            else:
                ensemble_preds_sum += preds * weight

        ens_preds = ensemble_preds_sum / total_w
        ens_classes = (ens_preds > 0.5).astype(int)
        ens_acc = accuracy_score(true_labels, ens_classes)
        names.append("ENSEMBLE")
        accuracies.append(ens_acc * 100)

        colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, accuracies, color=colors[: len(names)], edgecolor="black")
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{acc:.1f}%", ha="center", fontweight="bold")
        plt.title("Individual Models vs Ensemble Accuracy", fontsize=14, fontweight="bold")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 105)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


def build_and_train_ensemble(epochs=15):
    """
    Build 3 hybrid CNN models, train them, and return the ensemble.
    Models: VGG16 (w=0.4), MobileNetV2 (w=0.3), ResNet50 (w=0.3)
    Uses two-phase training: head warm-up + base fine-tuning.
    """
    print("\n" + "="*60)
    print("  BUILDING MULTI-ARCHITECTURE ENSEMBLE")
    print("="*60)

    ensemble = EnsembleClassifier()

    print("\n  Building individual hybrid CNN models...")
    ensemble.add_model("HybridCNN_VGG16",       build_vgg16_model(),     weight=0.4)
    ensemble.add_model("HybridCNN_MobileNetV2",  build_mobilenet_model(), weight=0.3)
    ensemble.add_model("HybridCNN_ResNet50",     build_resnet_model(),    weight=0.3)

    print("\n  Creating data generators...")
    train_gen, val_gen = create_generators()
    class_weights = get_class_weights()

    ensemble.train_all(train_gen, val_gen, class_weights, epochs=epochs)
    return ensemble, val_gen


def load_ensemble():
    """Load previously trained ensemble models from disk."""
    model_configs = [
        ("HybridCNN_VGG16",       0.4),
        ("HybridCNN_MobileNetV2", 0.3),
        ("HybridCNN_ResNet50",    0.3),
    ]
    custom_objects = {"focal_loss_fixed": focal_loss()}

    ensemble = EnsembleClassifier()
    loaded = 0

    for name, weight in model_configs:
        path = os.path.join(ENSEMBLE_DIR, f"{name}.h5")
        if os.path.exists(path):
            model = load_model(path, custom_objects=custom_objects)
            ensemble.add_model(name, model, weight)
            loaded += 1
        else:
            print(f"  [!] Not found: {path}")

    if loaded == 0:
        print("  No ensemble models found. Train first with build_and_train_ensemble().")
        return None

    print(f"\n  Loaded {loaded}/{len(model_configs)} ensemble models")
    return ensemble


def demonstrate_ensemble_concept():
    """Print theoretical explanation of ensemble learning"""
    print("\n" + "="*60)
    print("  ENSEMBLE LEARNING - THEORETICAL FOUNDATION")
    print("="*60)
    print("""
    1. DIVERSITY PRINCIPLE:
       - VGG16 captures fine-grained texture features
       - MobileNetV2 captures efficient spatial features
       - ResNet50 captures deep hierarchical features

    2. SOFT VOTING (WEIGHTED AVERAGING):
       P_ensemble(x) = sum_i( w_i * P_i(x) ) / sum_i( w_i )

    3. WHY ENSEMBLE IMPROVES ACCURACY:
       - Reduces variance: individual errors cancel out
       - Reduces overfitting: no single model dominates
       - Improves calibration: probabilities are more reliable

    4. OUR ENSEMBLE CONFIGURATION:
       +---------------------+--------+---------+
       | Model               | Weight | Params  |
       +---------------------+--------+---------+
       | HybridCNN_VGG16     |  0.40  |  15.2M  |
       | HybridCNN_MobileNet |  0.30  |   3.8M  |
       | HybridCNN_ResNet50  |  0.30  |  25.9M  |
       +---------------------+--------+---------+
       | TOTAL ENSEMBLE      |  1.00  |  44.9M  |
       +---------------------+--------+---------+
    """)


if __name__ == "__main__":
    print("="*60)
    print("  BRAIN TUMOR DETECTION - ENSEMBLE LEARNING MODULE")
    print("="*60)

    demonstrate_ensemble_concept()

    ensemble = load_ensemble()
    if ensemble is None:
        print("\n  No pre-trained ensemble found. Training now...")
        ensemble, val_gen = build_and_train_ensemble(epochs=5)
        ensemble.plot_training_history()
        ensemble.evaluate(val_gen)
        ensemble.plot_comparison(val_gen)
    else:
        print("\n  Pre-trained ensemble loaded successfully!")

    test_img = "brain_tumor_dataset/yes/Y1.jpg"
    if os.path.exists(test_img):
        print(f"\n  Demo prediction on: {test_img}")
        ensemble.predict_image(test_img)

    print("\n  Ensemble module ready.")