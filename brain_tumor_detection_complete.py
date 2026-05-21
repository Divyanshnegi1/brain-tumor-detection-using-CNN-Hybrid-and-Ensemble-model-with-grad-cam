

import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K
import cv2

from grad_cam_visualization import predict_and_visualize, visualize_with_gradcam, grad_cam
from comprehensive_evaluation import evaluate_model_comprehensive, test_additional_samples
from ensemble_learning import (
    EnsembleClassifier, build_and_train_ensemble, load_ensemble,
    demonstrate_ensemble_concept
)
from ensemble_prediction import predict_with_ensemble, ensemble_gradcam_all_models, demonstrate_ensemble_architecture


np.random.seed(42)
tf.random.set_seed(42)


DATASET_PATH = "brain_tumor_dataset"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

def focal_loss(alpha=0.8, gamma=2.0):
    """Focal loss to handle class imbalance"""
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def analyze_dataset():
    """Analyze the dataset distribution"""
    yes_path = os.path.join(DATASET_PATH, "yes")
    no_path = os.path.join(DATASET_PATH, "no")
    
    yes_count = len([f for f in os.listdir(yes_path) if os.path.isfile(os.path.join(yes_path, f))])
    no_count = len([f for f in os.listdir(no_path) if os.path.isfile(os.path.join(no_path, f))])
    
    print(f"Dataset Analysis:")
    print(f"Tumor images (yes): {yes_count}")
    print(f"No-tumor images (no): {no_count}")
    print(f"Total images: {yes_count + no_count}")
    print(f"Imbalance ratio: {yes_count/no_count:.2f}:1 (tumor:no-tumor)")
    
    return yes_count, no_count

def create_balanced_generators():
    """Create data generators with balanced sampling"""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = val_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator

def build_model():

    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=focal_loss(alpha=0.6, gamma=2.5),
        metrics=['accuracy']
    )
    
    return model

def predict_single_image(model, image_path):
    """Predict a single image"""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return
        
    try:
        img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        if prediction > 0.55:
            result = "Tumor Detected"
            confidence = prediction * 100
        else:
            result = "No Tumor Detected"
            confidence = (1 - prediction) * 100
            
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Image: {image_path}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*50)
        
    except Exception as e:
        print(f"Error predicting image: {e}")





def main():
    print("=== BRAIN TUMOR DETECTION USING HYBRID CNN MODEL ===\n")
    print("Project Status: 70% Completed with Ensemble Learning")
    print("="*60)
    
    yes_count, no_count = analyze_dataset()
    
    print("\nCreating balanced data generators...")
    train_gen, val_gen = create_balanced_generators()
    
    print("\nBuilding Hybrid CNN Model (VGG16 + Custom Classifier)...")
    model = build_model()
    print("\n" + "="*60)
    print("HYBRID CNN MODEL ARCHITECTURE")
    print("="*60)
    model.summary()
    
    print("\nTraining model...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    ]
    
    total = yes_count + no_count
    weight_for_0 = (1 / no_count) * (total / 2.0)
    weight_for_1 = (1 / yes_count) * (total / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}
    
    print(f"Class weights: {class_weights}")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    )
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    model.save("brain_tumor_model_balanced.h5")
    print("\nModel saved as 'brain_tumor_model_balanced.h5'")
    
    print("\n=== TESTING ON SAMPLE IMAGES WITH GRAD-CAM VISUALIZATION ===")
    
    tumor_images = [
        "brain_tumor_dataset/yes/Y1.jpg",
        "brain_tumor_dataset/yes/Y2.jpg",
        "brain_tumor_dataset/yes/Y10.jpg"
    ]
    
    for i, tumor_img in enumerate(tumor_images):
        if os.path.exists(tumor_img):
            print(f"\nTesting on tumor image {i+1}: {os.path.basename(tumor_img)}")
            predict_and_visualize(model, tumor_img)
        else:
            print(f"\nTumor image not found: {tumor_img}")
    
    no_tumor_images = [
        "brain_tumor_dataset/no/6 no.jpg",
        "brain_tumor_dataset/no/1 no.jpeg",
        "brain_tumor_dataset/no/10 no.jpg"
    ]
    
    for i, no_tumor_img in enumerate(no_tumor_images):
        if os.path.exists(no_tumor_img):
            print(f"\nTesting on no-tumor image {i+1}: {os.path.basename(no_tumor_img)}")
            predict_and_visualize(model, no_tumor_img)
        else:
            print(f"\nNo-tumor image not found: {no_tumor_img}")
    
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("="*60)
    evaluate_model_comprehensive()
    test_additional_samples()
    
    print("\n" + "="*60)
    print("REAL ENSEMBLE LEARNING (VGG16 + MobileNetV2 + ResNet50)")
    print("="*60)
    demonstrate_ensemble_concept()
    demonstrate_ensemble_architecture()
    
    ensemble = load_ensemble()
    if ensemble is None:
        print("\nTraining ensemble models (3 architectures, 5 epochs each)...")
        ensemble, val_gen = build_and_train_ensemble(epochs=5)
        ensemble.plot_training_history()
        ensemble.evaluate(val_gen)
        ensemble.plot_comparison(val_gen)
    else:
        print("\nPre-trained ensemble loaded successfully!")
        from ensemble_learning import create_generators
        _, val_gen = create_generators()
        ensemble.evaluate(val_gen)
        ensemble.plot_comparison(val_gen)
    
    sample_images = [
        "brain_tumor_dataset/yes/Y1.jpg",
        "brain_tumor_dataset/no/6 no.jpg",
    ]
    for sample_image in sample_images:
        if os.path.exists(sample_image):
            predict_with_ensemble(ensemble, sample_image)
    
    gradcam_image = "brain_tumor_dataset/yes/Y1.jpg"
    if os.path.exists(gradcam_image):
        ensemble_gradcam_all_models(ensemble, gradcam_image)
    
    print("\n" + "="*60)
    print("FUTURE SCOPE AND IMPROVEMENTS")
    print("="*60)
    print("""
PLANNED ENHANCEMENTS:

1. TUMOR SEGMENTATION:
   - Implement U-Net architecture for precise tumor boundary detection
   - Add segmentation mask generation and overlay
   - Include tumor size and location analysis

2. WEB DEPLOYMENT:
   - Flask/Streamlit web application for medical professionals
   - User-friendly interface with upload and visualization
   - Database integration for patient records

3. ADVANCED FEATURES:
   - Integration with DICOM medical imaging standards
   - Real-time processing capabilities
   - Multi-class classification (tumor types)
   - Model compression for edge deployment

4. CLINICAL VALIDATION:
   - Performance benchmarking against radiologists
   - Cross-dataset validation for generalizability
   - Prospective clinical study design
""")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
This project successfully demonstrates a Hybrid CNN + Ensemble approach for
brain tumor detection using medical MRI images.

KEY ACHIEVEMENTS:
  [1] Hybrid CNN Model (Functional API)
  [2] Real Multi-Architecture Ensemble
  [3] Grad-CAM Explainability
  [4] Comprehensive Evaluation
  [5] Academic-Ready Presentation
""")
    
    print("=== PROJECT EXECUTION COMPLETE ===")
    print("Hybrid CNN Model with Real Ensemble Learning - Ready for Viva")
    print("100% Complete - Final Year Major Project")

if __name__ == "__main__":
    main()