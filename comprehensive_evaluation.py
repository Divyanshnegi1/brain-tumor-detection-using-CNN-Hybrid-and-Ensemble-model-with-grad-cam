
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import cv2
import warnings
warnings.filterwarnings("ignore")

from grad_cam_visualization import predict_and_visualize
import tensorflow.keras.backend as K

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

def create_evaluation_generators():
    """Create data generators specifically for evaluation"""
    eval_datagen = ImageDataGenerator(rescale=1./255)
    
    eval_generator = eval_datagen.flow_from_directory(
        "brain_tumor_dataset",
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        shuffle=False,
        seed=42
    )
    
    return eval_generator

def evaluate_model_comprehensive(model_path="brain_tumor_model_balanced.h5"):
    """Perform comprehensive model evaluation"""
    print("="*60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    
    print("Loading trained model...")
    try:
        model = load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss(alpha=0.6, gamma=2.5)}
        )
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\nCreating evaluation data generators...")
    eval_gen = create_evaluation_generators()
    
    print("\nGenerating predictions...")
    predictions = model.predict(eval_gen, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    true_classes = eval_gen.classes
    
    class_labels = list(eval_gen.class_indices.keys())
    
    print(f"\nPrediction Statistics:")
    print(f"- Total samples evaluated: {len(true_classes)}")
    print(f"- True positives (tumor): {sum(true_classes)}")
    print(f"- True negatives (no tumor): {len(true_classes) - sum(true_classes)}")
    print(f"- Predicted positives (tumor): {sum(predicted_classes)}")
    print(f"- Predicted negatives (no tumor): {len(predicted_classes) - sum(predicted_classes)}")
    
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(
        true_classes, 
        predicted_classes, 
        target_names=class_labels, 
        digits=4
    )
    print(report)
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    fpr, tpr, _ = roc_curve(true_classes, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.show()
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n" + "="*60)
    print("DETAILED METRICS")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
    print(f"F1-Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")
    print(f"AUC-ROC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    
    print(f"\nSENSITIVITY/SPECIFICITY BREAKDOWN:")
    print(f"Sensitivity (True Positive Rate): {recall:.4f}")
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Misclassification Rate: {(fp+fn)/(tp+tn+fp+fn):.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1_score,
        'auc_roc': roc_auc,
        'confusion_matrix': cm
    }

def test_additional_samples(model_path="brain_tumor_model_balanced.h5"):
    """Test on additional sample images with Grad-CAM visualization"""
    print("\n" + "="*60)
    print("TESTING ON ADDITIONAL SAMPLES WITH GRAD-CAM")
    print("="*60)
    
    try:
        model = load_model(
            model_path, 
            custom_objects={'focal_loss_fixed': focal_loss(alpha=0.6, gamma=2.5)}
        )
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    test_images = [
        "brain_tumor_dataset/yes/Y1.jpg",
        "brain_tumor_dataset/yes/Y2.jpg", 
        "brain_tumor_dataset/yes/Y10.jpg",
        "brain_tumor_dataset/no/6 no.jpg",
        "brain_tumor_dataset/no/1 no.jpeg",
        "brain_tumor_dataset/no/10 no.jpg"
    ]
    
    successful_tests = 0
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n--- Testing: {os.path.basename(img_path)} ---")
            predict_and_visualize(model, img_path)
            successful_tests += 1
        else:
            print(f"\n⚠️  Image not found: {img_path}")
    
    print(f"\n✅ Successfully tested {successful_tests}/{len(test_images)} images")

def model_comparison_analysis():
    """Provide analysis of the model architecture and approach"""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*60)
    print("""
ARCHITECTURE CHOICES:
- Base Model: VGG16 (pre-trained on ImageNet)
- Transfer Learning: Feature extraction with frozen base
- Custom Top Layers:
  * Global Average Pooling
  * Batch Normalization
  * Dense(512) -> Dropout(0.3) -> BatchNorm
  * Dense(256) -> Dropout(0.2) -> BatchNorm  
  * Dense(128) -> Dropout(0.2) -> BatchNorm
  * Dense(64)  -> Dropout(0.1)
  * Output(Dense(1, sigmoid))
    """)

def main():
    """Main evaluation function"""
    print("🧠 BRAIN TUMOR DETECTION - COMPREHENSIVE EVALUATION")
    print("="*60)
    
    metrics = evaluate_model_comprehensive()
    
    if metrics:
        print(f"\n🎯 OVERALL PERFORMANCE SUMMARY:")
        print(f"   Overall Score: {((metrics['accuracy'] + metrics['auc_roc'])/2)*100:.1f}%")
        print(f"   Best Metric (AUC-ROC): {metrics['auc_roc']*100:.1f}%")
        print(f"   Clinical Metrics: Sensitivity={metrics['recall']*100:.1f}%, Specificity={metrics['specificity']*100:.1f}%")
    
    test_additional_samples()
    model_comparison_analysis()
    
    print("\n" + "="*60)
    print("PROJECT ENHANCEMENT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()