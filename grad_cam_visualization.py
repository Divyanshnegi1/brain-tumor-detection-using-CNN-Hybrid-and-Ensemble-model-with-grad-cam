

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array



OPTIMAL_LAYERS = {
    "vgg16":       "block5_conv3",
    "mobilenetv2": "block_16_project_BN",   
    "mobilenet":   "block_16_project_BN",
    "resnet50":    "conv5_block3_out",       
    "resnet":      "conv5_block3_out",
}


_LAYER_SEARCH_PATTERNS = [
    "block_16_project_BN",  
    "out_relu",              
    "conv5_block3_out",     
    "conv5_block3_3_conv",   
    "block5_conv3",          
    "block_16_expand_relu", 
    "conv5_block3_2_relu",   
    "block5_conv2",          
    "block_14_project_BN",   
]


def _get_best_conv_layer(model, preferred_name=None):
    """
    Find the best convolutional layer for Grad-CAM.

    Strategy:
      1. Try the explicitly requested layer name
      2. Try architecture-specific optimal layers (matched by scanning layer names)
      3. Try known fallback patterns
      4. Fall back to the last Conv2D layer in the model
    """
    all_layer_names = {l.name for l in model.layers}

   
    if preferred_name and preferred_name in all_layer_names:
        return model.get_layer(preferred_name)

    
    arch_detected = None
    for l_name in all_layer_names:
        if "block5_conv3" in l_name:
            arch_detected = "vgg16"
            break
        elif any("block_16" in n for n in all_layer_names) and \
             any("out_relu" in n or "block_16_project_BN" in n for n in all_layer_names):
            arch_detected = "mobilenetv2"
            break
        elif "conv5_block3" in l_name:
            arch_detected = "resnet50"
            break

    if arch_detected and OPTIMAL_LAYERS.get(arch_detected) in all_layer_names:
        return model.get_layer(OPTIMAL_LAYERS[arch_detected])

   
    model_name_lower = (model.name or "").lower()
    for arch_key, layer_name in OPTIMAL_LAYERS.items():
        if arch_key in model_name_lower and layer_name in all_layer_names:
            return model.get_layer(layer_name)

    
    for pattern in _LAYER_SEARCH_PATTERNS:
        if pattern in all_layer_names:
            return model.get_layer(pattern)

    
    spatial_layers = []
    for l in model.layers:
        try:
            out_shape = l.output_shape
            if isinstance(out_shape, list):
                out_shape = out_shape[0]
            if len(out_shape) == 4:  # (batch, h, w, channels)
                spatial_layers.append(l)
        except (AttributeError, TypeError):
            pass

    if spatial_layers:
        return spatial_layers[-1]

    raise ValueError("No convolutional layer found in model for Grad-CAM")



def _create_brain_mask(image_bgr, margin=10):
    """
    Create a binary mask of the brain region to suppress
    skull edges and background in the Grad-CAM heatmap.

    Uses OTSU thresholding with morphological cleanup, convex hull fitting,
    and edge erosion to ensure no skull-boundary pixels leak into the heatmap.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


    blurred = cv2.GaussianBlur(gray, (25, 25), 0)


    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_large, iterations=2)


    if margin > 0:
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1)
        )
        binary = cv2.erode(binary, erode_kernel, iterations=1)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [hull], -1, 255, -1)
    else:
        mask = binary


    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    mask = (mask / 255.0).astype(np.float32)

    return mask



def _compute_gradcam(model, img_array, target_layer):
    """
    Compute Grad-CAM heatmap using standard gradient weighting.

    This is more robust and consistent across all architectures compared
    to Grad-CAM++ (which requires second-order gradients that can be
    unstable for some models like MobileNetV2).

    Returns: raw_heatmap (2D numpy array), prediction_value (float)
    """
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_output, prediction = grad_model(img_tensor)
        # Use the predicted class score
        score = prediction[:, 0]


    grads = tape.gradient(score, conv_output)

    if grads is None:
        
        heatmap = tf.reduce_mean(conv_output[0], axis=-1).numpy()
        heatmap = np.maximum(heatmap, 0)
        return heatmap, float(prediction[0][0])

    
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))

    
    heatmap = tf.reduce_sum(weights * conv_output[0], axis=-1).numpy()

    
    heatmap = np.maximum(heatmap, 0)

    return heatmap, float(prediction[0][0])


def _compute_gradcam_plus_plus(model, img_array, target_layer):
    """
    Compute Grad-CAM++ heatmap for improved localization.

    Grad-CAM++ uses second-order gradients to weight the feature maps,
    producing more focused activation regions. Falls back to standard
    Grad-CAM if second-order gradients are unavailable.

    Returns: raw_heatmap (2D numpy array), prediction_value (float)
    """
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(conv_output_var := grad_model(img_tensor)[0])
        conv_output, prediction = grad_model(img_tensor)
        score = prediction[:, 0]

    try:
        
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_output, prediction = grad_model(img_tensor)
                score = prediction[:, 0]
            grads_first = tape1.gradient(score, conv_output)

        if grads_first is None:
            return _compute_gradcam(model, img_array, target_layer)

        grads_second = tape2.gradient(grads_first, conv_output)

        if grads_second is not None:
            relu_second = tf.nn.relu(grads_second)
            denom = 2.0 * relu_second + \
                    tf.reduce_sum(conv_output * relu_second,
                                  axis=(1, 2), keepdims=True) + 1e-8
            alpha = relu_second / denom
            weights = tf.reduce_sum(alpha * tf.nn.relu(grads_first),
                                    axis=(1, 2), keepdims=True)
        else:
            weights = tf.reduce_mean(grads_first, axis=(1, 2), keepdims=True)

        heatmap = tf.reduce_sum(weights * conv_output, axis=-1)[0].numpy()
        heatmap = np.maximum(heatmap, 0)
        return heatmap, float(prediction[0][0])

    except Exception:
        
        return _compute_gradcam(model, img_array, target_layer)



def _postprocess_heatmap(heatmap, image_size=224, brain_mask=None,
                          noise_percentile=65,
                          sharpening_power=2.0,
                          min_activation_area=50):
    """
    Post-process the raw Grad-CAM heatmap for clean medical visualization.

    Pipeline:
      1. Resize to image dimensions with cubic interpolation
      2. Activation thresholding (remove low-intensity noise)
      3. Apply brain-region mask (suppress background/skull)
      4. Connected-component filtering (keep only large activation blobs)
      5. Power-law sharpening to concentrate activations into peaks
      6. Edge-preserving bilateral smoothing
      7. Gentle Gaussian blur for final smoothness
      8. Percentile-based normalization for strong contrast
      9. Morphological closing to fill small gaps
    """
    
    heatmap = cv2.resize(heatmap, (image_size, image_size),
                          interpolation=cv2.INTER_CUBIC)

    
    if heatmap.max() > 0:
        positive_vals = heatmap[heatmap > 0]
        if len(positive_vals) > 0:
            threshold = np.percentile(positive_vals, noise_percentile)
            heatmap[heatmap < threshold] = 0

    
    if brain_mask is not None:
        if brain_mask.shape[:2] != (image_size, image_size):
            brain_mask = cv2.resize(brain_mask, (image_size, image_size))
        heatmap = heatmap * brain_mask

    
    if heatmap.max() > 0:
        heatmap_binary = (heatmap > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            heatmap_binary, connectivity=8
        )
        
        clean_mask = np.zeros_like(heatmap_binary)
        for label_idx in range(1, num_labels):  
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_activation_area:
                clean_mask[labels == label_idx] = 1
        heatmap = heatmap * clean_mask.astype(np.float32)

    
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max


    heatmap = np.power(heatmap, sharpening_power)

    
    heatmap_uint8_temp = np.uint8(255 * heatmap)
    heatmap_smooth = cv2.bilateralFilter(heatmap_uint8_temp, d=11,
                                          sigmaColor=75, sigmaSpace=75)
    heatmap = heatmap_smooth.astype(np.float32) / 255.0

   
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

    
    if heatmap.max() > 0:
        positive_vals = heatmap[heatmap > 0]
        if len(positive_vals) > 0:
            p_high = np.percentile(positive_vals, 95)
            if p_high > 0:
                heatmap = np.clip(heatmap / p_high, 0, 1)

    
    heatmap_binary = (heatmap > 0.1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    heatmap_binary = cv2.morphologyEx(heatmap_binary, cv2.MORPH_CLOSE,
                                       kernel, iterations=2)
    heatmap = heatmap * heatmap_binary.astype(np.float32)

    
    hm_max = heatmap.max()
    if hm_max > 0:
        heatmap = heatmap / hm_max

    return heatmap



def _create_overlay(original_bgr, heatmap_normalized, alpha=0.55,
                     low_activation_cutoff=0.05):
    """
    Create a clean medical-grade overlay of the Grad-CAM on the MRI.

    Uses JET colormap with per-pixel alpha blending. Low-activation
    areas are fully transparent, showing the raw MRI underneath.
    Only strong activation regions (tumor area) get the heatmap overlay.
    """
    heatmap_uint8 = np.uint8(255 * heatmap_normalized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)


    activation_mask = heatmap_normalized.copy()
    activation_mask[activation_mask < low_activation_cutoff] = 0
    activation_mask = activation_mask[..., np.newaxis]


    overlay = original_bgr.astype(np.float32) * (1 - activation_mask * alpha) + \
              heatmap_color.astype(np.float32) * (activation_mask * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    return overlay, heatmap_uint8, heatmap_color



def grad_cam(model, image_path, layer_name="block5_conv3"):
    """
    Generate professional medical-grade Grad-CAM visualization.

    Produces a 3-panel plot: Original MRI | Grad-CAM Heatmap | Overlay

    Args:
        model:      Trained Keras model
        image_path: Path to input MRI image
        layer_name: Preferred convolutional layer name (auto-detected if not found)
    """
    
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    
    original_bgr = cv2.imread(image_path)
    if original_bgr is None:
        print(f"❌ Could not read image: {image_path}")
        return
    original_bgr = cv2.resize(original_bgr, (224, 224))

    
    try:
        target_layer = _get_best_conv_layer(model, layer_name)
    except ValueError as e:
        print(f"❌ {e}")
        return

    
    try:
        raw_heatmap, prediction_val = _compute_gradcam_plus_plus(
            model, img_array, target_layer
        )
    except Exception as e:
        print(f"⚠️  Grad-CAM++ failed, using standard Grad-CAM: {e}")
        try:
            raw_heatmap, prediction_val = _compute_gradcam(
                model, img_array, target_layer
            )
        except Exception as e2:
            print(f"❌ Grad-CAM computation failed: {e2}")
            return

    
    brain_mask = _create_brain_mask(original_bgr, margin=10)

    
    heatmap = _postprocess_heatmap(
        raw_heatmap,
        image_size=224,
        brain_mask=brain_mask,
        noise_percentile=65,
        sharpening_power=2.0,
        min_activation_area=50
    )

    
    overlay, heatmap_uint8, heatmap_color = _create_overlay(
        original_bgr, heatmap, alpha=0.55, low_activation_cutoff=0.05
    )

    
    if prediction_val > 0.55:
        result_text = f"Tumor Detected ({prediction_val * 100:.1f}%)"
        title_color = "#e74c3c"
    else:
        result_text = f"No Tumor ({(1 - prediction_val) * 100:.1f}%)"
        title_color = "#2ecc71"

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

   
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original MRI Scan", fontsize=13, fontweight='bold',
                       color='white', pad=10)
    axes[0].axis("off")

    
    axes[1].imshow(heatmap_uint8, cmap="jet", vmin=0, vmax=255)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=13, fontweight='bold',
                       color='white', pad=10)
    axes[1].axis("off")

    
    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Tumor Region Highlighted", fontsize=13,
                       fontweight='bold', color='white', pad=10)
    axes[2].axis("off")

    plt.suptitle(
        f"Grad-CAM Visualization — {result_text}\n"
        f"Layer: {target_layer.name}  |  Image: {os.path.basename(image_path)}",
        fontsize=14, fontweight='bold', color='white', y=1.04
    )
    plt.tight_layout()
    plt.show()

    print(f"✅ Grad-CAM visualization generated")
    print(f"   Layer used: {target_layer.name}")
    print(f"   Prediction: {prediction_val:.4f} → {result_text}")
    print(f"   Heatmap range: [{heatmap_uint8.min()}, {heatmap_uint8.max()}]")



def visualize_with_gradcam(model, image_path):
    """Wrapper function to visualize Grad-CAM for an image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    print(f"\n{'=' * 60}")
    print(f" GRAD-CAM VISUALIZATION")
    print(f"{'=' * 60}")
    print(f"Image: {os.path.basename(image_path)}")

    try:
        grad_cam(model, image_path)
        return True
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def predict_and_visualize(model, image_path):
    """Predict class and visualize Grad-CAM"""
    normalized_path = os.path.normpath(image_path)

    if not os.path.exists(normalized_path):
        print(f"Error: Image not found at '{normalized_path}'")
        if 'brain_tumor_dataset' not in normalized_path:
            alt_path = os.path.join('brain_tumor_dataset', image_path)
            alt_path = os.path.normpath(alt_path)
            if os.path.exists(alt_path):
                normalized_path = alt_path
            else:
                print(f"Also tried alternative path: '{alt_path}' but it doesn't exist")
                return
        else:
            return

    try:
        img = load_img(normalized_path, target_size=(224, 224))
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

        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Image: {normalized_path}")
        print(f"Result: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("=" * 50)

        visualize_with_gradcam(model, normalized_path)

    except Exception as e:
        print(f"Error in prediction and visualization: {e}")
        import traceback
        traceback.print_exc()



def generate_gradcam_for_model(model, img_array, original_bgr, layer_name=None):
    """
    Generate Grad-CAM heatmap + overlay for a single model.
    Used by ensemble_prediction.py and streamlit_app.py for per-model Grad-CAM.

    Args:
        model:        Trained Keras model
        img_array:    Preprocessed image array (1, 224, 224, 3)
        original_bgr: Original image in BGR format (224, 224, 3)
        layer_name:   Optional preferred layer name

    Returns:
        heatmap_normalized (float32 0-1), heatmap_uint8, overlay, prediction_val, layer_name_used
    """
    try:
        target_layer = _get_best_conv_layer(model, layer_name)
    except ValueError:
        return None, None, None, None, None


    try:
        raw_heatmap, prediction_val = _compute_gradcam_plus_plus(
            model, img_array, target_layer
        )
    except Exception:
        try:
            raw_heatmap, prediction_val = _compute_gradcam(
                model, img_array, target_layer
            )
        except Exception:
            return None, None, None, None, None

    brain_mask = _create_brain_mask(original_bgr, margin=10)

    heatmap = _postprocess_heatmap(
        raw_heatmap,
        image_size=224,
        brain_mask=brain_mask,
        noise_percentile=65,
        sharpening_power=2.0,
        min_activation_area=50
    )

    overlay, heatmap_uint8, _ = _create_overlay(
        original_bgr, heatmap, alpha=0.55, low_activation_cutoff=0.05
    )

    return heatmap, heatmap_uint8, overlay, prediction_val, target_layer.name