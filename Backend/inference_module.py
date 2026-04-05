import os
import base64
import io

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess


# class labels
class_info = {
    0: "Mild_Demented",
    1: "Moderate_Demented",
    2: "Non_Demented",
    3: "Very_Mild_Demented",
}


# encode an RGB image as a base64 PNG string so it can be sent in JSON response
def _encode_png_base64(rgb_uint8: np.ndarray):
    """Encode an RGB uint8 image to base64 PNG string."""
    img = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# decode raw image bytes to BGR uint8 for OpenCV processing. try OpenCV first, fallback to PIL if needed
def _decode_image_bytes_to_bgr(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil, dtype=np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

# standardize Keras layer outputs to be single tensors
def _ensure_tensor_output(x, name: str):
    if isinstance(x, (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"{name} has {len(x)} outputs; expected a single tensor output")
        return x[0]
    return x

# Grad-CAM implementation for a convolutional layer inside a nested backbone ( used for EfficientNetB2 )
def gradcam_for_backbone(model, img_array, backbone_name: str, conv_layer_name: str, class_index: int, eps: float = 1e-8,):
    x_in = tf.convert_to_tensor(img_array, dtype=tf.float32) # convert input image to tensor 

    # get the backbone and conv layer
    backbone = model.get_layer(backbone_name)
    conv_layer = backbone.get_layer(conv_layer_name)

    # build a model that outputs the conv layer and the final output of the backbone
    backbone_to_conv = Model(
        inputs=backbone.input,
        outputs=[conv_layer.output, backbone.output],
    )

    # rebuild classifier head seperately so class scores can be computed from the backbone feature representation 
    backbone_output = _ensure_tensor_output(backbone.output, f"{backbone_name}.output")
    head_input = tf.keras.Input(shape=backbone_output.shape[1:])
    x = head_input
    start_idx = model.layers.index(backbone) + 1

    # recreate the layers that come after the backbone
    for layer in model.layers[start_idx:]:
        x = layer(x)
    head_only = Model(head_input, x)

    # record operations so gradients can be computed
    with tf.GradientTape() as tape:
        conv_val, features = backbone_to_conv(x_in, training=False)
        conv_val = _ensure_tensor_output(conv_val, f"{backbone_name}.{conv_layer_name}.output")
        features = _ensure_tensor_output(features, f"{backbone_name}.output")
        tape.watch(conv_val)

        #forward pass through classifier head
        pred_val = head_only(features, training=False)
        pred_val = _ensure_tensor_output(pred_val, "model.head.output")
        loss = pred_val[:, int(class_index)]

    # compute gradients of the class score with respect to feature maps
    grads = tape.gradient(loss, conv_val)

    # remove batch dimension and compute channel importance weights 
    conv_val = conv_val[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    # weighted sum across channels gives the class activation map 
    cam = tf.reduce_sum(weights * conv_val, axis=-1)

    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam) + eps)
    return cam.numpy().astype(np.float32)

# Grad-CAM implementation for a convolutional layer directly in the full model graph ( used for VGG16+CBAM )
def gradcam_for_layer(model, img_array, layer_name: str, class_index: int, eps: float = 1e-8):


    x_in = tf.convert_to_tensor(img_array, dtype=tf.float32)
    target_layer = model.get_layer(layer_name)
    target_output = _ensure_tensor_output(target_layer.output, f"{layer_name}.output")
    
    # find the index of the target layer so the model can be spit in two 
    layer_index = model.layers.index(target_layer)
    
    # build a model from input to the chosen target layer
    layer_to_target = tf.keras.Model(model.inputs, target_output)
    
    # build a model from target layer to output 
    head_input = tf.keras.Input(shape=target_output.shape[1:])
    x = head_input
    start_idx = layer_index + 1
    for layer in model.layers[start_idx:]:
        x = layer(x)
    head_model = tf.keras.Model(head_input, x)

    # record operations so gradients can be computed
    with tf.GradientTape() as tape:
        conv_val = layer_to_target(x_in, training=False)
        conv_val = _ensure_tensor_output(conv_val, f"{layer_name}.output")
        tape.watch(conv_val)
        preds = head_model(conv_val, training=False)
        preds = _ensure_tensor_output(preds, "model.head.output")
        loss = preds[:, int(class_index)]

    grads = tape.gradient(loss, conv_val)

    conv_val = conv_val[0]
    grads = grads[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * conv_val, axis=-1)

    cam = tf.maximum(cam, 0.0)
    cam = cam / (tf.reduce_max(cam) + eps)
    return cam.numpy().astype(np.float32)

# overlay a 0-1 CAM onto a BGR image using OpenCV. CAM is resized to match the image size, converted to a heatmap, and blended with the original image
def superimpose01(img_bgr_uint8: np.ndarray, cam01: np.ndarray, alpha: float = 0.7, cmap: int = cv2.COLORMAP_JET):

    # resize heatmap tp match original image size     
    heat = cv2.resize(
        cam01.astype(np.float32),
        (img_bgr_uint8.shape[1], img_bgr_uint8.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    # replace invalid values and clamp to 0-1 range 
    heat = np.clip(np.nan_to_num(heat), 0.0, 1.0)

    # convert heatmap to 0-255 and apply a colour map 
    heat_uint8 = np.uint8(255.0 * heat)
    heat_color = cv2.applyColorMap(heat_uint8, cmap)
    overlay_bgr = cv2.addWeighted(heat_color, alpha, img_bgr_uint8, 1.0 - alpha, 0.0)
    # convert back to rgb for display 
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return overlay_rgb


WVGG, WEFF = 0.5, 0.5  # ensemble weights

VGG_CAM_LAYER_NAME = "add_1"          # CBAM output / final conv feature in VGG 
EFF_BACKBONE_NAME = "efficientnetb2"  # backbone in EfficientNetB2 wrapper 
EFF_CAM_LAYER_NAME = "top_conv"       # conv layer inside EfficientNetB2 backbone


# load trained models from disk
VGG_PATH = os.path.join("export", "models", "vgg_model.keras")
EFF_PATH = os.path.join("export", "models", "eff_model.keras")

# make sure models exist before trying to load them 
if not os.path.exists(VGG_PATH) or not os.path.exists(EFF_PATH):
    raise FileNotFoundError(
        f"Expected trained models at:\n  {VGG_PATH}\n  {EFF_PATH}\n"
)
# load models without needing to compile because they are already trained 
vgg_model = load_model(VGG_PATH, compile=False)
eff_model = load_model(EFF_PATH, compile=False)


# main inference function called by the backend
def predict_one_image(image_bytes: bytes):
    
    # decode uploaded image and create a display version of 260x260 
    img_bgr = _decode_image_bytes_to_bgr(image_bytes)
    disp_bgr = cv2.resize(img_bgr, (260, 260), interpolation=cv2.INTER_AREA).astype(np.uint8)
    disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)

    # VGG input using vgg preproceesing 
    vgg_img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)
    vgg_in = np.expand_dims(vgg_img, axis=0)
    vgg_in = vgg_preprocess(vgg_in)

    # EfficientNet input (using efficientnet specific preprocessing 
    eff_img = cv2.resize(img_bgr, (260, 260), interpolation=cv2.INTER_AREA).astype(np.float32)
    eff_in = np.expand_dims(eff_img, axis=0)
    eff_in = eff_preprocess(eff_in)

    # predict class probabilities from both models 
    pvgg = vgg_model.predict(vgg_in, verbose=0)[0]
    peff = eff_model.predict(eff_in, verbose=0)[0]

    # combine the two probability vectors using simple average 
    ens = WVGG * pvgg + WEFF * peff

    # select class with highest probability 
    pred_idx = int(np.argmax(ens))
    pred_label = class_info[pred_idx]
    conf = float(ens[pred_idx])

    # build a class probability dictionary for frontend response 
    probs = {class_info[i]: float(ens[i]) for i in range(len(ens))}

    # generate Grad-CAM for both models using the predicted class 
    cam_vgg = gradcam_for_layer(
        vgg_model,
        vgg_in,
        VGG_CAM_LAYER_NAME,
        pred_idx,
    )
    cam_eff = gradcam_for_backbone(
        eff_model,
        eff_in,
        EFF_BACKBONE_NAME,
        EFF_CAM_LAYER_NAME,
        pred_idx,
    )

    # resize heatmaps to match the display size (260x260)
    cam_vgg_260 = cv2.resize(cam_vgg, (260, 260), interpolation=cv2.INTER_LINEAR)
    cam_eff_260 = cv2.resize(cam_eff, (260, 260), interpolation=cv2.INTER_LINEAR)

    # overlay both heatmpas on the original image 
    vgg_overlay_rgb = superimpose01(disp_bgr, cam_vgg_260, alpha=0.7)
    eff_overlay_rgb = superimpose01(disp_bgr, cam_eff_260, alpha=0.7)

    # return JSON friendly response with predicted class, confidence, probabilities, and base64 encoded images for display
    return {
        "predicted_class": pred_label,
        "confidence": conf,
        "probabilities": probs,
        "images": {
            "original_png_b64": _encode_png_base64(disp_rgb),
            "vgg_gradcam_png_b64": _encode_png_base64(vgg_overlay_rgb),
            "eff_gradcam_png_b64": _encode_png_base64(eff_overlay_rgb),
        },
    }
