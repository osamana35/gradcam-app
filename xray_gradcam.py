import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
import gdown

st.set_page_config(page_title="X-Ray Grad-CAM", layout="centered")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")

# ✅ ترتيب الفئات حسب training
class_names = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']

# ✅ تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"
if not os.path.exists(model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# ✅ focal loss
import tensorflow.keras.backend as K
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

get_custom_objects().update({"focal_loss": focal_loss()})
model = load_model(model_path, custom_objects={"focal_loss": focal_loss()})

# ✅ Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ✅ معالجة الصور
def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    img = cv2.merge([img, img, img])
    img = img.astype("float32") / 255.0
    return img

# ✅ واجهة المستخدم
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Failed to load image.")
    else:
        processed = preprocess_image(image)
        input_img = np.expand_dims(processed, axis=0)

        preds = model.predict(input_img)
        class_idx = int(np.argmax(preds[0]))
        class_name = class_names[class_idx]

        st.success(f"**Predicted Class:** {class_name}")

        heatmap = make_gradcam_heatmap(input_img, model, last_conv_layer_name='block5_conv3')
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted((processed * 255).astype("uint8"), 0.6, heatmap, 0.4, 0)

        st.image(superimposed, caption="Grad-CAM Visualization", use_column_width=True)





