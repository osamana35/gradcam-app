import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from matplotlib.patches import Patch
from grad_cam import generate_gradcam

# focal loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="X-Ray Diagnosis with Grad-CAM")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.caption("Model interpretation with heatmaps and class probabilities")

# تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# تحميل النموذج وتسجيل الدالة المخصصة
model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
model.compile(optimizer='adam', loss='categorical_crossentropy')
last_conv_layer_name = 'block5_conv3'

# رفع صورة
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])
if uploaded_file is None:
    st.info("Please upload an image to start the analysis.")
    st.stop()

# تجهيز الصورة
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

# Grad-CAM
heatmap, superimposed_img, predictions, class_idx = generate_gradcam(model, img_tensor, img_rgb, last_conv_layer_name)

# عرض النتائج
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
predicted_class_name = class_names[int(class_idx)]
confidence = float(predictions[0][class_idx]) * 100

st.markdown("### Prediction Summary")
col_pred, col_chart = st.columns([1, 2])
with col_pred:
    st.metric(label="Predicted Class", value=predicted_class_name)
    st.metric(label="Confidence", value=f"{confidence:.2f}%")
    explanation_dict = {
        'Bacterial Pneumonia': "Bacterial pneumonia often shows patchy or consolidated opacities.",
        'Normal': "The X-ray does not show signs typical of pneumonia.",
        'Viral Pneumonia': "Viral pneumonia may present as ground-glass opacities."
    }
    st.markdown(f"**Medical Insight:**\n\n{explanation_dict[predicted_class_name]}")

with col_chart:
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color=['red', 'green', 'blue'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probabilities")
    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

st.markdown("### Grad-CAM Heatmap")
col1, col2 = st.columns(2)
col1.image(img_rgb, caption="Original X-Ray", use_container_width=True)
col2.image(superimposed_img, caption="Grad-CAM Overlay", use_container_width=True)


