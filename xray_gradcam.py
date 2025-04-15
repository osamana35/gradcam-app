import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from matplotlib.patches import Patch
from grad_cam import generate_gradcam  # تأكد أن الملف grad_cam.py موجود بنفس المجلد

# تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"  # تأكد أنه رابط مباشر لملف .h5 الصحيح

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# تحميل النموذج
model = load_model(model_path, compile=False)
last_conv_layer_name = 'block5_conv3'

# واجهة Streamlit
st.set_page_config(layout="wide", page_title="X-Ray Diagnosis with Grad-CAM")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.caption("Model interpretation with heatmaps and class probabilities.")

# رفع صورة
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])
if uploaded_file is None:
    st.stop()

# تجهيز الصورة
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

# Grad-CAM
heatmap, superimposed_img, predictions, class_idx = generate_gradcam(
    model, img_tensor, img_rgb, last_conv_layer_name
)

# عرض النتائج
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
try:
    predicted_class_name = class_names[int(class_idx.numpy())]
except:
    st.error("❌ Failed to determine class index.")
    st.stop()

confidence = float(predictions[0][class_idx]) * 100

st.subheader("Prediction Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Class", predicted_class_name)
    st.metric("Confidence", f"{confidence:.2f}%")
with col2:
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color=['red', 'green', 'blue'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Class Probabilities")
    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
    st.pyplot(fig)

st.subheader("Grad-CAM Heatmap")
col1, col2 = st.columns(2)
col1.image(img_rgb, caption="Original X-Ray", use_container_width=True)
col2.image(superimposed_img, caption="Grad-CAM Overlay", use_container_width=True)


