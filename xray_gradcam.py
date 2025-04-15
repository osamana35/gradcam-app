import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import streamlit as st
import gdown
from tensorflow.keras.models import load_model
from matplotlib.patches import Patch
from grad_cam import generate_gradcam

# تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path, compile=False)  # بدون compile لحل مشكلة التحميل
last_conv_layer_name = 'block5_conv3'

# إعداد واجهة Streamlit
st.set_page_config(layout="wide", page_title="Grad-CAM X-Ray Diagnosis")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.caption("Model interpretation with heatmaps and class probabilities.")

# رفع الصورة
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

# تشغيل Grad-CAM
try:
    heatmap, superimposed_img, predictions, class_idx = generate_gradcam(
        model, img_tensor, img_rgb, last_conv_layer_name
    )
    
    # التأكد من أن class_idx رقم صحيح
    if isinstance(class_idx, tf.Tensor):
        class_idx = int(class_idx.numpy())
except Exception as e:
    st.error(f"Failed to process the image.\n\n{str(e)}")
    st.stop()

# تحديد الأصناف
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
try:
    predicted_class_name = class_names[class_idx]
    confidence = float(predictions[0][class_idx]) * 100
except Exception as e:
    st.error("Failed to determine class index.")
    st.stop()

# عرض النتائج
st.markdown("### Prediction Summary")
col_pred, col_chart = st.columns([1, 2])
with col_pred:
    st.metric(label="Predicted Class", value=predicted_class_name)
    st.metric(label="Confidence", value=f"{confidence:.2f}%")
    explanation_dict = {
        'Bacterial Pneumonia': "Bacterial pneumonia often shows patchy or consolidated opacities.",
        'Normal': "The X-ray does not show signs typical of pneumonia. The lungs appear clear.",
        'Viral Pneumonia': "Viral pneumonia may present as ground-glass opacities."
    }
    st.markdown(f"#### Medical Insight\n> {explanation_dict[predicted_class_name]}")

with col_chart:
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color=['red', 'green', 'blue'])
    ax.set_ylim([0, 1])
    ax.set_ylabel("Confidence")
    ax.set_title("Prediction Probabilities")
    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

# عرض Grad-CAM
st.markdown("### Grad-CAM Heatmap")
col1, col2 = st.columns(2)
col1.image(img_rgb, caption="Original X-Ray", use_container_width=True)
col2.image(superimposed_img, caption="Grad-CAM Overlay", use_container_width=True)

st.markdown("""---  
### Grad-CAM Attention Legend  
- Red: High Attention  
- Yellow: Moderate Attention  
- Blue: Low Attention  
---""")



