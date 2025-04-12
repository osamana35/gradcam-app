import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import streamlit as st
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from matplotlib.patches import Patch

# إعداد صفحة Streamlit
st.set_page_config(layout="wide", page_title="🧠 X-Ray Diagnosis with Grad-CAM")
st.title("🔍 Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.caption("AI-powered model interpretation with heatmaps and class probabilities.")

# تحميل النموذج
model_path = "mobilenet_model_clahe_aug.h5"
if not os.path.exists(model_path):
    st.error(f"🚫 Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy')
last_conv_layer_name = 'Conv_1'

# نموذج Grad-CAM
grad_model = Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("📤 Upload a Chest X-Ray Image", type=["jpg", "png", "jpeg"])
if uploaded_file is None:
    st.info("👆 Please upload an image to start the analysis.")
    st.stop()

# تجهيز الصورة
file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
img = cv2.imdecode(file_bytes, 1)
img = cv2.resize(img, (224, 224))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_tensor = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

# تنفيذ Grad-CAM
with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    class_idx = tf.argmax(predictions[0])
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_outputs)[0]
pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()

# تجهيز التراكب
heatmap_resized = cv2.resize(heatmap, (224, 224))
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img_rgb, 0.6, heatmap_colored, 0.4, 0)

# عرض النتائج
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
predicted_class_name = class_names[int(class_idx)]
confidence = float(predictions[0][class_idx]) * 100

st.markdown("""
### 🧠 Prediction Summary
""")

col_pred, col_chart = st.columns([1, 2])
with col_pred:
    st.metric(label="Predicted Class", value=predicted_class_name)
    st.metric(label="Confidence", value=f"{confidence:.2f}%")

    # تفسير طبي مبسط
    explanation_dict = {
        'Bacterial Pneumonia': "Bacterial pneumonia often shows patchy or consolidated opacities. The AI model has detected features resembling bacterial infection.",
        'Normal': "The X-ray does not show signs typical of pneumonia. The lungs appear clear and well-aerated.",
        'Viral Pneumonia': "Viral pneumonia may present as ground-glass opacities. The model has identified patterns consistent with viral infections."
    }
    st.markdown(f"#### 🩺 Medical Insight\n> {explanation_dict[predicted_class_name]}")

with col_chart:
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color=['red', 'green', 'blue'])
    ax.set_ylabel("Confidence")
    ax.set_ylim([0, 1])
    ax.set_title("Prediction Probabilities")
    for i, v in enumerate(predictions[0]):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    st.pyplot(fig)

st.markdown("""
### 🖼️ Grad-CAM Heatmap
""")

col1, col2 = st.columns(2)
col1.image(img_rgb, caption="🩻 Original X-Ray", use_container_width=True)
col2.image(superimposed_img, caption="🔥 Grad-CAM Overlay", use_container_width=True)

# توضيح معاني الألوان
st.markdown("""
<hr>
### 🎨 Grad-CAM Attention Legend
- <span style='color:red'><strong>Red</strong></span>: High Attention
- <span style='color:orange'><strong>Yellow</strong></span>: Moderate Attention
- <span style='color:blue'><strong>Blue</strong></span>: Low Attention
<hr>
""", unsafe_allow_html=True)

# نصيحة تعليمية إضافية
st.info("💡 Tip: Grad-CAM highlights regions in the X-ray that contributed most to the AI's decision. This can support radiologists in making a second opinion.")

st.success("✅ Visualization Complete")


