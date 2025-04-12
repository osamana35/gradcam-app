import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from matplotlib.patches import Patch

# إعداد صفحة Streamlit
st.set_page_config(layout="centered")
st.title("Grad-CAM Visualization for Chest X-Ray")

# تحميل النموذج
model_path = "mobilenet_model_clahe_aug.h5"
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

model = load_model(model_path)
model.compile(optimizer='adam', loss='categorical_crossentropy')
last_conv_layer_name = 'Conv_1'

# نموذج Grad-CAM
grad_model = Model(
    [model.inputs],
    [model.get_layer(last_conv_layer_name).output, model.output]
)

# اختيار صورة عشوائية
base_path = "Test"
if not os.path.exists(base_path):
    st.error(f"Image folder not found: {base_path}")
    st.stop()

class_names = sorted(os.listdir(base_path))
if not class_names:
    st.error("No classes found in the image directory.")
    st.stop()

random_class = random.choice(class_names)
class_path = os.path.join(base_path, random_class)
random_img_name = random.choice(os.listdir(class_path))
img_path = os.path.join(class_path, random_img_name)

# تجهيز الصورة
img = cv2.imread(img_path)
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
st.subheader("Predicted Class")
st.markdown(f"**True Class:** {random_class}<br>**Image:** {random_img_name}", unsafe_allow_html=True)

col1, col2 = st.columns(2)
col1.image(img_rgb, caption="Original X-Ray", use_column_width=True)
col2.image(superimposed_img, caption="Grad-CAM Overlay", use_column_width=True)

# توضيح معاني الألوان
st.markdown("""
<hr>
<h4>Grad-CAM Attention Legend</h4>
<ul>
    <li><span style='color:red'>Red</span>: High Attention</li>
    <li><span style='color:orange'>Yellow</span>: Moderate Attention</li>
    <li><span style='color:blue'>Blue</span>: Low Attention</li>
</ul>
<hr>
""", unsafe_allow_html=True)

st.success("Visualization Complete")

