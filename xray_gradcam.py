
import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import gdown

# تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path, compile=False)
last_conv_layer_name = 'block5_conv3'

# ✅ ترتيب الفئات حسب التدريب (الأصلي داخل النموذج)
class_names = ['Normal', 'Viral Pneumonia', 'Bacterial Pneumonia']

# واجهة Streamlit
st.title("Ray Diagnosis")
st.markdown("Model interpretation with heatmaps and class probabilities.")
st.subheader("Upload a Chest X-ray Image")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # قراءة الصورة وتحضيرها
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(image_rgb)
        img_array = np.expand_dims(img_array / 255.0, axis=0)

        # التنبؤ
        predictions = model.predict(img_array)
        class_idx = int(np.argmax(predictions[0]))
        predicted_class_name = class_names[class_idx]
        st.success(f"✅ **Predicted Class:** {predicted_class_name}")

        # توليد خريطة Grad-CAM
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # مزج الصورة الأصلية مع Grad-CAM
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

        # عرض الصورة النهائية
        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")






