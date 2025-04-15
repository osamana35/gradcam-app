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
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']

# واجهة Streamlit
st.set_page_config(page_title="X-Ray Grad-CAM", layout="centered")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.markdown("Model interpretation with heatmaps and class probabilities.")
st.subheader("Upload a Chest X-Ray Image")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(image_rgb)
        img_array = np.expand_dims(img_array / 255.0, axis=0)

        predictions = model.predict(img_array)

        if predictions is None or len(predictions) == 0 or predictions[0].size != len(class_names):
            st.error("⚠️ Model prediction failed or returned invalid output.")
            st.stop()

        class_idx = int(np.argmax(predictions[0]))
        if class_idx >= len(class_names):
            st.error("⚠️ Prediction index is out of valid range.")
            st.stop()

        predicted_class_name = class_names[class_idx]
        st.success(f"**Predicted Class:** {predicted_class_name}")

        # Grad-CAM
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
        cam = cv2.resize(cam.numpy(), (224, 224))
        cam -= cam.min()
        cam /= cam.max()
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")





