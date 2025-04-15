import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown

# تحميل النموذج
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"

if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# اسم الطبقة الأخيرة للـ Grad-CAM
last_conv_layer_name = 'block5_conv3'

# ✅ ترتيب الفئات حسب التدريب
class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']

# واجهة Streamlit
st.title("Ray Diagnosis")
st.markdown("Model interpretation with heatmaps and class probabilities.")
st.subheader("Upload a Chest X-ray Image")

uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # تحميل الصورة وتحويلها إلى رمادية
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))

        # تطبيق CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

        # تحويل الصورة إلى 3 قنوات
        image_rgb = cv2.merge([image, image, image])
        img_norm = image_rgb.astype('float32') / 255.0
        img_array = np.expand_dims(img_norm, axis=0)

        # التنبؤ
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds[0]))
        predicted_class = class_names[class_idx]
        st.success(f"✅ **Predicted Class:** {predicted_class}")

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
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)
        st.image(overlay, caption="Grad-CAM Heatmap", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")





