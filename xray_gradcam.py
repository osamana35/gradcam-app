import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from matplotlib.patches import Patch
from grad_cam import generate_gradcam
import gdown

# إعداد الصفحة
st.set_page_config(layout="wide", page_title="Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.title("Grad-CAM Visualization for Chest X-Ray Diagnosis")
st.caption("Model interpretation with heatmaps and class probabilities.")

# تحميل النموذج من Google Drive
model_path = "vgg16_best_852acc.h5"
file_id = "1--SxjRX5Sxh8NKcrV5ztx2WZiSQwBEGi"
if not os.path.exists(model_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, model_path, quiet=False)

# تعريف Focal Loss (نفس المستخدمة أثناء التدريب)
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

# تحميل النموذج مع تضمين الدالة المخصصة
model = load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss()})
model.compile(optimizer='adam', loss='categorical_crossentropy')
last_conv_layer_name = 'block5_conv3'

# رفع صورة من المستخدم
uploaded_file = st.file_uploader("Upload a Chest X-Ray Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # معالجة الصورة
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = np.expand_dims(img_rgb.astype("float32") / 255.0, axis=0)

    # التنبؤ + Grad-CAM
    try:
        heatmap, superimposed_img, predictions, class_idx = generate_gradcam(
            model, img_tensor, img_rgb, last_conv_layer_name
        )

        # استخراج الاسم والثقة
        class_names = ['Bacterial Pneumonia', 'Normal', 'Viral Pneumonia']
        class_idx = int(np.argmax(predictions, axis=1).item())
        predicted_class_name = class_names[class_idx]
        confidence = float(predictions[0][class_idx]) * 100

        # عرض النتيجة
        st.subheader("Prediction Summary")
        st.metric("Predicted Class", predicted_class_name)
        st.metric("Confidence", f"{confidence:.2f}%")

        st.subheader("Prediction Probabilities")
        fig, ax = plt.subplots()
        ax.bar(class_names, predictions[0], color=['red', 'green', 'blue'])
        ax.set_ylim([0, 1])
        for i, v in enumerate(predictions[0]):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center')
        st.pyplot(fig)

        st.subheader("Grad-CAM Heatmap")
        col1, col2 = st.columns(2)
        col1.image(img_rgb, caption="Original X-Ray", use_column_width=True)
        col2.image(superimposed_img, caption="Grad-CAM Overlay", use_column_width=True)

    except Exception as e:
        st.error(f"Failed to process the image.\n\n{str(e)}")



