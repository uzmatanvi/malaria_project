# ==========================================
# 🧫 Malaria Cell Classification App
# ==========================================

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import time

# ------------------------------------------
# 1️⃣ Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Malaria Cell Classification",
    page_icon="🧫",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center;'>🧫 Malaria Cell Classification</h1>
    <p style='text-align: center; font-size:18px;'>
    Upload a microscopic blood smear image to detect malaria parasites using Deep Learning.
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ------------------------------------------
# 2️⃣ Load Model (Cached)
# ------------------------------------------

@st.cache_resource
def load_model():
    model_path = "best_malaria_model.h5"

    if not os.path.exists(model_path):
        st.error("❌ Model file not found in repository.")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


model = load_model()

if model is None:
    st.stop()

# ------------------------------------------
# 3️⃣ File Upload
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a blood smear image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"]
)

# ------------------------------------------
# 4️⃣ Prediction Section
# ------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.divider()

    # Resize image based on model input
    input_shape = model.input_shape[1:3]
    image_resized = image.resize(input_shape)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Measure processing time
    start_time = time.time()
    prediction = model.predict(img_array)
    end_time = time.time()

    processing_time = end_time - start_time
    prob = prediction[0][0]

    # --------------------------------------
    # 5️⃣ Determine Result
    # --------------------------------------

    if prob > 0.5:
        result = "Uninfected"
        confidence = prob * 100
        box_color = "#ddffdd"
        text_color = "green"
        icon = "✅"
    else:
        result = "Parasitized"
        confidence = (1 - prob) * 100
        box_color = "#ffdddd"
        text_color = "red"
        icon = "🚨"

    # --------------------------------------
    # 6️⃣ Styled Diagnostic Result Card
    # --------------------------------------

    st.markdown(
        f"""
        <div style="padding:25px;
                    border-radius:15px;
                    background-color:{box_color};
                    text-align:center;">
            <h2 style="color:{text_color};">{icon} {result}</h2>
            <h3>Confidence Level: {confidence:.2f}%</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # --------------------------------------
    # 7️⃣ Probability Breakdown
    # --------------------------------------

    st.subheader("Probability Breakdown")

    st.write(f"🦠 Parasitized: {(1 - prob) * 100:.2f}%")
    st.write(f"🩸 Uninfected: {prob * 100:.2f}%")

    st.divider()

    # --------------------------------------
    # 8️⃣ Analysis Details
    # --------------------------------------

    st.subheader("Analysis Details")

    st.write(f"📏 Image Dimensions: {image.size}")
    st.write(f"⏱ Processing Time: {processing_time:.2f} seconds")
    st.write(f"📊 Model Input Size: {input_shape}")

    st.divider()

    # --------------------------------------
    # 9️⃣ Medical Disclaimer
    # --------------------------------------

    st.warning(
        "⚠️ This system provides AI-assisted predictions only. "
        "Results must be verified by a qualified healthcare professional. "
        "This tool is intended for screening and educational purposes only."
    )
