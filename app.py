import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# -------------------------------
# 1️⃣ Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Malaria Detector",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Malaria Cell Classification")
st.write("Upload a blood smear image to check for malaria parasites.")

# -------------------------------
# 2️⃣ Load Model (Keras 3 Compatible)
# -------------------------------
@st.cache_resource
def load_my_model():
    model_path = "best_malaria_model.h5"

    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False  # Important for compatibility
        )
        return model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None


model = load_my_model()

# -------------------------------
# 3️⃣ File Upload Section
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a blood smear image (JPG/PNG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        # Open and display image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=300)

        if model is not None:
            with st.spinner("Analyzing image..."):

                # -------------------------------
                # 4️⃣ Preprocessing
                # -------------------------------
                img = image.resize((64, 64))  # Ensure matches training size
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # -------------------------------
                # 5️⃣ Prediction
                # -------------------------------
                prediction = model.predict(img_array)

                prob = float(prediction[0][0])

                if prob > 0.5:
                    label = "Uninfected"
                    confidence = prob * 100
                    st.success(f"🟢 Result: {label}")
                else:
                    label = "Parasitized"
                    confidence = (1 - prob) * 100
                    st.error(f"🔴 Result: {label}")

                st.info(f"Confidence: {confidence:.2f}%")

        else:
            st.warning("Model failed to load. Please check deployment logs.")

    except Exception as e:
        st.error(f"Image processing failed: {e}")

else:
    st.info("Waiting for image upload...")
