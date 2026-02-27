import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# -----------------------------------
# 1️⃣ Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="Malaria Detector",
    page_icon="🔬",
    layout="centered"
)

st.title("🔬 Malaria Cell Classification")
st.write("Upload a blood smear image to detect malaria infection.")

# -----------------------------------
# 2️⃣ Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    model_path = "best_malaria_model.h5"

    if not os.path.exists(model_path):
        st.error("Model file not found.")
        return None

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False   # Prevent optimizer/version issues
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


model = load_model()

# -----------------------------------
# 3️⃣ File Upload
# -----------------------------------
uploaded_file = st.file_uploader(
    "Upload a blood smear image (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(image, caption="Uploaded Image", width=300)

        if model is not None:
            with st.spinner("Analyzing image..."):

                # -----------------------------------
                # 4️⃣ Preprocessing (Match Training Size)
                # -----------------------------------
                img = image.resize((64, 64))  # Change if your training size was different
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # -----------------------------------
                # 5️⃣ Prediction
                # -----------------------------------
                prediction = model.predict(img_array)
                probability = float(prediction[0][0])

                if probability > 0.5:
                    label = "Uninfected"
                    confidence = probability * 100
                    st.success(f"🟢 Result: {label}")
                else:
                    label = "Parasitized"
                    confidence = (1 - probability) * 100
                    st.error(f"🔴 Result: {label}")

                st.info(f"Confidence: {confidence:.2f}%")

        else:
            st.warning("Model failed to load. Please check logs.")

    except Exception as e:
        st.error(f"Image processing failed: {e}")

else:
    st.info("Waiting for image upload...")
