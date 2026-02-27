#App.py ka code :




import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. Page Configuration ---
st.set_page_config(page_title="Malaria Detector", page_icon="🔬")

st.title("🔬 Malaria Cell Classification")
st.write("Upload a cell image to check for malaria parasites.")

# --- 2. Model Loading ---
@st.cache_resource
def load_my_model():
    model_path = 'best_malaria_model.h5'
    if not os.path.exists(model_path):
        st.error(f"File {model_path} not found.")
        return None
    try:
        # We remove compile=False temporarily to see if Keras 3 handles it better
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        # If it still fails, we try the "Legacy" loader
        try:
            return tf.keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
        except:
            st.error(f"❌ Version Mismatch: {e}")
            return None
model = load_my_model()

# --- 3. File Uploader ---
uploaded_file = st.file_uploader("Upload a blood smear image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    
    if model is not None:
        with st.spinner("Analyzing..."):
            # --- 4. Preprocessing ---
            # Convert to RGB and resize
            img = image.convert("RGB")
            img = img.resize((64, 64))
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            
            # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
            img_array = np.expand_dims(img_array, axis=0)

            # --- 5. Prediction ---
            prediction = model.predict(img_array)
            
            # Assuming: 0 = Parasitized, 1 = Uninfected
            # (If your results seem backwards, swap the labels below)
            if prediction[0][0] > 0.5:
                label = "Uninfected"
                confidence = prediction[0][0] * 100
                st.success(f"Result: **{label}**")
            else:
                label = "Parasitized"
                confidence = (1 - prediction[0][0]) * 100
                st.error(f"Result: **{label}**")
            
            st.info(f"Confidence Level: {confidence:.2f}%")
    else:
        st.warning("Model is not loaded. Please check the logs.")

else:
    st.info("Waiting for image upload...")
