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
    # EXACT name from your GitHub
    model_path = 'best_malaria_model.h5'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found in the repository!")
        # Debug: list files to see what Streamlit sees
        st.write("Files found:", os.listdir("."))
        return None
    
    try:
        # compile=False is safer for loading models on different systems
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"❌ Error loading the .h5 file: {e}")
        return None

model = load_my_model()

# --- 3. File Uploader ---
uploaded_file = st.file_uploader("Upload a blood smear image (JPG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if model is not None:
        with st.spinner("Analyzing..."):
            # --- 4. Preprocessing ---
            # Convert to RGB and resize
            img = image.convert("RGB")
            img = img.resize((224, 224))
            
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
