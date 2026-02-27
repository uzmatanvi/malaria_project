import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- 1. Page Config & Header ---
st.set_page_config(page_title="Malaria Cell Detector", layout="centered")
st.title("🔬 Malaria Cell Detection")
st.write("Upload a blood smear image to detect if the cell is Parasitized or Uninfected.")

# --- 2. Load the Model ---
# Using @st.cache_resource so the model only loads once, making the app fast
@st.cache_resource
def load_my_model():
    model_path = 'your_model_filename.h5' # <--- MAKE SURE THIS MATCHES YOUR FILENAME
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model file '{model_path}' not found in the repository!")
        return None

model = load_my_model()

# --- 3. Image Upload ---
uploaded_file = st.file_uploader("Choose a cell image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("🔄 Classifying...")

    # --- 4. Preprocessing (The Fix) ---
    # 1. Convert to RGB (handles PNG transparency)
    img = image.convert('RGB')
    # 2. Resize to the 224x224 your model expects
    img = img.resize((224, 224))
    # 3. Convert to numpy array
    img_array = np.array(img)
    # 4. Normalize pixel values (0-255 -> 0-1)
    img_array = img_array / 255.0
    # 5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # --- 5. Prediction ---
    try:
        prediction = model.predict(img_array)
        
        # Logic for Binary Classification (1 output neuron)
        # Usually: 0 = Parasitized, 1 = Uninfected (Check your training labels!)
        if prediction[0][0] > 0.5:
            result = "Uninfected"
            confidence = prediction[0][0] * 100
            st.success(f"Prediction: {result} ({confidence:.2f}%)")
        else:
            result = "Parasitized"
            confidence = (1 - prediction[0][0]) * 100
            st.error(f"Prediction: {result} ({confidence:.2f}%)")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload an image to start.")
