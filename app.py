import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("🦟 Malaria Detection App")
st.header("Upload Blood Smear Image")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_malaria_model.h5")
# File uploader
uploaded_file = st.file_uploader("Choose image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Dummy prediction (replace with model later)
    if st.button("🔬 Detect Malaria"):
        model = load_model()
        img = image.resize((224, 224))  # Your model input size
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)[0][0]
        label = "Parasitized" if pred > 0.5 else "Uninfected"
        conf = pred if label == "Parasitized" else 1-pred
        
        st.success(f"**Result:** {label}")
        st.info(f"**Confidence:** {conf:.1%}")

