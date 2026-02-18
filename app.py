import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_image_detector.h5", compile=False)

model = load_model()

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #4A90E2;
    }
    .sub-text {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">AI vs Real Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Upload an image to check whether it is AI-generated or Real.</div>', unsafe_allow_html=True)

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("✅ This image is REAL")
        st.progress(float(prediction))
    else:
        st.error("⚠️ This image is AI GENERATED")
        st.progress(float(1 - prediction))

st.markdown("---")
st.markdown("Made by Aravind 🚀")
