import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🧠",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_image_detector.h5", compile=False)

model = load_model()

# Custom styling
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title section
st.title("🧠 AI vs Real Image Detector")
st.markdown("Upload an image to check whether it is **AI Generated** or **Real**.")

st.divider()

# Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.divider()

    if prediction > 0.5:
        st.success("✅ REAL Image Detected")
        st.progress(float(prediction))
    else:
        st.error("🤖 AI Generated Image Detected")
        st.progress(float(1 - prediction))
