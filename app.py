import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="centered"
)

# -------------------------------
# Custom CSS Styling
# -------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #FFFFFF;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #AAAAAA;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_image_detector.h5", compile=False)

model = load_model()

# -------------------------------
# Header Section
# -------------------------------
st.markdown('<div class="title">🤖 AI vs Real Image Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and let AI decide if it is Real or AI-Generated</div>', unsafe_allow_html=True)

st.divider()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.divider()

    if prediction > 0.5:
        st.markdown(
            '<div class="result-box" style="background-color:#1E5631; color:white;">✅ REAL IMAGE</div>',
            unsafe_allow_html=True
        )
        st.success(f"Confidence: {prediction*100:.2f}%")
    else:
        st.markdown(
            '<div class="result-box" style="background-color:#7B0000; color:white;">🚨 AI GENERATED</div>',
            unsafe_allow_html=True
        )
        st.error(f"Confidence: {(1-prediction)*100:.2f}%")

    st.divider()

# -------------------------------
# Footer
# -------------------------------
st.markdown(
    "<center><small>Built with ❤️ using TensorFlow & Streamlit</small></center>",
    unsafe_allow_html=True
)
