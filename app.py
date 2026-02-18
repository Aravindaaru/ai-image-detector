import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ai_image_detector.h5", compile=False)

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

st.title("🔍 AI Image Detector")
st.write("Upload an image to check if it is REAL or AI-Generated.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence = round(confidence * 100, 2)

    if prediction > 0.5:
        st.error(f"🤖 FAKE (AI-Generated)\nConfidence: {confidence}%")
    else:
        st.success(f"✅ REAL Image\nConfidence: {confidence}%")
