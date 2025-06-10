import streamlit as st
import torch
from PIL import Image
import os
import tempfile
import numpy as np

# Load YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'yolov5s', source='local')  # or 'yolov5m', etc.
    return model

model = load_model()

st.title("YOLOv5 Object Detection")
st.write("Upload an image and see what objects are detected!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img_path = tmp_file.name
        image.save(img_path)

    # Inference
    results = model(img_path)

    # Show results
    st.image(np.squeeze(results.render()), caption="Detected Image", use_column_width=True)
    st.write("Detection Results:")
    st.text(results.pandas().xyxy[0])  # Show detection details
