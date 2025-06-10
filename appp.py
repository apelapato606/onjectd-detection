import streamlit as st
import torch
from PIL import Image
import os
import tempfile
import numpy as np

# Load YOLOv5 model from Ultralytics GitHub
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # No need for 'local'
    return model

model = load_model()

st.title("🦾 YOLOv5 Object Detection")
st.write("Upload an image and see what objects are detected!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        img_path = tmp_file.name
        image.save(img_path)

    # Run inference
    results = model(img_path)

    # Show result image
    st.image(np.squeeze(results.render()), caption="Detected Image", use_column_width=True)

    # Show raw prediction results
    st.write("Detection Results:")
    st.dataframe(results.pandas().xyxy[0])  # Bounding boxes and labels
