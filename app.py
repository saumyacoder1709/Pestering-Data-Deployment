import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import streamlit as st

# Load model and processor
model = ViTForImageClassification.from_pretrained("vit-cropdisease")
processor = ViTImageProcessor.from_pretrained("vit-cropdisease")

# Streamlit UI
st.title("Crop Disease Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_class = model.config.id2label[probs.argmax().item()]

    st.markdown(f"### 🌾 Predicted Class: **{predicted_class}**")
