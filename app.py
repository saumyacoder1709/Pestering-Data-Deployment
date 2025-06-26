import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

# Load model and feature extractor
model_path = "vit-cropdisease"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_path)
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()

st.title("Crop Disease Prediction")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    inputs = feature_extractor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]

    st.success(f"🌿 Predicted Disease: **{predicted_label}**")
