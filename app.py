import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

id2label = {
    0: 'anthracnose_Cashew',
    1: 'bacterial blight_Cassava',
    2: 'brown spot_Cassava',
    3: 'fall armyworm_Maize',
    4: 'grasshoper_Maize',
    5: 'green mite_Cassava',
    6: 'gumosis_Cashew',
    7: 'healthy_Cashew',
    8: 'healthy_Cassava',
    9: 'healthy_Maize',
    10: 'healthy_Tomato',
    11: 'leaf beetle_Maize',
    12: 'leaf blight_Maize',
    13: 'leaf blight_Tomato',
    14: 'leaf curl_Tomato',
    15: 'leaf miner_Cashew',
    16: 'leaf spot_Maize',
    17: 'mosaic_Cassava',
    18: 'red rust_Cashew',
    19: 'septoria leaf spot_Tomato',
    20: 'streak virus_Maize',
    21: 'verticulium wilt_Tomato'
}

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
        predicted_label = id2label[predicted_class_idx]

    st.success(f"🌿 Predicted Disease: **{predicted_label}**")

