import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json
import os
import gdown

output_path = "alzheimer_resnet_model_best.keras"

# Download if not already downloaded
if not os.path.exists(output_path):
    url= "https://drive.google.com/uc?id=12r0a25ri-8St76F4mdotrLXcwbg2SBUE"
    gdown.download(url, output_path, quiet=False)

# Load model
model = load_model(output_path)

# Load class indices
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Map class indices
class_names = {v: k for k, v in class_indices.items()}

# Preprocess image
def preprocess(image):
    image = image.resize((224, 224))  # Adjust size as per model input
    image = image.convert('RGB')      # Ensure 3 channels (RGB), NOT grayscale
    image = np.array(image)
    image = image / 255.0             # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension --> (1, 224, 224, 3)
    return image


# Prediction function
def predict(image):
    processed = preprocess(image)
    predictions = model.predict(processed)[0]

    pred_class = np.argmax(predictions)
    confidence = np.max(predictions)
    
    return class_names[pred_class], confidence, predictions



# Streamlit UI
st.set_page_config(page_title="Alzheimer's Disease Prediction", layout="centered")

st.title("Alzheimer's Disease Detection")
st.subheader("Upload an MRI image to predict the stage of Alzheimer’s")

uploaded_file = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Predicting..."):
        pred_class, confidence, all_probs = predict(image)

    st.success(f"🎯 Predicted Stage: **{pred_class}** ({confidence*100:.2f}%)")
    
    st.subheader("📊 Confidence for All Classes:")
    st.bar_chart(all_probs)
