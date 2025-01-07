import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import os

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model paths (Ensure these paths are correct for your environment)
leukemia_model_path = "leukemia_model.pth"
lung_colon_model_path = "lung_colon_model.pth"

# Define class labels for each model
leukemia_classes = ["Healthy", "Leukemia"]
lung_colon_classes = ["Colon Adenocarcinoma", "Colon Benign Tissue", "Lung Adenocarcinoma", "Lung Benign Tissue", "Lung Squamous Cell Carcinoma"]

# Load and set up models
def load_model(model_path, num_classes):
    try:
        model = models.resnet34(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load each model
leukemia_model = load_model(leukemia_model_path, len(leukemia_classes))
lung_colon_model = load_model(lung_colon_model_path, len(lung_colon_classes))

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image, model, classes):
    try:
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100  # Confidence in %
        return classes[predicted.item()], confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit app interface
st.title("Cancer Classification")
st.write("This app can classify images related to Leukemia and Lung/Colon Cancer.")

# Option to choose the type of model
model_type = st.selectbox("Select the model for classification:", ("Leukemia", "Lung/Colon Cancer"))

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)

    # Ensure the image has 3 channels (RGB) by converting it if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Run prediction based on selected model
    if st.button("Predict"):
        if model_type == "Leukemia" and leukemia_model:
            result, confidence = predict(image, leukemia_model, leukemia_classes)
            if result:
                st.write(f"Prediction (Leukemia Model): **{result}** with confidence **{confidence:.2f}%**")
        elif model_type == "Lung/Colon Cancer" and lung_colon_model:
            result, confidence = predict(image, lung_colon_model, lung_colon_classes)
            if result:
                st.write(f"Prediction (Lung/Colon Cancer Model): **{result}** with confidence **{confidence:.2f}%**")
        else:
            st.write("Selected model is not loaded properly. Please check the model paths and try again.")