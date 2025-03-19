import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms
import base64
import cv2
import numpy as np

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="PPE Detect",
    page_icon="logo/logo.png",
    layout="centered"
)

# Cache the YOLO model to optimize performance
@st.cache_resource()
def load_model():
    return YOLO("model/best.pt")  # Ensure correct model path

model = load_model()

# Define image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# Function to perform PPE detection on images
def predict_ppe(image: Image.Image):
    try:
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        results = model.predict(image_tensor)
        output_image = results[0].plot()  # Overlay predictions
        return Image.fromarray(output_image)
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return None

# Function to encode image to base64 for embedding
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return None

# Function for real-time PPE detection using webcam
def live_ppe_detection():
    st.sidebar.write("Starting live detection...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.sidebar.error("Error: Could not open webcam.")
        return
    
    stframe = st.empty()
    stop_button = st.sidebar.button("Stop Live Detection", key="stop_button")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.sidebar.error("Failed to capture video frame.")
            break
        
        results = model.predict(frame)
        output_frame = results[0].plot()
        stframe.image(output_frame, channels="BGR")
        
        if stop_button:
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Display logo
image_base64 = get_base64_image("logo/logo.png")
if image_base64:
    st.markdown(
        f'<div style="text-align: center;"><img src="data:image/png;base64,{image_base64}" width="100"></div>',
        unsafe_allow_html=True
    )

# UI Customization
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #1E1E2F; }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { color: white; }
        h1 { text-align: center; font-size: 36px; font-weight: bold; color: #2C3E50; }
        div.stButton > button { background-color: #3498DB; color: white; font-weight: bold; }
        div.stButton > button:hover { background-color: #2980B9; }
    </style>
""", unsafe_allow_html=True)

# Sidebar - File Upload
st.sidebar.header("📤 Upload an Image")
uploaded_file = st.sidebar.file_uploader("Drag and drop or browse", type=['jpg', 'png', 'jpeg'])

# Sidebar - Live Predictions
st.sidebar.header("📡 Live Predictions")
if st.sidebar.button("Start Live Detection", key="start_button"):
    live_ppe_detection()

# Main Page
st.title("PPE Detect")
st.markdown("<p style='text-align: center;'>Detect personal protective equipment (PPE) in images.</p>", unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    if st.sidebar.button("🔍 Predict PPE", key="predict_button"):
        detected_image = predict_ppe(image)
        if detected_image:
            with col2:
                st.image(detected_image, caption="🎯 PPE Detection Result", use_container_width=True)
        else:
            st.error("Detection failed. Please try again.")


st.info("This app uses **YOLO** for PPE detection. Upload an image or start live detection to get started.")
