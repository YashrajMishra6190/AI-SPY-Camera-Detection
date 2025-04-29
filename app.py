import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# ✅ MUST be first Streamlit call
st.set_page_config(page_title="Spy Camera Detector", layout="wide")

# Load the YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.conf = 0.3  # Confidence threshold
    return model

model = load_model()

# Streamlit Page Config
#st.set_page_config(page_title="Spy Camera Detector", layout="wide")
st.title("🔍 AI-Powered Spy Camera Detection")
st.markdown("This app detects hidden cameras or suspicious objects using a YOLOv5 object detection model.")

# Define suspected camera-related classes
camera_keywords = ['cell phone', 'camera', 'remote', 'laptop', 'microwave', 'tv']  # you can modify

# Function to detect and draw boxes
def detect_objects(image_np):
    results = model(image_np)
    detections = results.pandas().xyxy[0]

    for i in range(len(detections)):
        label = detections.iloc[i]['name']
        if label.lower() in camera_keywords:
            xmin, ymin = int(detections.iloc[i]['xmin']), int(detections.iloc[i]['ymin'])
            xmax, ymax = int(detections.iloc[i]['xmax']), int(detections.iloc[i]['ymax'])
            confidence = detections.iloc[i]['confidence']

            # Draw rectangle & label
            cv2.rectangle(image_np, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(
                image_np,
                f"{label} {confidence:.2f}",
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    return image_np

# Webcam detection
st.subheader("📸 Real-Time Spy Camera Detection")
run_camera = st.checkbox("Start Webcam", value=False)
frame_window = st.image([])

if run_camera:
    cap = cv2.VideoCapture(0)

    st.info("Uncheck to stop webcam.")
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam error.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_frame = detect_objects(frame_rgb.copy())
        frame_window.image(result_frame, channels="RGB", use_column_width=True)

    cap.release()
    st.success("Webcam stopped.")

# Upload image detection
st.subheader("🖼️ Upload Image to Scan for Spy Cameras")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image.convert("RGB"))
    
    result_image = detect_objects(image_np.copy())
    st.image(result_image, caption="Detection Result", use_column_width=True)
