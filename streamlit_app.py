import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image
import os
import urllib.request

# =========================
# Load model secara otomatis
# =========================
@st.cache_resource
def load_model():
    model_path = "model/yolov5s.onnx"
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        with st.spinner("Downloading YOLOv5s model..."):
            url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.onnx"
            urllib.request.urlretrieve(url, model_path)
            st.success("✅ Model downloaded!")

    return ort.InferenceSession(model_path)

# Panggil model
model = load_model()

# =========================
# Streamlit antarmuka
# =========================
st.title("🔍 YOLOv5 ONNX Object Detection")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = cv2.resize(img_np, (640, 640))
    img_transposed = img_resized.transpose((2, 0, 1)) / 255.0
    input_tensor = img_transposed[np.newaxis, :].astype(np.float32)

    # Run inference
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})[0]
    
    boxes = outputs[0][:, :4]
    scores = outputs[0][:, 4]
    classes = outputs[0][:, 5].astype(int)

    result_img = img_np.copy()
    st.subheader("🎯 Detected Objects")

    conf_threshold = 0.3
    for box, score, cls_id in zip(boxes, scores, classes):
        if score > conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"Class {cls_id} ({score:.2f})"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            st.write(label)

    st.image(result_img, caption="📦 Detected Image", use_column_width=True)
