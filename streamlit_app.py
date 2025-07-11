import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image

st.title("🔍 Object Detection with YOLOv5 ONNX")

# Load ONNX model
@st.cache_resource
def load_model():
    return ort.InferenceSession("model/yolov5s.onnx")

model = load_model()

# Class names for COCO dataset (80 classes)
CLASS_NAMES = open("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/coco.names").read().strip().split("\n")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="Original Image", use_column_width=True)

    # Preprocessing
    img_resized = cv2.resize(img_np, (640, 640))
    img_transposed = img_resized.transpose((2, 0, 1)) / 255.0  # Normalize
    input_tensor = img_transposed[np.newaxis, :].astype(np.float32)

    # Inference
    outputs = model.run(None, {model.get_inputs()[0].name: input_tensor})[0]

    boxes = outputs[0][:, :4]
    scores = outputs[0][:, 4]
    classes = outputs[0][:, 5].astype(int)

    conf_threshold = 0.3
    result_img = img_np.copy()

    st.subheader("🎯 Detected Objects")
    for box, score, cls_id in zip(boxes, scores, classes):
        if score > conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(result_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            st.write(label)

    st.image(result_img, caption="Detected Image", use_column_width=True)
