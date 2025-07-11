import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import uuid

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8_model/best.pt")  # Ganti path jika perlu

model = load_model()

st.title("🧠 Object Detection App - YOLOv8")
st.markdown("Upload gambar untuk mendeteksi objek menggunakan model YOLOv8.")

uploaded_file = st.file_uploader("Pilih Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan gambar sementara
    img = Image.open(uploaded_file).convert("RGB")
    img_path = f"temp_{uuid.uuid4().hex}.jpg"
    img.save(img_path)

    st.image(img, caption="Gambar Diupload", use_column_width=True)

    with st.spinner("🔍 Mendeteksi objek..."):
        results = model.predict(source=img_path, save=True, conf=0.5)

    # Ambil path hasil
    result_path = results[0].save_dir
    output_img_path = os.path.join(result_path, os.path.basename(img_path))
