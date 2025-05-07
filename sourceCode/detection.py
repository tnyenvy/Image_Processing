import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# 💫 Cấu hình trang (gọi 1 lần duy nhất)
#st.set_page_config(page_title="Ứng dụng nhận diện thông minh 🍓", layout="centered")

# ======================= HÀM NHẬN DIỆN TRÁI CÂY ==========================
def fruit_detection():
    st.header("Nhận diện trái cây với YOLOv8")

    # Load mô hình YOLOv8 đã huấn luyện
    model_path = "E:/xlas/Image_Processing/model/fruit_best.pt"
    model = YOLO(model_path)

    # Upload ảnh từ người dùng
    uploaded_file = st.file_uploader("Tải lên một bức ảnh chứa trái cây:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh gốc bạn vừa chọn",  use_container_width=True)

        if st.button("🍍 Bắt đầu nhận diện trái cây"):
            with st.spinner("Đang nhận diện trái cây thơm ngon..."):
                image_np = np.array(image)
                results = model(image_np)
                annotated_frame = results[0].plot()
                st.image(annotated_frame, caption="✨ Ảnh sau khi nhận diện",  use_container_width=True)
                st.success("Nhận diện hoàn tất!")
    else:
        st.info("Hãy tải lên một bức ảnh trước nhé~")

# ====================== HÀM NHẬN DIỆN NHÂN VẬT ===========================
def character_detection():
    st.header("Nhận diện nhân vật với YOLOv8")

    # Load mô hình YOLOv8 đã huấn luyện
    model_path = "E:/xlas/Image_Processing/model/character_best.pt"
    model = YOLO(model_path)

    # Upload ảnh từ người dùng
    uploaded_file = st.file_uploader("Tải lên một bức ảnh:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh gốc bạn vừa chọn",  use_container_width=True)

        if st.button("Bắt đầu nhận diện nhân vật"):
            with st.spinner("Đang nhận diện nhân vật cute..."):
                image_np = np.array(image)
                results = model(image_np)
                annotated_frame = results[0].plot()
                st.image(annotated_frame, caption="✨ Ảnh sau khi nhận diện",  use_container_width=True)
                st.success("Nhận diện hoàn tất!")
    else:
        st.info("Hãy tải lên một bức ảnh trước nhé~")

# ========================== GIAO DIỆN CHÍNH =============================
# Danh sách các chức năng nhận diện
Detection_functions = {
    "Nhận diện trái cây": fruit_detection,
    "Nhận diện nhân vật": character_detection
}