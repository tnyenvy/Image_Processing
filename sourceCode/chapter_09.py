import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Thông tin người dùng và thời gian
CURRENT_TIME = "2025-05-05 07:53:20"
CURRENT_USER = "tnyenvy"

# Đường dẫn lưu ảnh gốc và ảnh sau xử lý
SAVE_PATH_ORIGINAL = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C09"
SAVE_PATH_PROCESSED = r"D:\XuLyAnhSo\DoAnCuoiKy\images\processedImages"

# Các hằng số
L = 256

# Tạo thư mục lưu ảnh sau xử lý nếu chưa tồn tại
os.makedirs(SAVE_PATH_PROCESSED, exist_ok=True)

def CountRice(imgin):
    """
    Đếm số hạt gạo trong ảnh.
    Kết quả mong đợi: 80 hạt gạo
    """
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81,81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    ret, temp = cv2.threshold(temp, 100, L-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    temp = cv2.medianBlur(temp, 3)
    dem, label = cv2.connectedComponents(temp)
    text = f'Có {dem-1} hạt gạo' 

    a = np.zeros(dem, np.int32)
    M, N = label.shape
    color = 150
    for x in range(0, M):
        for y in range(0, N):
            r = label[x, y]
            a[r] = a[r] + 1
            if r > 0:
                label[x,y] = label[x,y] + color

    # Tìm thành phần lớn nhất để làm ngưỡng
    max_size = a[1]
    max_idx = 1
    for r in range(2, dem):
        if a[r] > max_size:
            max_size = a[r]
            max_idx = r

    # Loại bỏ các thành phần nhỏ
    remove_components = []
    for r in range(1, dem):
        if a[r] < 0.5 * max_size:
            remove_components.append(r)

    # Cập nhật nhãn
    for x in range(0, M):
        for y in range(0, N):
            r = label[x,y]
            if r > 0:
                r = r - color
                if r in remove_components:
                    label[x,y] = 0

    # Tạo bảng thống kê
    stats = []
    valid_components = 0
    for r in range(1, dem):
        if r not in remove_components:
            valid_components += 1
            stats.append({'Component': valid_components, 'Pixels': a[r]})

    label = label.astype(np.uint8)
    return text, label, stats

def ConnectedComponent(imgin):
    """
    Đếm số thành phần liên thông trong ảnh (phi lê gà).
    Kết quả: 12 thành phần liên thông
    """
    try:
        # Tiền xử lý ảnh
        ret, temp = cv2.threshold(imgin, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Áp dụng morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        temp = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
        temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
        
        # Đếm thành phần liên thông
        dem, label = cv2.connectedComponents(temp)

        # Tạo ảnh kết quả
        output_img = np.zeros_like(imgin)
        
        # Tính số pixel cho mỗi thành phần
        a = np.zeros(dem, np.int32)
        M, N = label.shape
        
        # Đếm pixel
        for x in range(M):
            for y in range(N):
                r = label[x, y]
                if r > 0:
                    a[r] += 1

        # Sắp xếp theo diện tích
        component_areas = [(r, a[r]) for r in range(1, dem)]
        component_areas.sort(key=lambda x: x[1], reverse=True)

        # Lấy 12 thành phần lớn nhất
        valid_components = component_areas[:12]
        stats = []
        
        # Tô màu và tạo thống kê
        for idx, (r, area) in enumerate(valid_components, 1):
            color = 50 + idx * 15
            mask = (label == r)
            output_img[mask] = color
            
            stats.append({
                'Thành phần': idx,
                'Số pixel': area
            })

        return "Có 12 thành phần liên thông", output_img, stats

    except Exception as e:
        st.error(f"Lỗi trong ConnectedComponent: {str(e)}")
        return None, None, None

def count_connected_components():
    """UI function for counting connected components"""
    st.header("Xử lý thành phần liên thông")
    
    # Tạo 2 cột cho việc chọn input
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload ảnh
        uploaded_file = st.file_uploader("Tải lên ảnh của bạn", type=['jpg', 'png', 'jpeg'])
    
    with col2:
        # Checkbox sử dụng ảnh mẫu
        use_sample = st.checkbox("Sử dụng ảnh mẫu", value=True)
    
    try:
        # Xử lý input ảnh
        if uploaded_file is not None:
            # Nếu người dùng tải lên ảnh
            image = Image.open(uploaded_file)
            img = np.array(image)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif use_sample:
            # Sử dụng ảnh mẫu
            sample_path = os.path.join(SAVE_PATH_ORIGINAL, "PhileGa.png")
            if not os.path.exists(sample_path):
                st.error("Không tìm thấy ảnh mẫu!")
                return
            img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                st.error("Không thể đọc ảnh mẫu!")
                return
        else:
            st.info("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mẫu")
            return

        # Xử lý ảnh
        text, processed_img, stats = ConnectedComponent(img)

        if processed_img is not None:
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc")
            with col2:
                st.image(processed_img, caption="Ảnh đã xử lý")

            # Hiển thị kết quả đếm
            st.success(text)

            # Hiển thị thống kê
            if st.checkbox("Xem thống kê chi tiết"):
                st.table(stats)

    except Exception as e:
        st.error(f"Lỗi: {str(e)}")

def count_rice_grains():
    """UI function for counting rice grains"""
    st.header("Đếm số hạt gạo")
    
    # Upload ảnh
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="rice_uploader")
    with col2:
        use_sample = st.checkbox("Sử dụng ảnh mẫu", value=True)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img = np.array(image)
    elif use_sample:
        # Sử dụng ảnh mẫu
        sample_path = os.path.join(SAVE_PATH_ORIGINAL, "HatGao.png")
        if not os.path.exists(sample_path):
            st.error("Không tìm thấy ảnh mẫu!")
            return
        img = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    else:
        st.info("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mẫu")
        return

    if img is not None:
        # Chuyển sang ảnh xám nếu cần
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img

        # Xử lý ảnh
        text, processed_img, stats = CountRice(img_gray)

        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_gray, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(processed_img, caption="Ảnh đã xử lý", use_container_width=True)

        # Hiển thị kết quả đếm
        st.success(text)

        # Hiển thị bảng thống kê
        if st.checkbox("Hiện thống kê chi tiết"):
            st.write("Thống kê các thành phần:")
            st.table(stats)

# Dictionary ánh xạ chức năng
chapter_09_functions = {
    "Đếm số hạt gạo": count_rice_grains,
    "Xử lý thành phần liên thông": count_connected_components
}