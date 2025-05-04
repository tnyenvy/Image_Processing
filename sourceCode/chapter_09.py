import streamlit as st
import cv2
import numpy as np
from PIL import Image

def count_objects(image, threshold=127):
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours), binary

def connected_components(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)
    return num_labels - 1, labels

def XuLyAnh():
    st.title("Chương 9: Xử lý hình thái")
    
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            
        processing_option = st.selectbox(
            "Chọn phương pháp xử lý",
            ["Đếm số hạt gạo", "Xử lý thành phần liên thông"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Kết quả xử lý")
            
            if processing_option == "Đếm số hạt gạo":
                threshold = st.slider("Ngưỡng", 0, 255, 127)
                count, binary = count_objects(img_gray, threshold)
                st.image(binary, use_column_width=True)
                st.write(f"Số lượng hạt: {count}")
                
            elif processing_option == "Xử lý thành phần liên thông":
                count, labeled_img = connected_components(img_gray)
                # Normalize để hiển thị
                labeled_img = (labeled_img * 255 / labeled_img.max()).astype(np.uint8)
                st.image(labeled_img, use_column_width=True)
                st.write(f"Số thành phần liên thông: {count}")