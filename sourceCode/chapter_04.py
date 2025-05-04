import streamlit as st
import cv2
import numpy as np
from PIL import Image

def compute_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

def notch_reject_filter(image, d0=10):
    rows, cols = image.shape
    crow, ccol = rows//2, cols//2
    mask = np.ones((rows, cols), np.uint8)
    mask[crow-d0:crow+d0, ccol-d0:ccol+d0] = 0
    return mask

def XuLyAnh():
    st.title("Chương 4: Xử lý trong miền tần số")
    
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
            ["Spectrum", "Lọc trong miền tần số", "Bộ lọc notch-reject", "Xóa nhiễu Moire"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_column_width=True)
            
        with col2:
            st.subheader("Kết quả xử lý")
            
            if processing_option == "Spectrum":
                spectrum = compute_spectrum(img_gray)
                st.image(spectrum, use_column_width=True)
                
            elif processing_option == "Bộ lọc notch-reject":
                d0 = st.slider("Bán kính bộ lọc", 5, 50, 10)
                mask = notch_reject_filter(img_gray, d0)
                st.image(mask * 255, use_column_width=True)