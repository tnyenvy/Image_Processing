import cv2
import numpy as np
import cv2.xphoto
import streamlit as st

#1 Hiệu ứng nét chì
def pencil_sketch_effect():
    st.title("Vẽ bằng nét chì")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inverted_image = cv2.bitwise_not(gray_image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
        
        # Invert the blurred image
        inverted_blurred = cv2.bitwise_not(blurred)
        
        # Create the pencil sketch
        sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
        
        # Display the original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.header("Ảnh gốc")
            st.image(image, channels="BGR", caption="Ảnh gốc")
        with col2:
            st.header("Ảnh sau khi xử lý")
            st.image(sketch, channels="GRAY", caption="Vẽ bằng nét chì")

#2 Hiệu ứng tranh sơn dầu
def enhance_image(image):
    """Tăng cường màu sắc và độ tương phản của ảnh."""
    # Chuyển sang không gian màu HSV để tăng độ bão hòa
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.2)  # Tăng độ bão hòa lên 20%
    s = np.clip(s, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Tăng độ tương phản và độ sáng
    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def oil_painting_effect():
    st.title("Hiệu ứng tranh sơn dầu cải tiến")
    
    uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Tùy chọn tham số từ người dùng
        st.header("Tùy chỉnh hiệu ứng")
        brush_size = st.slider("Kích thước nét vẽ", 1, 10, 5, key="brush_size")
        dyn_ratio = st.slider("Mức độ chi tiết", 1, 10, 1, key="dyn_ratio")
        smoothness = st.slider("Độ mịn", 0, 5, 1, key="smoothness")
        
        # Tăng cường ảnh trước khi áp dụng hiệu ứng
        enhanced_image = enhance_image(image)
        
        # Áp dụng hiệu ứng tranh sơn dầu
        oil_paint = cv2.xphoto.oilPainting(enhanced_image, brush_size, dyn_ratio)
        
        # Làm mịn nhẹ để giảm nhiễu (nếu smoothness > 0)
        if smoothness > 0:
            oil_paint = cv2.bilateralFilter(oil_paint, 5, smoothness * 10, smoothness * 10)
        
        # Display the original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.header("Ảnh gốc")
            st.image(image, channels="BGR", caption="Ảnh gốc")
        with col2:
            st.header("Tranh sơn dầu")
            st.image(oil_paint, channels="BGR", caption="Hiệu ứng tranh sơn dầu")
        
        # Tùy chọn tải ảnh xuống
        st.download_button(
            label="Tải ảnh đã xử lý",
            data=cv2.imencode('.png', oil_paint)[1].tobytes(),
            file_name="oil_painting.png",
            mime="image/png"
        )


chapter_07_functions = {
    "Vẽ bằng nét chì": pencil_sketch_effect,
    "Hiệu ứng tranh sơn dầu": oil_painting_effect
}