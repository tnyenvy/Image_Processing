import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from scipy.fftpack import fft2, ifft2

# Đường dẫn lưu ảnh gốc và ảnh sau xử lý
SAVE_PATH_ORIGINAL = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C05"
SAVE_PATH_PROCESSED = r"D:\XuLyAnhSo\DoAnCuoiKy\images\processedImages"

# Định nghĩa L cho mức xám (8-bit)
L = 256

# Tạo thư mục lưu ảnh sau xử lý nếu chưa tồn tại
os.makedirs(SAVE_PATH_PROCESSED, exist_ok=True)


def InsertImage(image_file=None, path='default.png', display_column=None):
    """
    Xử lý ảnh tải lên hoặc sử dụng ảnh mặc định từ đường dẫn.
   
    Args:
        image_file: File ảnh do người dùng tải lên (hoặc None nếu không có).
        path: Đường dẫn đến ảnh mặc định (nếu không có ảnh tải lên).
        display_column: Cột Streamlit để hiển thị ảnh (nếu cần).

    Returns:
        frame: Mảng NumPy của ảnh đã xử lý.
    """
    global image
    if image_file is not None:
        # Xử lý ảnh do người dùng tải lên
        image = Image.open(image_file)
        frame = np.array(image)
        if frame.ndim == 3 and frame.shape[2] == 3:  # Kiểm tra ảnh màu
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if display_column:
            display_column.image(frame, caption="Ảnh đã tải lên")
        image.close()
    else:
        # Sử dụng ảnh mặc định nếu không có ảnh tải lên
        default_path = os.path.join(SAVE_PATH_ORIGINAL, path)
        if not os.path.exists(default_path):
            st.error(f"Không tìm thấy ảnh mặc định tại: {default_path}")
            return None
        image = Image.open(default_path)
        frame = np.array(image)
        if frame.ndim == 3 and frame.shape[2] == 3:  # Kiểm tra ảnh màu
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if display_column:
            display_column.image(frame, caption="Ảnh Mặc Định")
        image.close()
   
    return frame


def display_images(original_img, processed_img, original_caption="Ảnh Gốc", processed_caption="Ảnh Đã Xử Lý"):
    """Hàm hiển thị ảnh gốc và ảnh đã xử lý cạnh nhau."""
    col1, col2 = st.columns(2)  # Tạo 2 cột để hiển thị ảnh
   
    if isinstance(original_img, str):
        with col1:
            st.image(original_img, caption=original_caption, use_container_width=True)
    else:
        with col1:
            st.image(original_img, caption=original_caption, use_container_width=True)
   
    if isinstance(processed_img, str):
        with col2:
            st.image(processed_img, caption=processed_caption, use_container_width=True)
    else:
        with col2:
            st.image(processed_img, caption=processed_caption, use_container_width=True)


def save_processed_image(img, filename):
    """Lưu ảnh đã xử lý vào thư mục."""
    processed_path = os.path.join(SAVE_PATH_PROCESSED, filename)
    if isinstance(img, np.ndarray):
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        result_img = Image.fromarray(img)
        result_img.save(processed_path)
    else:
        img.save(processed_path)
    return processed_path


# -------------------------
# Các hàm xử lý ảnh
# -------------------------

def CreateMotionfilter(M, N):
    """Tạo bộ lọc chuyển động."""
    H = np.zeros((M, N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            if np.abs(phi) < 1.0e-6:
                RE = T * np.cos(phi)
                IM = -T * np.sin(phi)
            else:
                RE = (T * np.sin(phi) / phi) * np.cos(phi)
                IM = -(T * np.sin(phi) / phi) * np.sin(phi)
            H.real[u, v] = RE
            H.imag[u, v] = IM
    return H


def CreateMotionNoise(imgin):
    """Tạo nhiễu chuyển động cho ảnh."""
    if imgin is None:
        st.error("Không có ảnh đầu vào để xử lý.")
        return None
        
    if len(imgin.shape) > 2:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
        
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    H = CreateMotionfilter(M, N)
    G = F * H
    G = np.fft.ifftshift(G)
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L - 1)
    g = g.astype(np.uint8)
    return g


def CreateInverseMotionfilter(M, N):
    """Tạo bộ lọc nghịch đảo chuyển động."""
    H = np.zeros((M, N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi * ((u - M // 2) * a + (v - N // 2) * b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi) / T
                IM = np.sin(phi) / T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = (phi / (T * np.sin(phi))) * np.cos(phi)
                IM = (phi / (T * np.sin(phi))) * np.sin(phi)
            H.real[u, v] = RE
            H.imag[u, v] = IM
            phi_prev = phi
    return H


def DenoiseMotion(imgin):
    """Giảm nhiễu chuyển động từ ảnh."""
    if imgin is None:
        st.error("Không có ảnh đầu vào để xử lý.")
        return None
        
    if len(imgin.shape) > 2:
        imgin = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
        
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    H = CreateInverseMotionfilter(M, N)
    G = F * H
    G = np.fft.ifftshift(G)
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L - 1)
    g = g.astype(np.uint8)
    return g


# -------------------------
# Các chức năng chính
# -------------------------
def add_noise():
    """Tạo nhiễu ảnh."""
    st.subheader("Tạo nhiễu ảnh")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="add_noise_uploader")
    img = InsertImage(uploaded_file, "TaoNhieuAnh.png")
    if img is None:
        return

    processed_image = CreateMotionNoise(img)
    display_images(img, processed_image, "Ảnh Gốc", "Ảnh Tạo Nhiễu")
    save_processed_image(processed_image, "add_noise_processed.png")


def run_denoise_image_light():
    """Lọc ảnh ít nhiễu."""
    st.subheader("Lọc ảnh ít nhiễu")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="denoise_light_uploader")
    img = InsertImage(uploaded_file, "LocAnhItNhieu.png")
    if img is None:
        return

    processed_image = DenoiseMotion(img)
    display_images(img, processed_image, "Ảnh Gốc", "Ảnh Lọc Ít Nhiễu")
    save_processed_image(processed_image, "denoise_light_processed.png")


def run_denoise_image_heavy():
    """Lọc ảnh nhiều nhiễu."""
    st.subheader("Lọc ảnh nhiều nhiễu")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="denoise_heavy_uploader")
    img = InsertImage(uploaded_file, "LocAnhNhieuNhieu.png")
    if img is None:
        return

    temp = cv2.medianBlur(img, 7)
    processed_image = DenoiseMotion(temp)
    display_images(img, processed_image, "Ảnh Gốc", "Ảnh Lọc Nhiều Nhiễu")
    save_processed_image(processed_image, "denoise_heavy_processed.png")


# -------------------------
# Ánh xạ chức năng
# -------------------------
chapter_05_functions = {
    "Tạo nhiễu ảnh": add_noise,
    "Lọc ảnh ít nhiễu": run_denoise_image_light,
    "Lọc ảnh nhiều nhiễu": run_denoise_image_heavy
}