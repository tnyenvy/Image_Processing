import streamlit as st
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
import datetime
import matplotlib.pyplot as plt


# Đường dẫn lưu ảnh gốc và ảnh sau xử lý
SAVE_PATH_ORIGINAL = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C05"
SAVE_PATH_PROCESSED = r"D:\XuLyAnhSo\DoAnCuoiKy\images\processedImages"


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
    global image  # Khai báo biến toàn cục để lưu trữ ảnh
    if image_file is not None:
        # Xử lý ảnh do người dùng tải lên
        image = Image.open(image_file)
        frame = np.array(image)
        if frame.ndim == 3 and frame.shape[2] == 3:  # Kiểm tra ảnh màu
            frame = frame[:, :, :3]  # Lấy 3 kênh RGB
        if display_column:
            display_column.image(image, caption="Ảnh đã tải lên")
        image.close()
    else:
        # Sử dụng ảnh mặc định nếu không có ảnh tải lên
        default_path = os.path.join(SAVE_PATH_ORIGINAL, path)
        if not os.path.exists(default_path):
            st.error(f"Không tìm thấy ảnh mặc định tại: {default_path}")
            return None
        image = Image.open(default_path)
        frame = np.array(image)
        if display_column:
            display_column.image(image, caption="Ảnh Mặc Định")
        image.close()
   
    return frame


def display_images(original_img, processed_img, original_caption="Ảnh Gốc", processed_caption="Ảnh Đã Xử Lý"):
    """Hàm hiển thị ảnh gốc và ảnh đã xử lý cạnh nhau."""
    col1, col2 = st.columns(2)  # Tạo 2 cột để hiển thị ảnh
   
    # Kiểm tra xem original_img có phải là đường dẫn không
    if isinstance(original_img, str):
        with col1:
            st.image(original_img, caption=original_caption, use_container_width=True)
    else:
        with col1:
            st.image(original_img, caption=original_caption, use_container_width=True)
   
    # Kiểm tra xem processed_img có phải là đường dẫn không
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
        # Chuyển từ NumPy array sang PIL Image
        if img.dtype == np.float64 or img.dtype == np.float32:
            img = (img * 255).astype(np.uint8)
        result_img = Image.fromarray(img)
        result_img.save(processed_path)
    else:
        # Nếu đã là PIL Image
        img.save(processed_path)
    return processed_path

# 1. Tạo nhiễu ảnh
def add_noise():
    """Thêm nhiễu vào ảnh."""
    st.header("Tạo nhiễu ảnh")
    
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="noise_uploader")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "TaoNhieuAnh.png")
    
    if img is None:
        return
    
    # Chọn loại nhiễu
    noise_type = st.selectbox(
        "Chọn loại nhiễu:",
        ["Gaussian (Nhiễu Gauss)", "Salt & Pepper (Nhiễu muối tiêu)", "Speckle (Nhiễu đốm)"]
    )
    
    # Các tham số nhiễu
    if noise_type == "Gaussian (Nhiễu Gauss)":
        mean = st.slider("Giá trị trung bình (Mean):", -50.0, 50.0, 0.0, 1.0)
        sigma = st.slider("Độ lệch chuẩn (Sigma):", 0.0, 100.0, 25.0, 1.0)
    elif noise_type == "Salt & Pepper (Nhiễu muối tiêu)":
        salt_vs_pepper = st.slider("Tỷ lệ Salt vs. Pepper:", 0.0, 1.0, 0.5, 0.01)
        amount = st.slider("Mức độ nhiễu:", 0.0, 0.5, 0.05, 0.01)
    elif noise_type == "Speckle (Nhiễu đốm)":
        intensity = st.slider("Cường độ nhiễu:", 0.0, 1.0, 0.1, 0.01)
    
    if st.button("Tạo nhiễu"):
        try:
            # Tạo bản sao của ảnh để xử lý
            img_noisy = img.copy().astype(np.float32)
            
            # Thêm nhiễu tùy theo loại được chọn
            if noise_type == "Gaussian (Nhiễu Gauss)":
                noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
                img_noisy = img_noisy + noise
                img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
                noise_label = f"Gaussian_mean{mean}_sigma{sigma}"
            
            elif noise_type == "Salt & Pepper (Nhiễu muối tiêu)":
                img_noisy = img.copy()
                # Salt (white) noise
                salt_mask = np.random.random(img.shape[:2]) < amount * salt_vs_pepper
                if len(img.shape) == 3:  # Ảnh màu
                    for i in range(3):
                        img_noisy[salt_mask, i] = 255
                else:  # Ảnh xám
                    img_noisy[salt_mask] = 255
                
                # Pepper (black) noise
                pepper_mask = np.random.random(img.shape[:2]) < amount * (1 - salt_vs_pepper)
                if len(img.shape) == 3:  # Ảnh màu
                    for i in range(3):
                        img_noisy[pepper_mask, i] = 0
                else:  # Ảnh xám
                    img_noisy[pepper_mask] = 0
                
                noise_label = f"SaltPepper_amount{amount}_ratio{salt_vs_pepper}"
            
            elif noise_type == "Speckle (Nhiễu đốm)":
                noise = intensity * np.random.randn(*img.shape).astype(np.float32)
                img_noisy = img.astype(np.float32) * (1 + noise)
                img_noisy = np.clip(img_noisy, 0, 255).astype(np.uint8)
                noise_label = f"Speckle_intensity{intensity}"
            
            # Hiển thị kết quả
            display_images(img, img_noisy, "Ảnh gốc", f"Ảnh với {noise_type}")
            
            # Lưu ảnh đã xử lý
            processed_path = save_processed_image(img_noisy, f"noisy_{noise_label}.png")
            
        except Exception as e:
            st.error(f"Lỗi khi tạo nhiễu: {str(e)}")


# 2. Lọc ảnh ít nhiễu
def run_denoise_image_light():
    st.header("Xử lý ảnh nâng cao - Độ nhiễu thấp")
    
    uploaded_file = st.file_uploader("Chọn ảnh để xử lý:", type=['jpg', 'jpeg', 'png'], key="denoise_light_uploader")
    
    if uploaded_file is not None:
        # Đọc ảnh trực tiếp từ file upload
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Chuyển BGR sang RGB để hiển thị đúng
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Sử dụng ảnh mẫu
        img = cv2.imread("LocAnhItNhieu.png")
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        st.error("Không có ảnh đầu vào để xử lý.")
        return
    
    # Hiển thị ảnh gốc
    st.image(img, caption="Ảnh gốc", use_column_width=True)
    
    # Tham số cho việc xử lý ảnh
    st.subheader("Tùy chỉnh thông số")
    
    # Chọn kỹ thuật xử lý
    technique = st.selectbox(
        "Chọn kỹ thuật xử lý:",
        ["Unsharp Masking cao cấp", "Detail Enhancement", "Frequency Domain Filtering", "Edge-Preserving Smoothing"]
    )
    
    # Tham số cho từng kỹ thuật
    if technique == "Unsharp Masking cao cấp":
        col1, col2 = st.columns(2)
        with col1:
            kernel_size = st.slider("Kích thước kernel:", 3, 15, 5, 2, key="kernel_size_um")
        with col2:
            sharpening_amount = st.slider("Cường độ làm nét:", 1.0, 10.0, 5.0, 0.5, key="sharpening_um")
    
    elif technique == "Detail Enhancement":
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_s = st.slider("Sigma Spatial:", 1, 200, 60, 10, key="sigma_s")
        with col2:
            sigma_r = st.slider("Sigma Range:", 1, 100, 45, 5, key="sigma_r") / 100.0
        with col3:
            boost = st.slider("Detail Boost:", 1.0, 10.0, 4.0, 0.5, key="detail_boost")
    
    elif technique == "Frequency Domain Filtering":
        col1, col2 = st.columns(2)
        with col1:
            high_boost = st.slider("High Boost Factor:", 1.0, 10.0, 2.5, 0.5, key="high_boost")
        with col2:
            cutoff = st.slider("Cutoff Frequency (%):", 1, 50, 10, 1, key="cutoff")
    
    elif technique == "Edge-Preserving Smoothing":
        col1, col2, col3 = st.columns(3)
        with col1:
            d = st.slider("Độ mờ:", 5, 50, 15, 5, key="eps_d")
        with col2:
            sigmaColor = st.slider("Sigma Color:", 10, 200, 75, 5, key="eps_sigma_color")
        with col3:
            sharpness = st.slider("Độ sắc nét:", 1.0, 5.0, 2.5, 0.1, key="eps_sharpness")
    
    if st.button("Xử lý ảnh", key="process_light"):
        if technique == "Unsharp Masking cao cấp":
            advanced_unsharp_masking(img, kernel_size, sharpening_amount)
        elif technique == "Detail Enhancement":
            detail_enhancement(img, sigma_s, sigma_r, boost)
        elif technique == "Frequency Domain Filtering":
            frequency_domain_filtering(img, high_boost, cutoff)
        elif technique == "Edge-Preserving Smoothing":
            edge_preserving_smoothing(img, d, sigmaColor, sharpness)


def advanced_unsharp_masking(img, kernel_size=5, amount=5.0):
    """
    Unsharp Masking cao cấp với nhiều lớp lọc và điều chỉnh màu sắc.
    """
    try:
        # Đảm bảo kernel_size là số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Chuyển ảnh sang không gian Lab để xử lý kênh độ sáng riêng biệt
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Áp dụng unsharp masking chỉ trên kênh độ sáng (L)
            # Lọc Gaussian để tạo bản mờ
            blurred_l = cv2.GaussianBlur(l, (kernel_size, kernel_size), 0)
            
            # Tính mask và tăng cường chi tiết
            mask = cv2.subtract(l, blurred_l)
            sharpened_l = cv2.addWeighted(l, 1.0, mask, amount, 0)
            
            # Đảm bảo giá trị nằm trong khoảng [0, 255]
            sharpened_l = np.clip(sharpened_l, 0, 255).astype(np.uint8)
            
            # Tạo ảnh Lab mới và chuyển về RGB
            enhanced_lab = cv2.merge([sharpened_l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Áp dụng thêm một lớp chi tiết cho toàn bộ ảnh RGB
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9 + amount/2, -1],
                                          [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel_sharpening)
            
            # Tăng độ bão hòa màu sắc
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            # Tăng độ bão hòa
            s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
            enhanced_hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        else:
            # Xử lý ảnh grayscale
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            mask = cv2.subtract(img, blurred)
            enhanced = cv2.addWeighted(img, 1.0, mask, amount, 0)
            
            # Áp dụng thêm kernel sắc nét
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9 + amount/2, -1],
                                          [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel_sharpening)
        
        # Đảm bảo giá trị nằm trong khoảng [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Hiển thị và lưu kết quả
        display_and_save_result(img, enhanced, "Unsharp Masking Cao Cấp", f"unsharp_advanced_{int(amount)}.png")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")


def detail_enhancement(img, sigma_s=60, sigma_r=0.45, boost=4.0):
    """
    Sử dụng DetailEnhance của OpenCV để tăng cường chi tiết.
    Kết hợp với xử lý màu sắc nâng cao.
    """
    try:
        # Giảm nhiễu nhẹ trước khi tăng cường chi tiết
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21) if len(img.shape) == 3 else cv2.fastNlMeansDenoising(img, None, 5, 7, 21)
        
        # Tăng cường chi tiết
        enhanced = cv2.detailEnhance(denoised, sigma_s=sigma_s, sigma_r=sigma_r)
        
        # Tăng cường thêm chi tiết với một kernel sắc nét tùy chỉnh
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + (boost - 1), -1],
                           [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # Tăng độ tương phản
        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Tăng độ bão hòa
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * 1.3, 0, 255).astype(np.uint8)
        enhanced_hsv = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        # Đảm bảo giá trị nằm trong khoảng [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Hiển thị và lưu kết quả
        display_and_save_result(img, enhanced, "Detail Enhancement", f"detail_enhanced_{int(sigma_s)}_{int(sigma_r*100)}_{int(boost*10)}.png")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")


def frequency_domain_filtering(img, high_boost=2.5, cutoff_percent=10):
    """
    Xử lý ảnh trong miền tần số - một kỹ thuật mạnh mẽ để lọc nhiễu và tăng cường chi tiết.
    """
    try:
        # Chuyển đổi sang grayscale nếu là ảnh màu
        if len(img.shape) == 3:
            # Xử lý từng kênh màu
            b, g, r = cv2.split(img) if img.shape[2] == 3 else (img, img, img)
            
            # Xử lý ảnh trong miền tần số cho từng kênh
            enhanced_b = process_channel_fft(b, high_boost, cutoff_percent)
            enhanced_g = process_channel_fft(g, high_boost, cutoff_percent)
            enhanced_r = process_channel_fft(r, high_boost, cutoff_percent)
            
            # Kết hợp các kênh đã xử lý
            enhanced = cv2.merge([enhanced_b, enhanced_g, enhanced_r])
            
            # Tăng độ tương phản và độ sắc nét
            lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge([cl, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # Xử lý ảnh grayscale
            enhanced = process_channel_fft(img, high_boost, cutoff_percent)
        
        # Đảm bảo giá trị nằm trong khoảng [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Hiển thị và lưu kết quả
        display_and_save_result(img, enhanced, "Frequency Domain Filtering", f"frequency_enhanced_{int(high_boost*10)}_{cutoff_percent}.png")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh trong miền tần số: {str(e)}")


def process_channel_fft(channel, high_boost, cutoff_percent):
    """
    Xử lý một kênh màu trong miền tần số.
    """
    # Chuyển đổi sang floating point để tránh mất mát dữ liệu
    channel_float = channel.astype(np.float32)
    
    # Áp dụng FFT
    dft = cv2.dft(channel_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tạo high-pass filter với cutoff phụ thuộc vào kích thước ảnh
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2
    
    # Tạo mask cho high-pass filter
    mask = np.ones((rows, cols, 2), np.float32)
    r = min(crow, ccol) * cutoff_percent / 100
    
    # Tạo filter dạng Butterworth
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            # Butterworth high-pass filter
            mask[i, j] = 1 / (1 + (r / (d + 1e-5)) ** 4)
    
    # Tăng cường tần số cao (high boost)
    mask = mask * high_boost
    
    # Áp dụng filter
    fshift = dft_shift * mask
    
    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Chuẩn hóa lại giá trị pixel
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    
    return img_back


def edge_preserving_smoothing(img, d=15, sigmaColor=75, sharpness=2.5):
    """
    Kết hợp làm mịn bảo toàn cạnh và tăng cường chi tiết.
    """
    try:
        # Bước 1: Áp dụng Bilateral Filter để làm mịn và bảo toàn cạnh
        smoothed = cv2.bilateralFilter(img, d, sigmaColor, sigmaColor//2)
        
        # Bước 2: Tính toán mask của cạnh
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Phát hiện cạnh với Canny
        edges = cv2.Canny(gray, 50, 150)
        
        # Mở rộng cạnh để tránh mất chi tiết
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Làm mờ cạnh để tạo mặt nạ mềm
        edge_mask = cv2.GaussianBlur(edges, (0, 0), sigmaX=2)
        
        # Bước 3: Tạo phiên bản sắc nét
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1, 9 + sharpness, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        
        # Bước 4: Kết hợp ảnh làm mịn và ảnh sắc nét dựa trên mặt nạ cạnh
        if len(img.shape) == 3:
            # Mở rộng mặt nạ cạnh thành 3 kênh màu
            edge_mask_3ch = cv2.merge([edge_mask, edge_mask, edge_mask])
            
            # Chuẩn hóa mặt nạ về khoảng [0, 1]
            edge_mask_norm = edge_mask_3ch / 255.0
            
            # Kết hợp: Ở vùng cạnh sử dụng ảnh sắc nét, vùng đồng nhất sử dụng ảnh làm mịn
            enhanced = smoothed * (1 - edge_mask_norm) + sharpened * edge_mask_norm
        else:
            edge_mask_norm = edge_mask / 255.0
            enhanced = smoothed * (1 - edge_mask_norm) + sharpened * edge_mask_norm
        
        # Bước 5: Tăng độ tương phản và màu sắc
        if len(img.shape) == 3:
            # Tăng độ tương phản
            lab = cv2.cvtColor(enhanced.astype(np.uint8), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge([cl, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
            
            # Tăng độ bão hòa màu
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s * 1.2, 0, 255).astype(np.uint8)
            enhanced_hsv = cv2.merge([h, s, v])
            enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)
        
        # Đảm bảo giá trị nằm trong khoảng [0, 255]
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        # Hiển thị và lưu kết quả
        display_and_save_result(img, enhanced, "Edge-Preserving Smoothing", f"edge_preserved_{d}_{sigmaColor}_{int(sharpness*10)}.png")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")


def display_and_save_result(original, processed, technique_name, filename):
    """
    Hiển thị kết quả và lưu ảnh.
    """
    # Hiển thị kết quả
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Ảnh gốc", use_column_width=True)
    with col2:
        st.image(processed, caption=f"Ảnh sau xử lý: {technique_name}", use_column_width=True)
    
    # Lưu ảnh đã xử lý
    if len(processed.shape) == 3:
        save_img = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
    else:
        save_img = processed
    
    cv2.imwrite(filename, save_img)
    st.success(f"Đã lưu ảnh đã xử lý thành công: {filename}")
    
    # Tạo nút tải xuống
    with open(filename, "rb") as file:
        btn = st.download_button(
            label="Tải xuống ảnh đã xử lý",
            data=file,
            file_name=filename,
            mime="image/png"
        )

#3 Lọc ảnh nhiều nhiễu
def loc_anh_it_nhieu(image_path):
    # Đọc ảnh gốc
    img = cv2.imread(image_path)

    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Lọc nhiễu bằng Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # Làm sắc nét ảnh để tăng chi tiết sau khi lọc
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1), plt.imshow(gray, cmap='gray'), plt.title('Ảnh gốc (xám)')
    plt.subplot(1, 3, 2), plt.imshow(denoised, cmap='gray'), plt.title('Đã lọc nhiễu')
    plt.subplot(1, 3, 3), plt.imshow(sharpened, cmap='gray'), plt.title('Lọc + Làm sắc nét')
    plt.show()

    return sharpened

# Dictionary ánh xạ chức năng 
chapter_05_functions = {
    "Tạo nhiễu ảnh": add_noise,
    "Lọc ảnh ít nhiễu": run_denoise_image_light,
    "Lọc ảnh nhiều nhiễu": loc_anh_it_nhieu
}
