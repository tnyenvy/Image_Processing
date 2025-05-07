import streamlit as st
import cv2
import os
import numpy as np
from io import BytesIO
from PIL import Image
import datetime
import matplotlib.pyplot as plt

# Đường dẫn lưu ảnh gốc và ảnh sau xử lý
SAVE_PATH_ORIGINAL = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C04"
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

# 1. Hàm xử lý phổ tần số
def compute_spectrum(image):
    """Tính phổ tần số của ảnh."""
    try:
        f = np.fft.fft2(image)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        return fshift, magnitude_spectrum.astype(np.uint8)
    except Exception as e:
        st.error(f"Lỗi khi tính phổ tần số: {str(e)}")
        return None, None

def spectrum():
    """Hiển thị phổ tần số của ảnh."""
    st.header("Phổ tần số của ảnh")
    
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="spectrum_uploader")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "1. Spectrum.png")
    
    if img is None:
        return
    
    try:
        # Chuyển sang ảnh xám nếu là ảnh màu
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Tính phổ tần số
        fshift, magnitude_spectrum = compute_spectrum(img_gray)
        
        if fshift is None or magnitude_spectrum is None:
            return
        
        # Hiển thị kết quả
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(magnitude_spectrum, caption="Phổ tần số", use_container_width=True)
        
        # Lưu ảnh đã xử lý
        processed_path = save_processed_image(magnitude_spectrum, "1. Spectrum_processed.png")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý ảnh: {str(e)}")

# 2. Hàm lọc trong miền tần số
def apply_frequency_filter(image, filter_type, cutoff, cutoff2=None, order=1):
    """
    Áp dụng bộ lọc trong miền tần số cho ảnh đầu vào
    
    Parameters:
    -----------
    image : numpy.ndarray
        Ảnh đầu vào (grayscale)
    filter_type : str
        Loại bộ lọc: "Low Pass", "High Pass", "Band Pass", "Band Reject", "Butterworth LP", "Butterworth HP", "Gaussian LP", "Gaussian HP"
    cutoff : int
        Tần số cắt chính
    cutoff2 : int, optional
        Tần số cắt thứ hai (cho lọc Band Pass/Reject)
    order : int, optional
        Bậc của bộ lọc Butterworth
        
    Returns:
    --------
    tuple
        (mask, filtered_image, spectrum_filtered)
    """
    # Chuẩn bị ảnh
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Tạo lưới tọa độ
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    
    # Tính khoảng cách từ tâm
    d = np.sqrt(u**2 + v**2)
    
    # Tránh chia cho 0
    d = np.maximum(d, 0.1)
    
    # Tạo mặt nạ lọc theo loại bộ lọc
    if filter_type == "Low Pass":
        mask = (d <= cutoff).astype(float)
    elif filter_type == "High Pass":
        mask = (d > cutoff).astype(float)
    elif filter_type == "Band Pass":
        inner_radius = cutoff
        outer_radius = cutoff2 if cutoff2 is not None else cutoff + cutoff//2
        mask = ((d >= inner_radius) & (d <= outer_radius)).astype(float)
    elif filter_type == "Band Reject":
        inner_radius = cutoff
        outer_radius = cutoff2 if cutoff2 is not None else cutoff + cutoff//2
        mask = ((d < inner_radius) | (d > outer_radius)).astype(float)
    elif filter_type == "Butterworth LP":
        mask = 1 / (1 + (d / cutoff)**(2 * order))
    elif filter_type == "Butterworth HP":
        mask = 1 / (1 + (cutoff / d)**(2 * order))
    elif filter_type == "Gaussian LP":
        mask = np.exp(-(d**2) / (2 * cutoff**2))
    elif filter_type == "Gaussian HP":
        mask = 1 - np.exp(-(d**2) / (2 * cutoff**2))
    else:
        mask = np.ones((rows, cols))
    
    # Áp dụng FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    # Áp dụng bộ lọc
    fshift_filtered = fshift * mask
    
    # Tính phổ sau khi lọc để hiển thị
    spectrum_filtered = np.log(np.abs(fshift_filtered) + 1)
    spectrum_filtered = cv2.normalize(spectrum_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Chuyển ngược lại miền không gian
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Chuẩn hóa kết quả
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mask, img_back, spectrum_filtered

def frequency_domain_filter():
    st.header("Lọc trong miền tần số")
    
    # Tải lên ảnh
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="filter_uploader")
    
    if uploaded_file is not None:
        # Đọc ảnh và chuyển sang grayscale
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_array
            
        # Tạo layout với 2 cột
        main_cols = st.columns([1, 2])
        
        with main_cols[0]:
            st.subheader("Điều khiển")
            
            # Chọn loại bộ lọc
            filter_options = [
                "Low Pass", "High Pass", 
                "Band Pass", "Band Reject", 
                "Butterworth LP", "Butterworth HP", 
                "Gaussian LP", "Gaussian HP"
            ]
            filter_type = st.selectbox("Loại bộ lọc", filter_options)
            
            # Xác định giá trị tối đa cho cutoff dựa trên kích thước ảnh
            max_cutoff = min(img_gray.shape) // 2
            
            # Tham số cơ bản (cutoff chính)
            cutoff = st.slider("Tần số cắt", 1, max_cutoff, max_cutoff // 8)
            
            # Tham số bổ sung dựa trên loại bộ lọc
            cutoff2 = None
            order = 1
            
            if filter_type in ["Band Pass", "Band Reject"]:
                cutoff2 = st.slider("Tần số cắt thứ hai", cutoff + 1, max_cutoff, min(cutoff + max_cutoff // 8, max_cutoff))
            
            if filter_type in ["Butterworth LP", "Butterworth HP"]:
                order = st.slider("Bậc của bộ lọc", 1, 10, 2)
            
            # Hiển thị mô tả bộ lọc
            filter_info = {
                "Low Pass": "Giữ lại các tần số thấp và loại bỏ các tần số cao.",
                "High Pass": "Giữ lại các tần số cao và loại bỏ các tần số thấp.",
                "Band Pass": "Giữ lại các tần số trong một dải nhất định.",
                "Band Reject": "Loại bỏ các tần số trong một dải nhất định.",
                "Butterworth LP": "Lọc thông thấp Butterworth với sự chuyển tiếp mượt mà.",
                "Butterworth HP": "Lọc thông cao Butterworth với sự chuyển tiếp mượt mà.",
                "Gaussian LP": "Lọc thông thấp Gaussian, giảm dần mượt mà theo phân phối Gaussian.",
                "Gaussian HP": "Lọc thông cao Gaussian, tăng dần mượt mà theo phân phối Gaussian."
            }
            
            st.info(filter_info.get(filter_type, ""))
        
        # Áp dụng bộ lọc với tham số đã chọn
        if filter_type in ["Band Pass", "Band Reject"]:
            mask, filtered_image, spectrum_filtered = apply_frequency_filter(img_gray, filter_type, cutoff, cutoff2)
        elif filter_type in ["Butterworth LP", "Butterworth HP"]:
            mask, filtered_image, spectrum_filtered = apply_frequency_filter(img_gray, filter_type, cutoff, order=order)
        else:
            mask, filtered_image, spectrum_filtered = apply_frequency_filter(img_gray, filter_type, cutoff)
        
        # Tính phổ tần số gốc
        _, spectrum_original = compute_spectrum(img_gray)
        
        # Hiển thị mask cho mục đích trực quan
        mask_visual = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Hiển thị kết quả trong cột thứ hai
        with main_cols[1]:
            st.subheader("Kết quả")
            
            # Tạo tabs cho các loại hiển thị khác nhau
            tabs = st.tabs(["Ảnh", "Phổ tần số", "Mặt nạ"])
            
            with tabs[0]:
                result_cols = st.columns(2)
                with result_cols[0]:
                    st.image(image, caption="Ảnh gốc", use_container_width=True)
                with result_cols[1]:
                    st.image(filtered_image, caption=f"Ảnh sau khi lọc {filter_type}", use_container_width=True)
            
            with tabs[1]:
                spectrum_cols = st.columns(2)
                with spectrum_cols[0]:
                    st.image(spectrum_original, caption="Phổ tần số gốc", use_container_width=True)
                with spectrum_cols[1]:
                    st.image(spectrum_filtered, caption="Phổ tần số sau khi lọc", use_container_width=True)
            
            with tabs[2]:
                st.image(mask_visual, caption="Mặt nạ bộ lọc", use_container_width=True)
        
        # Thêm nút tải xuống ảnh đã lọc
        buffered = BytesIO()
        Image.fromarray(filtered_image).save(buffered, format="PNG")
        st.download_button(
            label="📥 Tải xuống ảnh đã lọc",
            data=buffered.getvalue(),
            file_name=f"filtered_{filter_type.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
    else:
        st.info("👆 Vui lòng tải lên một ảnh để bắt đầu.")

# 3. Hàm áp dụng bộ lọc Notch-Reject
def notch_reject_filter():
    """UI cho bộ lọc Notch-Reject."""
    st.header("Bộ lọc Notch-Reject")
   
    # Tab cho các chức năng khác nhau
    tab_names = ["Lọc ảnh", "Vẽ bộ lọc"]
    tab_selected = st.radio("Chọn chức năng:", tab_names, horizontal=True)
   
    if tab_selected == "Lọc ảnh":
        # Chọn giữa upload ảnh hoặc dùng ảnh mẫu
        col1, col2 = st.columns([1, 1])
        with col1:
            uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="notch_uploader")
        with col2:
            use_sample = st.checkbox("Sử dụng ảnh mẫu", value=False)
       
        # Xử lý ảnh
        if uploaded_file is not None:
            img = InsertImage(uploaded_file)
        elif use_sample:
            img = InsertImage(None, "3. Bộ lọc Notch.png")
        else:
            st.info("Vui lòng tải lên ảnh hoặc chọn sử dụng ảnh mẫu")
            return
       
        if img is None:
            return
           
        try:
            # Chuyển sang ảnh xám
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
           
            # Tính phổ tần số
            _, magnitude_spectrum = compute_spectrum(img_gray)
           
            # Hiển thị ảnh gốc và phổ tần số
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(magnitude_spectrum, caption="Phổ tần số", use_container_width=True)
           
            # UI cho việc tự động phát hiện điểm notch
            auto_detect = st.checkbox("Tự động phát hiện điểm nhiễu", value=True)
           
            # Kích thước ảnh và tọa độ tâm
            rows, cols = img_gray.shape
            crow, ccol = rows // 2, cols // 2
           
            if auto_detect:
                # Tham số cho việc phát hiện tự động
                col1, col2 = st.columns(2)
                with col1:
                    detection_threshold = st.slider("Ngưỡng phát hiện", 50, 250, 200)
                    min_distance = st.slider("Khoảng cách tối thiểu từ tâm", 5, 100, 20)
                with col2:
                    max_points = st.slider("Số điểm notch tối đa", 1, 10, 5)
               
                # Tìm các điểm có giá trị phổ cao
                high_value_points = np.where(magnitude_spectrum > detection_threshold)
               
                # Lọc và sắp xếp các điểm
                notch_points = []
                for y, x in zip(high_value_points[0], high_value_points[1]):
                    dy, dx = y - crow, x - ccol
                    distance = np.sqrt(dy**2 + dx**2)
                    if distance >= min_distance:
                        notch_points.append((dy, dx))
               
                # Sắp xếp theo khoảng cách và lấy các điểm xa nhất
                notch_points.sort(key=lambda p: p[0]**2 + p[1]**2, reverse=True)
                notch_points = notch_points[:max_points]
               
                # Vẽ các điểm đã phát hiện
                spectrum_with_points = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2RGB)
                for dy, dx in notch_points:
                    y, x = crow + dy, ccol + dx
                    cv2.circle(spectrum_with_points, (x, y), 5, (0, 0, 255), -1)
                    cv2.circle(spectrum_with_points, (2*ccol - x, 2*crow - y), 5, (0, 255, 0), -1)
               
                st.image(spectrum_with_points, caption="Phổ với các điểm notch đã phát hiện", use_container_width=True)
               
            else:
                # UI cho việc chọn điểm thủ công
                num_notches = st.number_input("Số lượng cặp điểm notch", min_value=1, max_value=5, value=1)
               
                notch_points = []
                for i in range(num_notches):
                    col1, col2 = st.columns(2)
                    with col1:
                        y_offset = st.slider(f"Notch {i+1}: Y-offset từ tâm", -crow, crow, 0, key=f"y_offset_{i}")
                    with col2:
                        x_offset = st.slider(f"Notch {i+1}: X-offset từ tâm", -ccol, ccol, 0, key=f"x_offset_{i}")
                    notch_points.append((y_offset, x_offset))
           
            # Tham số cho bộ lọc
            d0 = st.slider("Bán kính bộ lọc Notch", 1, 50, 10)
           
            # Áp dụng bộ lọc khi người dùng nhấn nút
            if st.button("Áp dụng bộ lọc"):
                # Áp dụng bộ lọc
                mask, filtered_image = apply_notch_reject_filter(img_gray, notch_points, d0)
               
                if filtered_image is not None:
                    # Tính phổ của ảnh sau khi lọc
                    _, spectrum_filtered = compute_spectrum(filtered_image)
                   
                    # Hiển thị kết quả trong tabs
                    result_tabs = st.tabs(["Ảnh kết quả", "Phổ tần số"])
                   
                    with result_tabs[0]:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img_gray, caption="Ảnh gốc", use_container_width=True)
                        with col2:
                            st.image(filtered_image, caption="Ảnh sau khi lọc", use_container_width=True)
                   
                    with result_tabs[1]:
                        col1, col2 = st.columns(2)
                        with col1:
                            mask_display = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            st.image(mask_display, caption="Mặt nạ bộ lọc", use_container_width=True)
                        with col2:
                            st.image(spectrum_filtered, caption="Phổ sau khi lọc", use_container_width=True)
                   
                    # So sánh chi tiết
                    if st.checkbox("Hiển thị so sánh chi tiết"):
                        size = min(rows, cols) // 4
                        crop_original = img_gray[crow-size:crow+size, ccol-size:ccol+size]
                        crop_filtered = filtered_image[crow-size:crow+size, ccol-size:ccol+size]
                       
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(crop_original, caption="Chi tiết ảnh gốc", use_container_width=True)
                        with col2:
                            st.image(crop_filtered, caption="Chi tiết ảnh đã lọc", use_container_width=True)
                   
                    # Lưu kết quả
                    if st.button("Lưu kết quả"):
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"notch_filtered_{timestamp}.png"
                        save_processed_image(filtered_image, filename)
                        st.success(f"Đã lưu ảnh tại: {os.path.join(SAVE_PATH_PROCESSED, filename)}")
       
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
   
    elif tab_selected == "Vẽ bộ lọc":
        # Sử dụng giao diện riêng cho vẽ bộ lọc
        draw_notch_filter_ui()


def CreateNotchRejectFilter(P=250, Q=180, D0=10, n=2, notch_points=None):
    """
    Tạo bộ lọc Notch-Reject dựa trên các điểm notch đã cho.
    
    Args:
        P, Q: Kích thước bộ lọc (mặc định 250x180)
        D0: Bán kính của bộ lọc (mặc định 10)
        n: Bậc của bộ lọc (mặc định 2)
        notch_points: Danh sách các điểm notch (u, v) (mặc định là None và sẽ sử dụng các điểm cố định)
        
    Returns:
        H: Mặt nạ bộ lọc
    """
    # Sử dụng các điểm mặc định nếu không có điểm nào được cung cấp
    if notch_points is None:
        notch_points = [
            (44, 58),
            (40, 119),
            (86, 59),
            (82, 119)
        ]
    
    H = np.ones((P, Q), np.float32)
    
    for u in range(0, P):
        for v in range(0, Q):
            h = 1.0
            
            # Áp dụng bộ lọc cho mỗi điểm notch
            for u_notch, v_notch in notch_points:
                # Bộ lọc cho điểm gốc
                Duv = np.sqrt((u-u_notch)**2 + (v-v_notch)**2)
                if Duv > 0:
                    h = h*1.0/(1.0 + np.power(D0/Duv, 2*n))
                else:
                    h = h*0.0
                
                # Bộ lọc cho điểm đối xứng
                Duv = np.sqrt((u-(P-u_notch))**2 + (v-(Q-v_notch))**2)
                if Duv > 0:
                    h = h*1.0/(1.0 + np.power(D0/Duv, 2*n))
                else:
                    h = h*0.0
                    
            H[u,v] = h
            
    return H


def DrawNotchRejectFilter(H=None, L=128):
    """
    Trực quan hóa bộ lọc Notch-Reject.
    
    Args:
        H: Mặt nạ bộ lọc (nếu None, sẽ tạo mặt nạ mới)
        L: Giá trị cường độ tối đa (độ tương phản) (mặc định 128)
        
    Returns:
        filter_visualization: Hình ảnh trực quan của bộ lọc
    """
    # Tạo bộ lọc nếu chưa được cung cấp
    if H is None:
        H = CreateNotchRejectFilter()
        
    # Chuyển đổi bộ lọc thành hình ảnh có thể hiển thị
    H_display = H*(L-1)
    H_display = H_display.astype(np.uint8)
    
    # Tạo hình ảnh màu để trực quan hóa tốt hơn
    filter_color = cv2.applyColorMap(H_display, cv2.COLORMAP_JET)
    
    return filter_color


def draw_notch_filter_ui():
    """UI chuyên biệt cho việc vẽ bộ lọc Notch-Reject."""
    st.subheader("Vẽ bộ lọc Notch-Reject")
   
    # Tham số cho vẽ bộ lọc
    col1, col2 = st.columns(2)
    with col1:
        P = st.slider("Chiều cao bộ lọc", 100, 500, 250, step=10)
        Q = st.slider("Chiều rộng bộ lọc", 100, 500, 180, step=10)
        D0 = st.slider("Bán kính bộ lọc", 1, 50, 10, key="draw_d0")
    with col2:
        n = st.slider("Bậc của bộ lọc", 1, 10, 2)
        L = st.slider("Độ tương phản", 1, 255, 128)
   
    # UI cho chọn điểm notch
    st.subheader("Cấu hình điểm Notch")
    use_custom_points = st.checkbox("Tùy chỉnh điểm Notch", value=False)
    
    if use_custom_points:
        num_notches = st.number_input("Số lượng cặp điểm notch", min_value=1, max_value=5, value=2, key="draw_num_notches")
        
        notch_points = []
        for i in range(num_notches):
            col1, col2 = st.columns(2)
            with col1:
                u = st.slider(f"Notch {i+1}: U", 0, P-1, min(44 + i*10, P-1), key=f"draw_u_{i}")
            with col2:
                v = st.slider(f"Notch {i+1}: V", 0, Q-1, min(58 + i*10, Q-1), key=f"draw_v_{i}")
            notch_points.append((u, v))
    else:
        # Sử dụng các điểm mẫu cố định
        notch_points = [
            (44, 58),
            (40, 119),
            (86, 59),
            (82, 119)
        ]
        # Hiển thị các điểm mẫu
        st.info("Sử dụng các điểm Notch mẫu:")
        for i, (u, v) in enumerate(notch_points):
            st.write(f"Notch {i+1}: U={u}, V={v}")
   
    # Vẽ bộ lọc khi người dùng nhấn nút
    if st.button("Vẽ bộ lọc"):
        with st.spinner("Đang vẽ bộ lọc..."):
            # Vẽ bộ lọc sử dụng các thông số đã chọn
            H = CreateNotchRejectFilter(P, Q, D0, n, notch_points)
            filter_visualization = DrawNotchRejectFilter(H, L)
           
            if filter_visualization is not None:
                # Hiển thị kết quả
                st.image(filter_visualization, caption="Bộ lọc Notch-Reject", use_container_width=True)
               
                # Thông tin về các điểm notch
                st.subheader("Thông tin bộ lọc")
                st.write(f"Kích thước bộ lọc: {P}x{Q}")
                st.write(f"Bán kính bộ lọc (D0): {D0}")
                st.write(f"Bậc của bộ lọc (n): {n}")
                st.write(f"Số lượng cặp điểm notch: {len(notch_points)}")
               
                # Hiển thị thông tin chi tiết về các điểm
                st.write("Chi tiết điểm notch:")
                for i, (u, v) in enumerate(notch_points):
                    st.write(f"- Notch {i+1}: U={u}, V={v}")
               
                # Lưu kết quả
                if st.button("Lưu bộ lọc"):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"notch_filter_{timestamp}.png"
                    save_processed_image(filter_visualization, filename)
                    st.success(f"Đã lưu bộ lọc tại: {os.path.join(SAVE_PATH_PROCESSED, filename)}")


def apply_notch_reject_filter(img, notch_points, D0, n=2):
    """
    Áp dụng bộ lọc Notch-Reject lên ảnh.
    
    Args:
        img: Ảnh đầu vào
        notch_points: Danh sách các điểm notch (y_offset, x_offset) từ tâm
        D0: Bán kính bộ lọc
        n: Bậc của bộ lọc (mặc định là 2)
        
    Returns:
        mask: Mặt nạ bộ lọc
        filtered_image: Ảnh sau khi lọc
    """
    # Lấy kích thước ảnh
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # Tạo DFT
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tạo mặt nạ với tất cả các giá trị là 1
    mask = np.ones((rows, cols, 2), np.float32)
    
    # Áp dụng bộ lọc notch-reject cho từng điểm
    for y_offset, x_offset in notch_points:
        # Tọa độ điểm notch
        y, x = crow + y_offset, ccol + x_offset
        y_sym, x_sym = crow - y_offset, ccol - x_offset
        
        # Áp dụng bộ lọc Butterworth notch-reject
        for u in range(rows):
            for v in range(cols):
                # Tính khoảng cách từ điểm (u,v) đến điểm notch
                Duv1 = np.sqrt((u-y)**2 + (v-x)**2)
                Duv2 = np.sqrt((u-y_sym)**2 + (v-x_sym)**2)
                
                # Áp dụng công thức bộ lọc Butterworth
                if Duv1 > 0:
                    mask[u,v] *= 1.0/(1.0 + np.power(D0/Duv1, 2*n))
                else:
                    mask[u,v] *= 0.0
                    
                if Duv2 > 0:
                    mask[u,v] *= 1.0/(1.0 + np.power(D0/Duv2, 2*n))
                else:
                    mask[u,v] *= 0.0
    
    # Áp dụng mặt nạ và thực hiện DFT ngược
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    # Chuẩn hóa ảnh kết quả
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return mask[:,:,0], img_back

# 4. Hàm xóa nhiễu Moire
def remove_moire(image, r_low=30, r_high=None, strength=1.0):
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    
    # Chuyển sang miền tần số
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Tạo mặt nạ cho lọc thông thấp (giữ tần số thấp)
    mask = np.ones((rows, cols, 2), np.float32)
    
    # Tạo lưới tọa độ
    u = np.arange(rows)
    v = np.arange(cols)
    u, v = np.meshgrid(u - crow, v - ccol, indexing='ij')
    d = np.sqrt(u**2 + v**2)
    
    # Áp dụng lọc thông thấp
    if r_high is None:
        # Áp dụng Butterworth lowpass filter
        n = 2  # Order của bộ lọc Butterworth
        filter_low = 1 / (1 + (d / r_low)**(2*n))
    else:
        # Áp dụng band-reject filter
        filter_low = 1 - np.exp(-0.5 * ((d**2 - r_low**2) / (d * r_high))**2)
    
    # Áp dụng mặt nạ
    mask[:, :, 0] = mask[:, :, 0] * filter_low
    mask[:, :, 1] = mask[:, :, 1] * filter_low
    
    # Kết hợp mặt nạ với mức độ mạnh yếu
    blended_mask = np.ones((rows, cols, 2), np.float32) * (1 - strength) + mask * strength
    
    # Áp dụng mặt nạ vào ảnh trong miền tần số
    fshift_filtered = dft_shift * blended_mask
    
    # Chuyển về miền không gian
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    # Chuẩn hóa kết quả
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return filter_low, img_back

def remove_moire_ui():
    """UI cho xóa nhiễu Moire."""
    st.header("Xóa nhiễu Moire")
    
    uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'png', 'jpeg'], key="moire_uploader")
    
    if uploaded_file is not None:
        try:
            # Đọc ảnh
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Kiểm tra ảnh màu và định dạng
            is_color = len(img_array.shape) == 3 and img_array.shape[2] >= 3
            
            # Tham số cho bộ lọc
            r_low = st.slider("Bán kính tần số thấp (giữ lại)", 10, 100, 30)
            filter_type = st.radio("Loại bộ lọc", ["Lowpass", "Band-reject"])
            
            r_high = None
            if filter_type == "Band-reject":
                r_high = st.slider("Độ rộng băng tần reject", 5, 50, 20)
            
            strength = st.slider("Mức độ lọc", 0.0, 1.0, 0.8)
            
            # Xử lý ảnh màu hoặc grayscale
            if is_color:
                # Chuyển đổi ảnh sang đúng định dạng
                if img_array.shape[2] == 4:  # Nếu có kênh alpha
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                
                # Tách và xử lý từng kênh màu
                channels = cv2.split(img_array)
                filtered_channels = []
                masks = []
                
                for channel in channels:
                    mask, filtered_channel = remove_moire(channel, r_low, r_high, strength)
                    filtered_channels.append(filtered_channel)
                    masks.append(mask)
                
                # Ghép các kênh màu đã xử lý
                filtered_image = cv2.merge(filtered_channels)
                mask = masks[0]  # Sử dụng mặt nạ của kênh đầu tiên để hiển thị
            else:
                # Xử lý ảnh grayscale
                img_gray = img_array
                mask, filtered_image = remove_moire(img_gray, r_low, r_high, strength)
            
            # Hiển thị kết quả
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Ảnh gốc", use_container_width=True)
                st.image(mask * 255, caption="Mặt nạ bộ lọc", use_container_width=True, clamp=True)
            
            with col2:
                st.image(filtered_image, caption="Ảnh sau khi xóa nhiễu Moire", use_container_width=True)
                
                # Phần hiển thị chi tiết phóng to
                if st.checkbox("Hiển thị phóng to chi tiết"):
                    zoom_factor = st.slider("Độ phóng đại", 1, 5, 2)
                    height, width = img_array.shape[:2]
                    center_y, center_x = height // 2, width // 2
                    crop_size = min(height, width) // (zoom_factor * 2)
                    
                    # Cắt vùng trung tâm của ảnh gốc và ảnh đã xử lý
                    crop_original = img_array[center_y-crop_size:center_y+crop_size,
                                           center_x-crop_size:center_x+crop_size]
                    crop_filtered = filtered_image[center_y-crop_size:center_y+crop_size,
                                                center_x-crop_size:center_x+crop_size]
                    
                    st.image(crop_original, caption="Vùng trung tâm ảnh gốc", use_container_width=True)
                    st.image(crop_filtered, caption="Vùng trung tâm ảnh đã lọc", use_container_width=True)
                    
        except Exception as e:
            st.error(f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}")
            st.error("Vui lòng thử lại với ảnh khác hoặc kiểm tra định dạng ảnh")


# Dictionary ánh xạ chức năng
chapter_04_functions = {
    "Spectrum": spectrum,
    "Lọc trong miền tần số": frequency_domain_filter,
    "Bộ lọc notch-reject": notch_reject_filter, 
    "Xóa nhiễu Moire": remove_moire_ui,
}