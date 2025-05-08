import streamlit as st
import os
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Đường dẫn lưu ảnh gốc và ảnh sau xử lý
SAVE_PATH_ORIGINAL = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C03"
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

# 1. Negative Image
def negative_image():
    """Chuyển đổi ảnh thành âm bản (negative)."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="negative_image")
    
    # Chọn chế độ xử lý
    processing_mode = st.radio(
        "Chọn chế độ xử lý:",
        ("Ảnh màu", "Ảnh trắng đen"),
        help="Ảnh màu sẽ xử lý riêng từng kênh RGB. Ảnh trắng đen sẽ chuyển đổi sang grayscale trước khi xử lý."
    )
    
    try:
        # Xử lý ảnh
        if uploaded_file is not None:
            img = InsertImage(uploaded_file)
            if img is None:
                st.error("Không thể đọc ảnh đã tải lên")
                return
        else:
            img = InsertImage(None, "1. NegativeImage.png")
            if img is None:
                st.error("Không thể đọc ảnh mặc định")
                return

        # Xử lý theo chế độ đã chọn
        if processing_mode == "Ảnh trắng đen":
            # Chuyển sang ảnh grayscale nếu là ảnh màu
            if len(img.shape) == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img
                
            # Tạo ảnh âm bản
            negative = cv2.bitwise_not(img_gray)
            
            # Hiển thị histogram trước và sau xử lý
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.hist(img_gray.ravel(), 256, [0, 256], color='gray')
            ax1.set_title('Histogram ảnh gốc')
            ax1.set_xlabel('Giá trị pixel')
            ax1.set_ylabel('Số lượng pixel')
            
            ax2.hist(negative.ravel(), 256, [0, 256], color='gray')
            ax2.set_title('Histogram ảnh âm bản')
            ax2.set_xlabel('Giá trị pixel')
            ax2.set_ylabel('Số lượng pixel')
            
            plt.tight_layout()
            st.pyplot(fig)
            
        else:  # Ảnh màu
            # Đảm bảo ảnh là ảnh màu
            if len(img.shape) != 3:
                st.error("Vui lòng sử dụng ảnh màu cho chế độ này")
                return
                
            # Tạo ảnh âm bản
            negative = cv2.bitwise_not(img)
            
            # Hiển thị histogram cho từng kênh màu
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            colors = ['red', 'green', 'blue']
            
            for i, color in enumerate(colors):
                # Histogram ảnh gốc
                axes[0, i].hist(img[:,:,i].ravel(), 256, [0, 256], color=color, alpha=0.7)
                axes[0, i].set_title(f'Histogram kênh {color} (gốc)')
                axes[0, i].set_xlabel('Giá trị pixel')
                axes[0, i].set_ylabel('Số lượng pixel')
                
                # Histogram ảnh âm bản
                axes[1, i].hist(negative[:,:,i].ravel(), 256, [0, 256], color=color, alpha=0.7)
                axes[1, i].set_title(f'Histogram kênh {color} (âm bản)')
                axes[1, i].set_xlabel('Giá trị pixel')
                axes[1, i].set_ylabel('Số lượng pixel')
            
            plt.tight_layout()
            st.pyplot(fig)

        # Lưu ảnh đã xử lý
        processed_path = save_processed_image(negative, f"1. NegativeImage_{processing_mode}_processed.png")
        
        # Hiển thị ảnh gốc và ảnh đã xử lý
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, caption="Ảnh gốc", use_container_width=True)
            
        with col2:
            st.image(negative, caption="Ảnh âm bản", use_container_width=True)
        
        # Thêm nút tải xuống ảnh đã xử lý
        if os.path.exists(processed_path):
            with open(processed_path, 'rb') as file:
                btn = st.download_button(
                    label="Tải xuống ảnh đã xử lý",
                    data=file,
                    file_name=f"negative_image_{processing_mode}_processed.png",
                    mime="image/png"
                )
                
    except Exception as e:
        st.error(f"Có lỗi xảy ra: {str(e)}")
        st.error("Vui lòng thử lại với ảnh khác")
        
# 2. Logarithmic Transform
def logarit_image():
    """Biến đổi logarit để tăng độ sáng cho các vùng tối."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="logarit_image")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "2. LogaritImage.png")
    
    if img is None:
        return
    
    # Chuyển sang float để tính toán
    img_float = img.astype(float)
    
    # Hệ số c để điều chỉnh phạm vi đầu ra
    c = 255 / np.log(1 + np.max(img_float))
    
    # Phép biến đổi logarit
    log_transformed = c * np.log(1 + img_float)
    
    # Chuyển về uint8 để hiển thị
    log_transformed = np.uint8(log_transformed)
    
    # Lưu ảnh đã xử lý
    processed_path = save_processed_image(log_transformed, "2. LogaritImage_processed.png")
    
    # Hiển thị ảnh
    if uploaded_file is not None:
        display_images(img, log_transformed)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "2. LogaritImage.png")
        display_images(original_path, processed_path)

# 3. Power-Law (Gamma) Transform
def luy_thua_image():
    """Biến đổi lũy thừa (gamma) để điều chỉnh độ tương phản."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="luy_thua_image")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "3. LuyThua.png")
    
    if img is None:
        return
    
    # Thêm thanh trượt để điều chỉnh gamma
    gamma = st.slider("Gamma (γ)", 0.1, 5.0, 1.0, 0.1)
    
    # Chuyển sang float để tính toán
    img_float = img.astype(float) / 255.0
    
    # Phép biến đổi gamma (lũy thừa)
    gamma_transformed = np.power(img_float, gamma)
    
    # Chuyển về uint8 để hiển thị
    gamma_transformed = np.uint8(gamma_transformed * 255)
    
    # Lưu ảnh đã xử lý
    processed_path = save_processed_image(gamma_transformed, f"3. LuyThua_gamma_{gamma}_processed.png")
    
    # Hiển thị ảnh
    if uploaded_file is not None:
        display_images(img, gamma_transformed)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "3. LuyThua.png")
        display_images(original_path, processed_path)

# 4. Piecewise-Linear Transform
def bien_doi_tuyen_tinh():
    """Biến đổi tuyến tính từng phần để cải thiện độ tương phản."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="bien_doi_tuyen_tinh")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "4. BienDoiTuyenTinhTungPhan.png")
    
    if img is None:
        return
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Thêm các thanh trượt để điều chỉnh các điểm xác định
    st.subheader("Điều chỉnh các điểm biến đổi")
    r_min = st.slider("r_min", 0, 255, 50, 1)
    r_max = st.slider("r_max", r_min, 255, 200, 1)
    s_min = st.slider("s_min", 0, 255, 0, 1)
    s_max = st.slider("s_max", s_min, 255, 255, 1)
    
    # Tạo mảng lookup để ánh xạ giá trị
    lut = np.zeros(256, dtype=np.uint8)
    
    # Phần 1: từ 0 đến r_min
    for i in range(0, r_min + 1):
        lut[i] = s_min * i / r_min if r_min > 0 else s_min
    
    # Phần 2: từ r_min đến r_max
    for i in range(r_min + 1, r_max + 1):
        lut[i] = s_min + (s_max - s_min) * (i - r_min) / (r_max - r_min) if r_max > r_min else s_max
    
    # Phần 3: từ r_max đến 255
    for i in range(r_max + 1, 256):
        lut[i] = s_max + (255 - s_max) * (i - r_max) / (255 - r_max) if 255 > r_max else 255
    
    # Áp dụng biến đổi
    result = cv2.LUT(img_gray, lut)
    
    # Hiển thị đồ thị ánh xạ
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(256), lut, 'b-')
    ax.plot([0, r_min, r_max, 255], [0, s_min, s_max, 255], 'ro')
    ax.grid(True)
    ax.set_title('Biến đổi tuyến tính từng phần')
    ax.set_xlabel('Giá trị đầu vào (r)')
    ax.set_ylabel('Giá trị đầu ra (s)')
    ax.set_xlim([0, 255])
    ax.set_ylim([0, 255])
    st.pyplot(fig)
    
    # Lưu ảnh đã xử lý
    processed_path = save_processed_image(result, "4. BienDoiTuyenTinhTungPhan_processed.png")
    
    # Hiển thị ảnh
    if uploaded_file is not None:
        display_images(img, result)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "4. BienDoiTuyenTinhTungPhan.png")
        display_images(original_path, processed_path)

# 5. Histogram
def Logarit(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c*np.log(1.0 + r)
            imgout[x, y] = np.uint8(s)
    return imgout

def LogaritColor(imgin):
    M, N, C = imgin.shape
    imgout = np.zeros((M, N, C), np.uint8)
    c = (L-1)/np.log(L)
    for x in range(0, M):
        for y in range(0, N):
            r = imgin[x, y, 2]
            g = imgin[x, y, 1]
            b = imgin[x, y, 0]  # OpenCV uses BGR
            if r == 0:
                r = 1
            if g == 0:
                g = 1
            if b == 0:
                b = 1
            r = c*np.log(1.0 + r)
            g = c*np.log(1.0 + g)
            b = c*np.log(1.0 + b)
            imgout[x, y, 2] = np.uint8(r)
            imgout[x, y, 1] = np.uint8(g)
            imgout[x, y, 0] = np.uint8(b)
    return imgout

def histogram():
    """Xử lý ảnh để làm nổi bật các thành phần bị ẩn."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="histogram")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "5. Histogram.png")
    
    if img is None:
        st.error("Không thể đọc ảnh. Vui lòng thử lại.")
        return
    
    # Xử lý ảnh để làm nổi bật chi tiết ẩn
    if img.ndim == 3:  # Ảnh màu
        img_processed = LogaritColor(img)
    else:  # Ảnh grayscale
        img_processed = Logarit(img)
    
    # Hiển thị ảnh gốc và ảnh đã xử lý
    st.header("Kết quả xử lý ảnh")
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Ảnh Gốc", width=400)
    with col2:
        st.image(img_processed, caption="Ảnh sau khi xử lý (Logarit)", width=400)
    
    # Hiển thị histogram của ảnh đã xử lý (nếu cần)
    st.subheader("Histogram của ảnh đã xử lý")
    if img_processed.ndim == 3:
        # Vẽ histogram cho từng kênh màu
        fig, axs = plt.subplots(2, 2, figsize=(6, 4))  # Thu nhỏ biểu đồ
        img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
        
        # Histogram cho ảnh grayscale
        axs[0, 0].hist(img_gray.ravel(), 256, [0, 256], color='black')
        axs[0, 0].set_title('Grayscale Histogram')
        axs[0, 0].set_xlabel('Giá trị pixel')
        axs[0, 0].set_ylabel('Số lượng pixel')
        
        # Histogram cho kênh đỏ
        axs[0, 1].hist(img_processed[:, :, 2].ravel(), 256, [0, 256], color='red')
        axs[0, 1].set_title('Red Channel Histogram')
        axs[0, 1].set_xlabel('Giá trị pixel')
        axs[0, 1].set_ylabel('Số lượng pixel')
        
        # Histogram cho kênh xanh lá
        axs[1, 0].hist(img_processed[:, :, 1].ravel(), 256, [0, 256], color='green')
        axs[1, 0].set_title('Green Channel Histogram')
        axs[1, 0].set_xlabel('Giá trị pixel')
        axs[1, 0].set_ylabel('Số lượng pixel')
        
        # Histogram cho kênh xanh dương
        axs[1, 1].hist(img_processed[:, :, 0].ravel(), 256, [0, 256], color='blue')
        axs[1, 1].set_title('Blue Channel Histogram')
        axs[1, 1].set_xlabel('Giá trị pixel')
        axs[1, 1].set_ylabel('Số lượng pixel')
        
        plt.tight_layout()
    else:
        # Vẽ histogram cho ảnh grayscale
        fig, ax = plt.subplots(figsize=(5, 2))  # Thu nhỏ biểu đồ
        ax.hist(img_processed.ravel(), 256, [0, 256], color='black')
        ax.set_title('Grayscale Histogram')
        ax.set_xlabel('Giá trị pixel')
        ax.set_ylabel('Số lượng pixel')
    
    st.pyplot(fig)

# 6. Histogram Equalization
def can_bang_histogram():
    """Cân bằng histogram để cải thiện độ tương phản."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="can_bang_histogram")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "6. CanBangHistogram.png")
    
    if img is None:
        return
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Cân bằng histogram
    equalized = cv2.equalizeHist(img_gray)
    
    # Vẽ histogram trước và sau khi cân bằng
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram trước khi cân bằng
    axs[0, 0].hist(img_gray.ravel(), 256, [0, 256], color='black')
    axs[0, 0].set_title('Histogram trước khi cân bằng')
    axs[0, 0].set_xlabel('Giá trị pixel')
    axs[0, 0].set_ylabel('Số lượng pixel')
    
    # Histogram sau khi cân bằng
    axs[0, 1].hist(equalized.ravel(), 256, [0, 256], color='black')
    axs[0, 1].set_title('Histogram sau khi cân bằng')
    axs[0, 1].set_xlabel('Giá trị pixel')
    axs[0, 1].set_ylabel('Số lượng pixel')
    
    # Vẽ CDF (Cumulative Distribution Function)
    hist_original = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    cdf_original = hist_original.cumsum()
    cdf_original_normalized = cdf_original * hist_original.max() / cdf_original.max()
    
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    cdf_equalized = hist_equalized.cumsum()
    cdf_equalized_normalized = cdf_equalized * hist_equalized.max() / cdf_equalized.max()
    
    axs[1, 0].plot(cdf_original_normalized, color='black')
    axs[1, 0].set_title('CDF trước khi cân bằng')
    axs[1, 0].set_xlabel('Giá trị pixel')
    axs[1, 0].set_ylabel('CDF')
    
    axs[1, 1].plot(cdf_equalized_normalized, color='black')
    axs[1, 1].set_title('CDF sau khi cân bằng')
    axs[1, 1].set_xlabel('Giá trị pixel')
    axs[1, 1].set_ylabel('CDF')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(equalized, "6. CanBangHistogram_processed.png")
    
    if uploaded_file is not None:
        display_images(img, equalized)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "6. CanBangHistogram.png")
        display_images(original_path, processed_path)

# 7. Color Histogram Equalization
def can_bang_histogram_anh_mau():
    """Cân bằng histogram cho ảnh màu."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="can_bang_histogram_mau")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "7. Cân bằng Histogram của ảnh màu.jpg")
    
    if img is None:
        return
    
    # Kiểm tra xem có phải ảnh màu không
    if img.ndim != 3:
        st.error("Cần ảnh màu cho chức năng này!")
        return
    
    # Phương pháp cân bằng histogram
    method = st.radio(
        "Chọn phương pháp cân bằng histogram:",
        ("Cân bằng từng kênh RGB", "Chuyển sang HSV và cân bằng kênh V")
    )
    
    if method == "Cân bằng từng kênh RGB":
        # Tách các kênh màu
        r, g, b = cv2.split(img)
        
        # Cân bằng histogram cho từng kênh
        r_eq = cv2.equalizeHist(r)
        g_eq = cv2.equalizeHist(g)
        b_eq = cv2.equalizeHist(b)
        
        # Ghép các kênh lại
        equalized = cv2.merge((r_eq, g_eq, b_eq))
    else:
        # Chuyển sang không gian màu HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Tách các kênh
        h, s, v = cv2.split(hsv)
        
        # Cân bằng histogram cho kênh V (giá trị)
        v_eq = cv2.equalizeHist(v)
        
        # Ghép các kênh lại
        hsv_eq = cv2.merge((h, s, v_eq))
        
        # Chuyển lại thành RGB
        equalized = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)
    
    # Vẽ histogram cho ảnh gốc và ảnh đã cân bằng
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Histogram cho ảnh gốc
    colors = ('r', 'g', 'b')
    for i, color in enumerate(colors):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        axs[0, i].plot(hist, color=color)
        axs[0, i].set_title(f'Original {color.upper()} Channel')
        axs[0, i].set_xlim([0, 256])
    
    # Histogram cho ảnh đã cân bằng
    for i, color in enumerate(colors):
        hist = cv2.calcHist([equalized], [i], None, [256], [0, 256])
        axs[1, i].plot(hist, color=color)
        axs[1, i].set_title(f'Equalized {color.upper()} Channel')
        axs[1, i].set_xlim([0, 256])
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(equalized, "7. CanBangHistogramAnhMau_processed.png")
    
    if uploaded_file is not None:
        display_images(img, equalized)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "7. Cân bằng Histogram của ảnh màu.jpg")
        display_images(original_path, processed_path)

# 8. Local Histogram Equalization
def local_histogram():
    """Cân bằng histogram cục bộ để cải thiện chi tiết cục bộ."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="local_histogram")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "8. LocalHistogram.png")
    
    if img is None:
        return
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Kích thước cửa sổ cho cân bằng histogram cục bộ
    block_size = st.slider("Kích thước cửa sổ (kích thước lẻ)", 3, 99, 11, 2)
    
    # Đảm bảo block_size là số lẻ
    if block_size % 2 == 0:
        block_size += 1
    
    # Giá trị clip limit để hạn chế nhiễu
    clip_limit = st.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
    
    # Sử dụng CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(block_size, block_size))
    equalized = clahe.apply(img_gray)
    
    # Vẽ histogram trước và sau khi cân bằng
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Histogram trước khi cân bằng
    axs[0].hist(img_gray.ravel(), 256, [0, 256], color='black')
    axs[0].set_title('Histogram trước khi cân bằng cục bộ')
    axs[0].set_xlabel('Giá trị pixel')
    axs[0].set_ylabel('Số lượng pixel')
    
    # Histogram sau khi cân bằng
    axs[1].hist(equalized.ravel(), 256, [0, 256], color='black')
    axs[1].set_title('Histogram sau khi cân bằng cục bộ')
    axs[1].set_xlabel('Giá trị pixel')
    axs[1].set_ylabel('Số lượng pixel')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(equalized, "8. LocalHistogram_processed.png")
    
    if uploaded_file is not None:
        display_images(img, equalized)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "8. LocalHistogram.png")
        display_images(original_path, processed_path)

# 9. Histogram Statistics
def thong_ke_histogram():
    """Hiển thị thống kê histogram của ảnh."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="thong_ke_histogram")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "9. ThongKeHistogram.png")
    
    if img is None:
        return
    
    # Hiển thị ảnh gốc trước
    st.header("Ảnh Gốc")
    if uploaded_file is not None:
        st.image(img, caption="Ảnh Gốc", width=400)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "9. ThongKeHistogram.png")
        st.image(original_path, caption="Ảnh Gốc", width=400)
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Tính thống kê cho từng kênh màu
        channels = cv2.split(img)
        channel_names = ['Red', 'Green', 'Blue']
        
        st.subheader("Thống kê và Histogram cho từng kênh màu")
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            mean_val = np.mean(channel)
            median_val = np.median(channel)
            std_val = np.std(channel)
            min_val = np.min(channel)
            max_val = np.max(channel)
            
            st.write(f"**Kênh {name}:**")
            st.write(f"- Giá trị trung bình: {mean_val:.2f}")
            st.write(f"- Giá trị trung vị: {median_val:.2f}")
            st.write(f"- Độ lệch chuẩn: {std_val:.2f}")
            st.write(f"- Giá trị nhỏ nhất: {min_val}")
            st.write(f"- Giá trị lớn nhất: {max_val}")
            
            # Vẽ histogram
            fig, ax = plt.subplots(figsize=(5, 2))  # Thu nhỏ biểu đồ
            ax.hist(channel.ravel(), 256, [0, 256], color=name.lower())
            ax.set_title(f'Histogram kênh {name}')
            ax.set_xlabel('Giá trị pixel')
            ax.set_ylabel('Số lượng pixel')
            st.pyplot(fig)
    else:
        # Tính thống kê cho ảnh grayscale
        mean_val = np.mean(img)
        median_val = np.median(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        st.subheader("Thống kê và Histogram cho ảnh grayscale")
        st.write(f"- Giá trị trung bình: {mean_val:.2f}")
        st.write(f"- Giá trị trung vị: {median_val:.2f}")
        st.write(f"- Độ lệch chuẩn: {std_val:.2f}")
        st.write(f"- Giá trị nhỏ nhất: {min_val}")
        st.write(f"- Giá trị lớn nhất: {max_val}")
        
        # Vẽ histogram
        fig, ax = plt.subplots(figsize=(5, 2))  # Thu nhỏ biểu đồ
        ax.hist(img.ravel(), 256, [0, 256], color='gray')
        ax.set_title('Histogram ảnh grayscale')
        ax.set_xlabel('Giá trị pixel')
        ax.set_ylabel('Số lượng pixel')
        st.pyplot(fig)

# 10. Box Filter
def loc_box():
    """Áp dụng bộ lọc hộp (Box Filter) để làm mờ ảnh."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="loc_box")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "10. LocBox.png")
    
    if img is None:
        return
    
    # Kích thước kernel
    kernel_size = st.slider("Kích thước kernel", 1, 31, 5, 2)
    
    # Đảm bảo kernel size là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Áp dụng bộ lọc box
    blurred = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
    
    # Lưu ảnh đã xử lý
    processed_path = save_processed_image(blurred, f"10. LocBox_size_{kernel_size}_processed.png")
    
    # Hiển thị ảnh
    if uploaded_file is not None:
        display_images(img, blurred)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "10. LocBox.png")
        display_images(original_path, processed_path)
    
    # Hiển thị ma trận kernel
    st.subheader("Ma trận kernel Box Filter")
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    st.write(f"Shape: {kernel.shape}")
    st.write("Các giá trị trong kernel:")
    st.write(kernel)

# 11. Gaussian Filter
def loc_gauss():
    """Áp dụng bộ lọc Gauss để làm mờ ảnh."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="loc_gauss")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "11. LocGauss.png")
    
    if img is None:
        return
    
    # Kích thước kernel
    kernel_size = st.slider("Kích thước kernel", 1, 31, 5, 2)
    
    # Đảm bảo kernel size là số lẻ
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Sigma (độ lệch chuẩn)
    sigma = st.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
    
    # Áp dụng bộ lọc Gauss
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Lưu ảnh đã xử lý
    processed_path = save_processed_image(blurred, f"11. LocGauss_size_{kernel_size}_sigma_{sigma}_processed.png")
    
    # Hiển thị ảnh
    if uploaded_file is not None:
        display_images(img, blurred)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "11. LocGauss.png")
        display_images(original_path, processed_path)
    
    # Tạo và hiển thị ma trận kernel Gauss
    st.subheader("Ma trận kernel Gaussian Filter")
    # Tạo kernel 1D
    x = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    
    # Công thức Gauss 1D
    kernel_1d = np.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Tạo kernel 2D từ hai kernel 1D
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    
    # Hiển thị kernel
    st.write(f"Shape: {kernel_2d.shape}")
    st.write("Các giá trị trong kernel:")
    st.write(kernel_2d)
    
    # Hiển thị kernel dưới dạng hình ảnh
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(kernel_2d, cmap='viridis')
    ax.set_title('Gaussian Kernel Visualization')
    plt.colorbar(im)
    st.pyplot(fig)

# 12. Thresholding
def phan_nguong():
    """Phân ngưỡng ảnh để phân đoạn đối tượng."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="phan_nguong")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "12. PhanNguong.png")
    
    if img is None:
        return
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Loại phân ngưỡng
    threshold_type = st.selectbox(
        "Chọn phương pháp phân ngưỡng:",
        ("Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted", "Otsu", "Adaptive Mean", "Adaptive Gaussian")
    )
    
    if threshold_type in ["Binary", "Binary Inverted", "Truncate", "To Zero", "To Zero Inverted"]:
        # Ngưỡng
        threshold_value = st.slider("Giá trị ngưỡng", 0, 255, 127)
        
        # Áp dụng phân ngưỡng
        if threshold_type == "Binary":
            _, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif threshold_type == "Binary Inverted":
            _, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
        elif threshold_type == "Truncate":
            _, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TRUNC)
        elif threshold_type == "To Zero":
            _, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TOZERO)
        elif threshold_type == "To Zero Inverted":
            _, thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_TOZERO_INV)
    
    elif threshold_type == "Otsu":
        # Phân ngưỡng Otsu tự động chọn ngưỡng tối ưu
        _, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold_value = _  # Lấy giá trị ngưỡng được chọn
        st.write(f"Giá trị ngưỡng Otsu tự động: {threshold_value}")
    
    else:  # Adaptive thresholding
        block_size = st.slider("Kích thước block", 3, 99, 11, 2)
        # Đảm bảo block_size là số lẻ
        if block_size % 2 == 0:
            block_size += 1
        
        C = st.slider("Hằng số C", -10, 10, 2)
        
        if threshold_type == "Adaptive Mean":
            thresholded = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, block_size, C)
        else:  # "Adaptive Gaussian"
            thresholded = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, block_size, C)
    
    # Vẽ histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(img_gray.ravel(), 256, [0, 256], color='black')
    
    if threshold_type not in ["Adaptive Mean", "Adaptive Gaussian"]:
        # Vẽ đường thẳng tại ngưỡng
        ax.axvline(x=threshold_value, color='r', linestyle='--')
        ax.text(threshold_value + 5, 0, f'Threshold = {threshold_value}', color='r')
    
    ax.set_title('Histogram với ngưỡng')
    ax.set_xlabel('Giá trị pixel')
    ax.set_ylabel('Số lượng pixel')
    st.pyplot(fig)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(thresholded, f"12. PhanNguong_{threshold_type}_processed.png")
    
    if uploaded_file is not None:
        display_images(img, thresholded)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "12. PhanNguong.png")
        display_images(original_path, processed_path)

# 13. Median Filter
def loc_median():
    """Áp dụng bộ lọc trung vị để loại bỏ nhiễu muối tiêu."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="loc_median")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "13. LocMedian.png")
    
    if img is None:
        return
    
    try:
        # Kích thước kernel
        kernel_size = st.slider("Kích thước kernel", 1, 15, 3, 2)
        
        # Đảm bảo kernel size là số lẻ
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Tùy chọn thêm nhiễu muối tiêu cho demo
        add_noise = st.checkbox("Thêm nhiễu muối tiêu vào ảnh gốc")
        
        if add_noise:
            # Tạo bản sao ảnh để thêm nhiễu
            noisy_img = np.copy(img)
            
            # Tỷ lệ nhiễu
            noise_ratio = st.slider("Tỷ lệ nhiễu", 0.0, 0.5, 0.05, 0.01)
            
            # Số pixel nhiễu
            num_pixels = int(noise_ratio * img.shape[0] * img.shape[1])
            
            # Thêm nhiễu muối (trắng)
            for _ in range(num_pixels // 2):
                x = np.random.randint(0, img.shape[0])
                y = np.random.randint(0, img.shape[1])
                if img.ndim == 3:
                    noisy_img[x, y, :] = 255
                else:
                    noisy_img[x, y] = 255
            
            # Thêm nhiễu tiêu (đen)
            for _ in range(num_pixels // 2):
                x = np.random.randint(0, img.shape[0])
                y = np.random.randint(0, img.shape[1])
                if img.ndim == 3:
                    noisy_img[x, y, :] = 0
                else:
                    noisy_img[x, y] = 0
            
            img_to_filter = noisy_img
            save_processed_image(noisy_img, "13. LocMedian_noisy.png")
        else:
            img_to_filter = img
        
        # Áp dụng bộ lọc trung vị
        if img_to_filter.ndim == 3:
            # Lọc từng kênh cho ảnh màu
            channels = cv2.split(img_to_filter)
            filtered_channels = [cv2.medianBlur(channel, kernel_size) for channel in channels]
            filtered = cv2.merge(filtered_channels)
        else:
            # Lọc trực tiếp cho ảnh grayscale
            filtered = cv2.medianBlur(img_to_filter, kernel_size)
        
        # Lưu và hiển thị ảnh
        processed_path = save_processed_image(filtered, f"13. LocMedian_size_{kernel_size}_processed.png")
        
        if uploaded_file is not None:
            display_images(img_to_filter if add_noise else img, filtered, 
                         "Ảnh Gốc (Đã thêm nhiễu)" if add_noise else "Ảnh Gốc")
        else:
            original_path = os.path.join(SAVE_PATH_ORIGINAL, "13. LocMedian.png")
            display_images(img_to_filter if add_noise else original_path, processed_path,
                         "Ảnh Gốc (Đã thêm nhiễu)" if add_noise else "Ảnh Gốc")
            
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi xử lý ảnh: {str(e)}")
        st.error("Vui lòng đảm bảo ảnh đầu vào hợp lệ.")

# 14. Sharpen
def sharpen():
    """Làm sắc nét ảnh bằng các bộ lọc làm nét."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="sharpen")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "14. Sharpen.png")
    
    if img is None:
        return
    
    # Phương pháp làm sắc nét
    sharpen_method = st.selectbox(
        "Chọn phương pháp làm sắc nét:",
        ("Laplacian", "Unsharp Masking")
    )
    
    if sharpen_method == "Laplacian":
        # Chuyển sang grayscale nếu là ảnh màu
        if img.ndim == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img
        
        # Kernel Laplacian
        kernel_type = st.radio(
            "Chọn loại kernel Laplacian:",
            ("4-neighbour", "8-neighbour", "Custom")
        )
        
        if kernel_type == "4-neighbour":
            kernel = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ])
        elif kernel_type == "8-neighbour":
            kernel = np.array([
                [1, 1, 1],
                [1, -8, 1],
                [1, 1, 1]
            ])
        else:  # Custom
            center_value = st.slider("Giá trị trung tâm", -20, 0, -8)
            kernel = np.ones((3, 3))
            kernel[1, 1] = center_value
            st.write("Ma trận kernel tùy chỉnh:")
            st.write(kernel)
        
        # Hệ số tỷ lệ
        scale = st.slider("Hệ số tỷ lệ", 0.1, 5.0, 1.0, 0.1)
        
        # Áp dụng bộ lọc Laplacian
        laplacian = cv2.filter2D(img_gray, -1, kernel)
        
        # Tăng độ tương phản
        sharpened = np.clip(img_gray - scale * laplacian, 0, 255).astype(np.uint8)
        
    else:  # Unsharp Masking
        # Hệ số tỷ lệ
        amount = st.slider("Hệ số tỷ lệ (amount)", 0.1, 5.0, 1.5, 0.1)
        
        # Radius của Gaussian blur
        radius = st.slider("Bán kính làm mờ (radius)", 1, 21, 5, 2)
        
        # Ngưỡng
        threshold = st.slider("Ngưỡng (threshold)", 0, 50, 0, 1)
        
        # Làm mờ ảnh
        blurred = cv2.GaussianBlur(img, (radius, radius), 0)
        
        # Tính hiệu của ảnh gốc và ảnh đã làm mờ
        if img.ndim == 3:
            sharpened = np.zeros_like(img)
            for i in range(3):
                detail = img[:, :, i].astype(float) - blurred[:, :, i].astype(float)
                # Áp dụng ngưỡng
                detail = np.where(np.abs(detail) < threshold, 0, detail)
                # Kết hợp với ảnh gốc
                sharpened[:, :, i] = np.clip(img[:, :, i].astype(float) + amount * detail, 0, 255).astype(np.uint8)
        else:
            detail = img.astype(float) - blurred.astype(float)
            detail = np.where(np.abs(detail) < threshold, 0, detail)
            sharpened = np.clip(img.astype(float) + amount * detail, 0, 255).astype(np.uint8)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(sharpened, f"14. Sharpen_{sharpen_method}_processed.png")
    
    if uploaded_file is not None:
        display_images(img, sharpened)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "14. Sharpen.png")
        display_images(original_path, processed_path)

# 15. Gradient
def gradient():
    """Tính toán và hiển thị gradient của ảnh."""
    st.subheader("Tải lên ảnh của bạn hoặc sử dụng ảnh mặc định")
    uploaded_file = st.file_uploader("Chọn ảnh", type=['png', 'jpg', 'jpeg'], key="gradient")
    
    # Xử lý ảnh
    if uploaded_file is not None:
        img = InsertImage(uploaded_file)
    else:
        img = InsertImage(None, "15. Gradient.png")
    
    if img is None:
        return
    
    # Chuyển sang grayscale nếu là ảnh màu
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img
    
    # Phương pháp tính gradient
    gradient_method = st.selectbox(
        "Chọn phương pháp tính gradient:",
        ("Sobel", "Scharr", "Prewitt", "Roberts", "Canny Edge Detection")
    )
    
    if gradient_method == "Sobel":
        # Kích thước kernel Sobel
        ksize = st.slider("Kích thước kernel", 1, 7, 3, 2)
        if ksize == 1:
            ksize = -1  # CV_SCHARR
        
        # Tính gradient theo hướng x và y
        sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Chuyển đổi sang uint8
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        
        # Kết hợp hai gradient
        gradient_magnitude = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        
        # Hiển thị gradient theo từng hướng
        col1, col2, col3 = st.columns(3)
        col1.image(sobelx, caption="Gradient theo hướng X", use_container_width=True)
        col2.image(sobely, caption="Gradient theo hướng Y", use_container_width=True)
        col3.image(gradient_magnitude, caption="Tổng hợp Gradient", use_container_width=True)
        
    elif gradient_method == "Scharr":
        # Tính gradient theo hướng x và y sử dụng Scharr
        scharrx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
        
        # Chuyển đổi sang uint8
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        
        # Kết hợp hai gradient
        gradient_magnitude = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        
        # Hiển thị gradient theo từng hướng
        col1, col2, col3 = st.columns(3)
        col1.image(scharrx, caption="Gradient theo hướng X", use_container_width=True)
        col2.image(scharry, caption="Gradient theo hướng Y", use_container_width=True)
        col3.image(gradient_magnitude, caption="Tổng hợp Gradient", use_container_width=True)
        
    elif gradient_method == "Prewitt":
        # Kernel Prewitt
        kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        # Tính gradient theo hướng x và y sử dụng Prewitt
        prewittx = cv2.filter2D(img_gray, -1, kernelx)
        prewitty = cv2.filter2D(img_gray, -1, kernely)
        
        # Kết hợp hai gradient
        gradient_magnitude = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
        
        # Hiển thị gradient theo từng hướng
        col1, col2, col3 = st.columns(3)
        col1.image(prewittx, caption="Gradient theo hướng X", use_container_width=True)
        col2.image(prewitty, caption="Gradient theo hướng Y", use_container_width=True)
        col3.image(gradient_magnitude, caption="Tổng hợp Gradient", use_container_width=True)
        
    elif gradient_method == "Roberts":
        # Kernel Roberts
        kernelx = np.array([[1, 0], [0, -1]])
        kernely = np.array([[0, 1], [-1, 0]])
        
        # Tính gradient theo hướng x và y sử dụng Roberts
        robertsx = cv2.filter2D(img_gray, -1, kernelx)
        robertsy = cv2.filter2D(img_gray, -1, kernely)
        
        # Kết hợp hai gradient
        gradient_magnitude = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)
        
        # Hiển thị gradient theo từng hướng
        col1, col2, col3 = st.columns(3)
        col1.image(robertsx, caption="Gradient theo hướng X", use_container_width=True)
        col2.image(robertsy, caption="Gradient theo hướng Y", use_container_width=True)
        col3.image(gradient_magnitude, caption="Tổng hợp Gradient", use_container_width=True)
        
    else:  # Canny Edge Detection
        # Threshold cho Canny
        low_threshold = st.slider("Ngưỡng dưới", 0, 255, 50)
        high_threshold = st.slider("Ngưỡng trên", low_threshold, 255, 150)
        
        # Kích thước kernel Gaussian
        kernel_size = st.slider("Kích thước kernel Gaussian", 3, 15, 5, 2)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Làm mờ ảnh để giảm nhiễu
        blurred = cv2.GaussianBlur(img_gray, (kernel_size, kernel_size), 0)
        
        # Áp dụng thuật toán Canny
        gradient_magnitude = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Lưu và hiển thị ảnh
    processed_path = save_processed_image(gradient_magnitude, f"15. Gradient_{gradient_method}_processed.png")
    
    if uploaded_file is not None:
        display_images(img, gradient_magnitude)
    else:
        original_path = os.path.join(SAVE_PATH_ORIGINAL, "15. Gradient.png")
        display_images(original_path, processed_path)

# Dictionary để ánh xạ chức năng
chapter_03_functions = {
    "Negative Image": negative_image,
    "Logarit ảnh": logarit_image,
    "Lũy thừa ảnh": luy_thua_image,
    "Biến đổi tuyến tính từng phần": bien_doi_tuyen_tinh,
    "Histogram": histogram,
    "Cân bằng Histogram": can_bang_histogram,
    "Cân bằng Histogram của ảnh màu": can_bang_histogram_anh_mau,
    "Local Histogram": local_histogram,
    "Thống kê Histogram": thong_ke_histogram,
    "Lọc Box": loc_box,
    "Lọc Gauss": loc_gauss,
    "Phân Ngưỡng": phan_nguong,
    "Lọc Median": loc_median,
    "Sharpen": sharpen,
    "Gradient": gradient
}