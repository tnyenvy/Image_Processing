import streamlit as st
import os
import base64
from datetime import datetime

def load_image(image_path):
    """Tải ảnh từ đường dẫn và chuyển đổi sang base64."""
    try:
        if not os.path.exists(image_path):
            st.error(f"Không tìm thấy tệp ảnh: {image_path}")
            return None
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode()
            return encoded
    except Exception as e:
        st.error(f"Lỗi khi đọc ảnh: {str(e)}")
        return None

def cs_sidebar():
    """Tạo Sidebar với các Chapter và chức năng xử lý ảnh tương ứng."""
    st.sidebar.title("Xử lý ảnh số")

    # Hiển thị logo trong sidebar
    logo_path = "D:/XuLyAnhSo/DoAnCuoiKy/images/logo.png"
    encoded_logo = load_image(logo_path)
    
    if encoded_logo:
        st.sidebar.markdown(
            f"""
            <div style="text-align: center; cursor: pointer;" 
                 onclick="window.location.href='/'">
                <img src="data:image/png;base64,{encoded_logo}" 
                     style="max-width: 100%; height: auto;">
            </div>
            """,
            unsafe_allow_html=True
        )


    # Tùy chọn Chapter với tùy chọn trống mặc định
    chapters = ["", "Chapter_03", "Chapter_04", "Chapter_05", "Chapter_09", "Nhận diện", "Biến ảnh thành tranh vẽ tay"]
    selected_chapter = st.sidebar.selectbox("Chọn Chapter:", chapters)

    # Xử lý các chức năng tương ứng với Chapter
    selected_function = None

    if selected_chapter == "Chapter_03":
        functions = [
            "",
            "Negative Image",
            "Logarit ảnh",
            "Lũy thừa ảnh",
            "Biến đổi tuyến tính từng phần",
            "Histogram",
            "Cân bằng Histogram",
            "Cân bằng Histogram của ảnh màu",
            "Local Histogram",
            "Thống kê Histogram",
            "Lọc Box",
            "Lọc Gauss",
            "Phân Ngưỡng",
            "Lọc Median",
            "Sharpen",
            "Gradient"
        ]
        selected_function = st.sidebar.selectbox("Biến đổi độ sáng và lọc:", functions)

    elif selected_chapter == "Chapter_04":
        functions = [
            "",
            "Spectrum",
            "Lọc trong miền tần số",
            "Bộ lọc notch-reject",
            "Xóa nhiễu Moire"
        ]
        selected_function = st.sidebar.selectbox("Xử lý ảnh trong miền tần số:", functions)

    elif selected_chapter == "Chapter_05":
        functions = [
            "",
            "Tạo nhiễu ảnh",
            "Lọc ảnh ít nhiễu",
            "Lọc ảnh nhiều nhiễu"
        ]
        selected_function = st.sidebar.selectbox("Xử lý nhiễu ảnh:", functions)

    elif selected_chapter == "Chapter_09":
        functions = [
            "",
            "Đếm số hạt gạo",
            "Xử lý thành phần liên thông"
        ]
        selected_function = st.sidebar.selectbox("Xử lý hình thái:", functions)

    elif selected_chapter == "Nhận diện":
        functions = [
            "",
            "Nhận diện trái cây",
            "Nhận diện nhân vật"
        ]
        selected_function = st.sidebar.selectbox("Nhận diện trái cây:", functions)

    elif selected_chapter == "Biến ảnh thành tranh vẽ tay":
        functions = [
            "",
            "Vẽ bằng nét chì",
            "Hiệu ứng hoạt hình"
        ]
        selected_function = st.sidebar.selectbox("Hiệu ứng vẽ tay và hoạt hình:", functions)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center'>
            <p>Made with ❤️</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    return selected_chapter, selected_function