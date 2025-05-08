import streamlit as st
import numpy as np
import cv2
from datetime import datetime
import os
from sidebar import cs_sidebar
from error_pages import handle_error

# Import các hàm xử lý từ chapter_03, chapter_04, chapter_05, chapter_09
from chapter_03 import chapter_03_functions
from chapter_04 import chapter_04_functions
from chapter_05 import chapter_05_functions
from chapter_09 import chapter_09_functions
from detection import Detection_functions
from chapter_07 import chapter_07_functions
from chapter_11 import chapter_11_functions
from video_processing import video_blur_background_interface_stream

def show_home():
    """Hiển thị trang Home với thông tin cá nhân và danh sách nội dung."""

    # Tiêu đề chính (cỡ lớn, căn giữa)
    st.markdown("""
        <h1 style='text-align: center; font-size: 48px;'>Ứng dụng Xử lý Ảnh Số</h1>
        <h2 style='text-align: center;'>22110259 - Huỳnh Minh Tuấn</h2>
        <h2 style='text-align: center;'>22110274 - Trần Ngọc Yến Vy</h2>
        <hr style='margin-top: 30px; margin-bottom: 30px;'>
    """, unsafe_allow_html=True)


    # Tiêu đề căn giữa
    st.markdown("<h2 style='text-align: center;'>📌 Danh sách nội dung đã triển khai</h2>", unsafe_allow_html=True)

    # Tạo 2 cột để hiển thị đẹp
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<p style='font-size:23px;'>🌞 <b>CHƯƠNG 3</b>: Biến đổi độ sáng, âm bản, Logarit, lũy thừa,...</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>🎞️ <b>CHƯƠNG 4</b>: Lọc trung bình, lọc Gaussian, lọc trung vị,...</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>📼 <b>CHƯƠNG 5</b>: Biến đổi Fourier, lọc trong miền tần số,...</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>🎨 <b>CHƯƠNG 7</b>: Biến ảnh thành tranh vẽ tay</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<p style='font-size:23px;'>🔍 <b>CHƯƠNG 9</b>: Xử lý hình thái học</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>🔐 <b>CHƯƠNG 11</b>: Ẩn và giải mã ảnh (Steganography)</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>👁️ <b>NHẬN DIỆN</b>: Nhận diện khuôn mặt, vật thể,...</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:23px;'>🎥 <b>XỬ LÝ VIDEO</b>: Làm mờ nền cho video</p>", unsafe_allow_html=True)


def main():
    try:
        st.set_page_config(
            page_title="Xử lý Ảnh Số",
            page_icon="🖼️",
            layout="wide"
        )

        # Tạo Sidebar và lấy Chapter cùng chức năng được chọn
        selected_chapter, selected_function = cs_sidebar()

        # Nếu không có chapter nào được chọn, hiển thị trang Home
        if not selected_chapter or not selected_function:
            show_home()
        # Xử lý logic theo Chapter và chức năng
        elif selected_chapter == "Chapter_03":
            if selected_function in chapter_03_functions:
                chapter_03_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Chapter_04":
            if selected_function in chapter_04_functions:
                chapter_04_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Chapter_05":
            if selected_function in chapter_05_functions:
                chapter_05_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Chapter_09":
            if selected_function in chapter_09_functions:
                chapter_09_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Nhận diện":
            if selected_function in Detection_functions:
                Detection_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Biến ảnh thành tranh vẽ tay":
            if selected_function in chapter_07_functions:
                chapter_07_functions[selected_function]()
            else:
                show_home()
        elif selected_chapter == "Ẩn và giải mã ảnh (Steganography)":
            if selected_function in chapter_11_functions:
                chapter_11_functions[selected_function]()
            else:
                show_home()

        elif selected_chapter == "Làm mờ nền trong video":
            video_blur_background_interface_stream()

    except Exception as e:
        handle_error(str(e))  

if __name__ == "__main__":
    main()