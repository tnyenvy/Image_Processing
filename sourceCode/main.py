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

def show_home():
    """Hiển thị trang Home với thông tin cá nhân."""
    st.title("Ứng dụng Xử lý Ảnh Số")
    st.subheader("22110259 - Huỳnh Minh Tuấn")
    st.subheader("22110274 - Trần Ngọc Yến Vy")
    st.write("""
        Chào mừng bạn đến với ứng dụng xử lý ảnh số. 
        Vui lòng chọn một chức năng từ Sidebar để bắt đầu xử lý!
    """)

def main():
    try:
        st.set_page_config(
            page_title="Ứng dụng Xử lý Ảnh Số",
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
    except Exception as e:
        handle_error(str(e))  

if __name__ == "__main__":
    main()