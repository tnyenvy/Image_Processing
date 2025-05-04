import streamlit as st
from sidebar import cs_sidebar
from chapter_03 import XuLyAnh as chapter_03_processing
from chapter_04 import XuLyAnh as chapter_04_processing
from chapter_09 import XuLyAnh as chapter_09_processing

def main():
    st.set_page_config(
        page_title="Ứng dụng Xử lý Ảnh Số",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Tạo sidebar
    selected_chapter = cs_sidebar()
    
    # Xử lý chọn chapter
    if selected_chapter == "Chapter_03":
        chapter_03_processing()
    elif selected_chapter == "Chapter_04":
        chapter_04_processing()
    elif selected_chapter == "Chapter_09":
        chapter_09_processing()

if __name__ == "__main__":
    main()