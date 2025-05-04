import streamlit as st
import base64
from PIL import Image
import io

def load_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

def cs_sidebar():
    st.sidebar.title("Xử lý Ảnh Số")
    
    # Logo hoặc hình ảnh đại diện
    img_to_bytes = load_image("logo.png")  # Thay đổi đường dẫn logo của bạn
    st.sidebar.markdown(
        f'''<img src='data:image/png;base64,{img_to_bytes}' class='img-fluid' width=200 height=200>''',
        unsafe_allow_html=True
    )
    
    # Menu chọn chapter
    chapter = st.sidebar.radio(
        "Chọn chương",
        ["Chapter_03: Biến đổi độ sáng và lọc",
         "Chapter_04: Xử lý trong miền tần số",
         "Chapter_09: Xử lý hình thái"]
    )
    
    if "Chapter_03" in chapter:
        return "Chapter_03"
    elif "Chapter_04" in chapter:
        return "Chapter_04"
    else:
        return "Chapter_09"