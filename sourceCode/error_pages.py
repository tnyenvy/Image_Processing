import streamlit as st
from datetime import datetime

def show_404_page():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.markdown("""
    <style>
    .error-container {
        text-align: center;
        padding: 40px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .error-code {
        font-size: 120px;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
    }
    .error-message {
        font-size: 24px;
        color: #34495e;
        margin: 20px 0;
    }
    .error-description {
        font-size: 16px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    .error-time {
        font-size: 14px;
        color: #95a5a6;
        margin-top: 20px;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    .home-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: #3498db;
        color: white;
        text-decoration: none;
        border-radius: 5px;
        transition: background-color 0.3s;
        margin: 10px;
    }
    .home-button:hover {
        background-color: #2980b9;
    }
    .error-icon {
        font-size: 50px;
        margin: 20px 0;
        color: #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="error-container">
        <div class="error-code">404</div>
        <div class="error-icon">⚠️</div>
        <div class="error-message">Oops! Trang không tồn tại</div>
        <div class="error-description">
            Có vẻ như trang bạn đang tìm kiếm không tồn tại hoặc đã bị di chuyển.
            <br>Vui lòng kiểm tra lại đường dẫn hoặc quay về trang chủ.
        </div>
        <a href="/" class="home-button">Quay về trang chủ</a>
        <a href="javascript:window.location.reload()" class="home-button">Tải lại trang</a>
        <div class="error-time">Thời gian: {current_time}</div>
    </div>
    """, unsafe_allow_html=True)

def handle_error(error_message):
    show_404_page()
    with st.expander("Chi tiết lỗi", expanded=False):
        st.error(error_message)
