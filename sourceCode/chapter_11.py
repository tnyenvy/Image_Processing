import cv2
import numpy as np
import streamlit as st

def hide_data(cover_image, payload, password=None):
    """Hide data into the cover image using LSB steganography.
    Embeds payload size at the beginning to allow for automatic extraction later."""
    # Convert cover image to work with
    if len(cover_image.shape) == 2:  # If grayscale
        cover_image = cv2.cvtColor(cover_image, cv2.COLOR_GRAY2BGR)
    
    flat_image = cover_image.flatten()
    
    # Prepare payload
    if isinstance(payload, str):  # If payload is text
        payload_bytes = payload.encode('utf-8')
        is_text = np.array([1], dtype=np.uint8)  # Flag to indicate text payload
        payload_data = np.frombuffer(payload_bytes, dtype=np.uint8)
    else:  # If payload is image
        if len(payload.shape) == 2:  # If grayscale
            payload = cv2.cvtColor(payload, cv2.COLOR_GRAY2BGR)
        # Store payload shape for reconstruction
        height, width, channels = payload.shape
        shape_data = np.array([height, width, channels], dtype=np.uint32)
        shape_bytes = shape_data.tobytes()
        
        is_text = np.array([0], dtype=np.uint8)  # Flag to indicate image payload
        payload_data = np.concatenate([np.frombuffer(shape_bytes, dtype=np.uint8), payload.flatten()])
    
    # Get payload size and create header
    payload_size = len(payload_data)
    size_bytes = np.array([payload_size], dtype=np.uint32).tobytes()
    size_data = np.frombuffer(size_bytes, dtype=np.uint8)
    
    # Combine header (is_text flag + size) and payload
    combined_data = np.concatenate([is_text, size_data, payload_data])
    
    # Ensure payload fits into the cover image
    if len(combined_data) * 8 > len(flat_image):
        raise ValueError(f"Payload is too large to fit into the cover image. Need {len(combined_data) * 8} bits but have {len(flat_image)} bits.")

    # Convert payload to binary and embed into LSB of the cover image
    combined_bits = np.unpackbits(combined_data)
    for i in range(len(combined_bits)):
        flat_image[i] = (flat_image[i] & ~1) | combined_bits[i]

    # Reshape the image back to its original shape
    encoded_image = flat_image.reshape(cover_image.shape)
    return encoded_image

def extract_data(encoded_image, password=None):
    """Extract data from the encoded image using LSB steganography.
    Automatically extracts the payload based on embedded header information."""
    flat_image = encoded_image.flatten()
    
    # Extract the is_text flag (1 byte)
    is_text_bits = flat_image[:8] & 1
    is_text = np.packbits(is_text_bits)[0]
    
    # Extract the size (4 bytes = 32 bits)
    size_bits = flat_image[8:40] & 1
    size_bytes = np.packbits(size_bits)
    payload_size = np.frombuffer(size_bytes, dtype=np.uint32)[0]
    
    # Extract the payload data
    payload_bits = flat_image[40:40 + payload_size * 8] & 1
    payload_bytes = np.packbits(payload_bits)
    
    if is_text:
        # Interpret payload as text
        return payload_bytes.tobytes().decode('utf-8', errors='replace'), True
    else:
        # Extract shape information (3 uint32 = 12 bytes)
        shape_bytes = payload_bytes[:12]
        shape_data = np.frombuffer(shape_bytes, dtype=np.uint32)
        height, width, channels = shape_data
        
        # Extract and reshape the image data
        image_bytes = payload_bytes[12:]
        try:
            image_data = image_bytes.reshape((height, width, channels))
            return image_data, False
        except ValueError:
            # If reshape fails (e.g., due to incomplete data)
            st.error(f"Không thể khôi phục ảnh: Dữ liệu không phù hợp với kích thước {height}x{width}x{channels}")
            return None, False

def steganography_encode():
    """Streamlit interface for encoding data into an image."""
    st.title("Giấu dữ liệu vào ảnh")
    cover_file = st.file_uploader("Tải ảnh bìa (cover image):", type=["jpg", "png", "jpeg"])
    
    payload_type = st.radio("Loại dữ liệu cần giấu:", ["Ảnh", "Văn bản"])
    
    if payload_type == "Ảnh":
        payload_file = st.file_uploader("Tải ảnh cần giấu:", type=["jpg", "png", "jpeg"])
        payload_text = None
    else:
        payload_text = st.text_area("Nhập văn bản cần giấu:")
        payload_file = None
        
    password = st.text_input("Mật khẩu (tùy chọn):", type="password")

    if st.button("Thực hiện giấu dữ liệu"):
        if cover_file:
            # Read cover image
            cover_bytes = np.asarray(bytearray(cover_file.read()), dtype=np.uint8)
            cover_image = cv2.imdecode(cover_bytes, cv2.IMREAD_UNCHANGED)
            
            if payload_type == "Ảnh" and payload_file:
                # Read image payload
                payload_bytes = np.asarray(bytearray(payload_file.read()), dtype=np.uint8)
                payload = cv2.imdecode(payload_bytes, cv2.IMREAD_UNCHANGED)
                
                # Calculate size in KB
                payload_size_kb = payload.nbytes / 1024
                cover_size_kb = cover_image.nbytes / 1024
                max_payload_kb = cover_image.size / 8 / 1024  # Each pixel can store 1 bit, 8 bits = 1 byte
                
                st.info(f"Kích thước ảnh cần giấu: {payload_size_kb:.2f} KB")
                st.info(f"Kích thước tối đa có thể giấu: {max_payload_kb:.2f} KB")
                
                try:
                    # Hide data in the cover image
                    encoded_image = hide_data(cover_image, payload, password)
                    
                    # Save the encoded image
                    _, encoded_buffer = cv2.imencode('.png', encoded_image)
                    encoded_image_bytes = encoded_buffer.tobytes()
                    
                    st.success("Dữ liệu đã được giấu thành công!")
                    st.image(encoded_image, channels="BGR", caption="Ảnh chứa dữ liệu")
                    st.download_button(
                        label="Tải ảnh đã giấu dữ liệu",
                        data=encoded_image_bytes,
                        file_name="encoded_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
            
            elif payload_type == "Văn bản" and payload_text:
                try:
                    # Calculate size in KB
                    payload_size_kb = len(payload_text.encode('utf-8')) / 1024
                    max_payload_kb = cover_image.size / 8 / 1024
                    
                    st.info(f"Kích thước văn bản cần giấu: {payload_size_kb:.2f} KB")
                    st.info(f"Kích thước tối đa có thể giấu: {max_payload_kb:.2f} KB")
                    
                    # Hide text in the cover image
                    encoded_image = hide_data(cover_image, payload_text, password)
                    
                    # Save the encoded image
                    _, encoded_buffer = cv2.imencode('.png', encoded_image)
                    encoded_image_bytes = encoded_buffer.tobytes()
                    
                    st.success("Dữ liệu đã được giấu thành công!")
                    st.image(encoded_image, channels="BGR", caption="Ảnh chứa dữ liệu")
                    st.download_button(
                        label="Tải ảnh đã giấu dữ liệu",
                        data=encoded_image_bytes,
                        file_name="encoded_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")
            else:
                st.warning("Vui lòng cung cấp dữ liệu cần giấu.")
        else:
            st.warning("Vui lòng tải ảnh bìa.")

def steganography_decode():
    """Streamlit interface for decoding data from an image."""
    st.title("Giải mã dữ liệu từ ảnh")
    encoded_file = st.file_uploader("Tải ảnh chứa dữ liệu (encoded image):", type=["jpg", "png", "jpeg"])
    password = st.text_input("Mật khẩu (nếu có):", type="password")

    if st.button("Giải mã dữ liệu"):
        if encoded_file:
            # Read encoded image
            encoded_bytes = np.asarray(bytearray(encoded_file.read()), dtype=np.uint8)
            encoded_image = cv2.imdecode(encoded_bytes, cv2.IMREAD_UNCHANGED)

            try:
                # Extract data from the encoded image
                payload, is_text = extract_data(encoded_image, password)

                # Display the extracted payload
                if payload is not None:
                    st.success("Dữ liệu đã được giải mã thành công!")
                    
                    if is_text:
                        st.text_area("Văn bản đã giải mã:", payload, height=300)
                        
                        # Allow download of text as file
                        text_bytes = payload.encode('utf-8')
                        st.download_button(
                            label="Tải văn bản đã giải mã",
                            data=text_bytes,
                            file_name="extracted_text.txt",
                            mime="text/plain"
                        )
                    else:
                        st.image(payload, caption="Ảnh đã giải mã")
                        
                        # Save and allow download of extracted image
                        _, img_encoded = cv2.imencode('.png', payload)
                        img_bytes = img_encoded.tobytes()
                        
                        st.download_button(
                            label="Tải ảnh đã giải mã",
                            data=img_bytes,
                            file_name="extracted_image.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"Lỗi khi giải mã: {str(e)}")
        else:
            st.warning("Vui lòng tải ảnh chứa dữ liệu.")

chapter_11_functions = {
    "Giấu dữ liệu vào ảnh": steganography_encode,
    "Giải mã dữ liệu từ ảnh": steganography_decode
}

