import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from tempfile import NamedTemporaryFile
import time

#1 Xử lý video được import vào
def blur_background_in_video_stream(input_path, blur_intensity=21, frame_width=640):
    """
    Process video frames to blur the background and save the processed video.
    Displays a progress bar during processing and provides a download button after completion.
    """
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở File.")

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    # Calculate the height to maintain aspect ratio
    aspect_ratio = original_height / original_width
    frame_height = int(frame_width * aspect_ratio)

    # Notify the user about the processing
    st.text(f"Đang xử lý video với độ phân giải {original_width}x{original_height} ở {fps} FPS...")

    # Temporary file for saving the processed video
    output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = output_file.name

    # VideoWriter to save the processed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Progress bar
    progress_bar = st.progress(0)

    # Process frames
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame to the specified width
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        result = segmentation.process(rgb_frame)

        # Generate mask
        mask = result.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask

        # Blur the entire frame
        blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

        # Combine blurred frame and original frame using the mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert mask to 3 channels
        output_frame = frame * mask_3d + blurred_frame * (1 - mask_3d)

        # Write the processed frame to the output video file
        out.write(output_frame.astype(np.uint8))

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    # Release resources
    cap.release()
    out.release()
    segmentation.close()

    # Notify the user that processing is complete
    st.success("Video đã được xử lý hoàn tất!")

    # Provide a download button for the processed video
    with open(output_path, "rb") as video_file:
        st.download_button(
            label="Tải video đã xử lý",
            data=video_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

#2 Xử lý trực tiếp từ camera
def process_camera_stream(blur_intensity=21, frame_width=640):
    """
    Process video from the webcam to blur the background in real time.
    """
    # Initialize MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    # Open webcam stream
    cap = cv2.VideoCapture(0)  # 0 indicates the default webcam
    if not cap.isOpened():
        st.error("Không thể mở camera. Vui lòng kiểm tra kết nối.")
        return

    # Placeholder for displaying video
    video_placeholder = st.empty()

    # Process frames in real-time
    st.text("Đang mở camera... Nhấn 'Stop' để dừng.")
    stop_button = st.checkbox("Stop")  # Add a stop control

    while True:
        if stop_button:
            break

        ret, frame = cap.read()
        if not ret:
            st.warning("Không nhận được khung hình từ camera.")
            break

        # Resize frame to specified width
        original_height, original_width, _ = frame.shape
        aspect_ratio = original_height / original_width
        frame_height = int(frame_width * aspect_ratio)
        frame = cv2.resize(frame, (frame_width, frame_height))

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        result = segmentation.process(rgb_frame)

        # Generate mask
        mask = result.segmentation_mask
        mask = (mask > 0.5).astype(np.uint8)  # Threshold the mask

        # Blur the entire frame
        blurred_frame = cv2.GaussianBlur(frame, (blur_intensity, blur_intensity), 0)

        # Combine blurred frame and original frame using the mask
        mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Convert mask to 3 channels
        output_frame = frame * mask_3d + blurred_frame * (1 - mask_3d)

        # Convert output frame to RGB for display in Streamlit
        output_frame_rgb = cv2.cvtColor(output_frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # Display the frame
        video_placeholder.image(output_frame_rgb, channels="RGB", width=frame_width)

    # Release resources
    cap.release()
    segmentation.close()
    st.success("Đã dừng camera.")

def video_blur_background_interface_stream():
    """
    Streamlit interface for uploading a video or processing directly from the camera.
    """
    st.title("Làm mờ nền trong video (Blur Background in Video)")
    
    # Select processing method
    processing_method = st.radio(
        "Chọn phương thức xử lý:",
        ("Import video file", "Process directly from camera")
    )
    
    blur_intensity = st.slider("Độ mờ của nền (Blur Intensity)", min_value=5, max_value=101, step=2, value=21)
    frame_width = st.number_input("Chiều rộng khung hình (px):", min_value=320, max_value=1920, step=10, value=640)

    if processing_method == "Import video file":
        uploaded_file = st.file_uploader("Tải video lên (chỉ hỗ trợ MP4)", type=["mp4"])

        if uploaded_file is not None:
            # Save uploaded video to a temporary file
            temp_input = NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_input.write(uploaded_file.read())
            temp_input.close()

            if st.button("Bắt đầu xử lý video"):
                # Process and stream the video
                blur_background_in_video_stream(temp_input.name, blur_intensity=blur_intensity, frame_width=frame_width)

    elif processing_method == "Process directly from camera":
        if st.button("Mở camera"):
            # Process the webcam stream
            process_camera_stream(blur_intensity=blur_intensity, frame_width=frame_width)