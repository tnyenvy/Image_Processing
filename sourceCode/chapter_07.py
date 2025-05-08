import cv2
import numpy as np
import streamlit as st

def pencil_sketch_effect():
    st.title("Pencil Sketch Effect")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        inverted_image = cv2.bitwise_not(gray_image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted_image, (21, 21), sigmaX=0, sigmaY=0)
        
        # Invert the blurred image
        inverted_blurred = cv2.bitwise_not(blurred)
        
        # Create the pencil sketch
        sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
        
        # Display the result
        st.image(sketch, channels="GRAY", caption="Pencil Sketch Effect")

def cartoon_effect():
    st.title("Cartoon Effect")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Convert to a cartoon
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        # Display the result
        st.image(cartoon, caption="Cartoon Effect")

chapter_07_functions = {
    "Pencil Sketch Effect": pencil_sketch_effect,
    "Cartoon Effect": cartoon_effect
}