import sys
from PIL import Image
import cv2
import numpy as np
sys.path.append('./library')
from library.bosung_streamlit.sidebar import *
from library.bosung_streamlit.Chapter03 import *

list_images_c3 = ['1. Negative Image.png','2. Logarit Image.png','3. LuyThua.png','4. BienDoiTuyenTinhTungPhan.png','5. Histogram.png']

def XuLyAnh():
    global image_file
    global c1, c2
    st.title("Xử lý ảnh - " + Get_XuLyAnh_C3())
    image_file = st.file_uploader("Upload Images", type=["bmp", "png", "jpg", "jpeg"])
    c1, c2 = st.columns(2)
    imgin = None

    def InsertImage(image_file, path=''):
        global image
        if image_file is not None:
            image = Image.open(image_file)  # You must transform your picture into grey
            frame = np.array(image)
            frame = frame[:, :, [2, 1, 0]]  # BGR -> RGB
            image.close()
        else:
            image = Image.open("./images/ImageProcessing/" + path)  # You must transform your picture into grey
            c1.image(image, caption=None)
            image.close()

    def Chosen_Processing(option, path=''):
        global imgin
        if image_file is not None:
            path = image_file.name
        else:
            InsertImage(image_file, list_images_c3[Get_Index_Image()])
        imgin = cv2.imread("./images/ImageProcessing/" + path, cv2.IMREAD_GRAYSCALE)
        if(image_file is not None):
            path = image_file.name
        else:
            InsertImage(image_file, list_images_c3[Get_Index_Image()])
            imgin = cv2.imread("./images/ImageProcessing/" + path, cv2.IMREAD_GRAYSCALE)

    if(option == '1. Negative Image.'):
        Image_Negative()
    elif(option == '2. Logarit ảnh.'):
        Image_Logarit()
    elif(option == '3. Lũy thừa ảnh.'):
        Image_Power()
    elif(option == '4. Biến đổi tuyến tính từng phần'):
        Image_PiecewiseLinear()
    elif(option == '5. Histogram'):
        Image_Histogram()
    elif(option == '6. Cân bằng Histogram'):
        Image_HistEqual()
    elif(option == '7. Cân bằng Histogram của ảnh màu.'):
        image_temp = cv2.imread("./images/ImageProcessing/" + path, cv2.IMREAD_COLOR)
        Image_HistEqualColor(image_temp)
    elif(option == '8. Local Histogram.'):
        Image_LocalHist()
    elif(option == '9. Thống kê Histogram'):
        Image_HistStat()
    elif(option == '10. Lọc Box'):
        Image_MyBoxFilter()
    elif(option == '11. Lọc Gauss'):
        Image_LowPassGauss()
    elif(option == '12. Phân Ngưỡng'):
        Image_Threshold()
    elif(option == '13. Lọc Median'):
        Image_MedianFilter()
    elif(option == '14. Sharper'):
        Image_MedianFilter()
    elif(option == '15. Gradient'):
        Image_MedianFilter()
def Image_All():
    global image_file
    if Get_XyLyAnh_C3() != '':
        if image_file is not None:
            # Mở để xử lý
            InsertImage(image_file)
            Chosen_Processing(Get_XyLyAnh_C3())
        else:
            Chosen_Processing(Get_XyLyAnh_C3(), list_images_c3[Get_Index_Image()])
def Image_Negative():
    output_image = Negative(imgin)
    c2.image(output_image, caption=None)

def Image_Logarit():
    output_image = Logarit(imgin)
    c2.image(output_image, caption=None)

def Image_Power():
    output_image = Power(imgin)
    c2.image(output_image, caption=None)

def Image_PiecewiseLinear():
    output_image = PiecewiseLinear(imgin)
    c2.image(output_image, caption=None)
def Image_MyBoxFilter():
    output_image = MyBoxFilter(imgin)
    c2.image(output_image, caption=None)

def Image_MedianFilter():
    output_image = MedianFilter(imgin)
    c2.image(output_image, caption=None)

def Image_Threshold():
    output_image = Threshold(imgin)
    c2.image(output_image, caption=None)

def Image_Sharpen():
    output_image = Sharpen(imgin)
    c2.image(output_image, caption=None)

def Image_Gradient():
    output_image = Gradient(imgin)
    c2.image(output_image, caption=None)

def Image_onLowpassGauss():
    # Áp dụng Gaussian Blur
    output_image = cv2.GaussianBlur(imgin, (43, 43), 7.0)
    c2.image(output_image, caption=None)

if __name__ == "__main__":
    configure()
    cs_sidebar()
    XuLyAnh()
    Image_All()