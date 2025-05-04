import cv2
import numpy as np
import os

# Đường dẫn thư mục lưu ảnh
SAVE_PATH = r"D:\XuLyAnhSo\DoAnCuoiKy\images\imageProcessing_C03"

# Danh sách tên ảnh 
list_images_c3 = [
    '1. NegativeImage.png',
    '2. LogaritImage.png',
    '3. LuyThua.png',
    '4. BienDoiTuyenTinhTungPhan.png',
    '5. Histogram.png',
    '6. CanBangHistogram.png',
    '7. CanBangHistogramAnhMau.png',
    '8. LocalHistogram.png',
    '9. ThongKeHistogram.png',
    '10. LocBox.png',
    '11. LocGauss.png',
    '12. PhanNguong.png',
    '13. LocMedian.png',
    '14. Sharpen.png',
    '15. Gradient.png'
]

def create_sample_folder():
    """Tạo thư mục lưu ảnh nếu chưa tồn tại."""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

def create_sample_images(size=(512, 512)):
    """Tạo ảnh mẫu tương ứng với danh sách chức năng."""
    
    # 1. Negative Image
    gradient = np.linspace(0, 255, size[0], dtype=np.uint8)
    image = np.tile(gradient, (size[1], 1))
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[0]), image)
    
    # 2. Logarit ảnh
    x = np.linspace(0, 255, size[0])
    y = np.linspace(0, 255, size[1])
    X, Y = np.meshgrid(x, y)
    image = np.log1p(X * Y) * 30
    image = np.clip(image, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[1]), image)
    
    # 3. Lũy thừa ảnh
    x = np.linspace(0, 1, size[0])
    y = np.linspace(0, 1, size[1])
    X, Y = np.meshgrid(x, y)
    image = np.power(X * Y, 2) * 255
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[2]), image.astype(np.uint8))
    
    # 4. Biến đổi tuyến tính từng phần
    image = np.zeros(size, dtype=np.uint8)
    for i in range(size[0]):
        for j in range(size[1]):
            if i < size[0] / 3:
                image[i, j] = i * 255 / (size[0] / 3)
            elif i < 2 * size[0] / 3:
                image[i, j] = 127
            else:
                image[i, j] = 127 + (i - 2 * size[0] / 3) * 128 / (size[0] / 3)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[3]), image)
    
    # 5. Histogram
    image = np.zeros(size, dtype=np.uint8)
    image[:size[0] // 3, :] = 50
    image[size[0] // 3:2 * size[0] // 3, :] = 150
    image[2 * size[0] // 3:, :] = 250
    noise = np.random.normal(0, 20, size).astype(np.uint8)
    image = cv2.add(image, noise)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[4]), image)
    
    # 6. Cân bằng Histogram
    img_hist = np.random.normal(128, 50, size).astype(np.uint8)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[5]), img_hist)
    
    # 7. Cân bằng Histogram của ảnh màu
    img_color = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img_color[:, :, 0] = np.random.normal(128, 50, size).astype(np.uint8)
    img_color[:, :, 1] = np.random.normal(100, 40, size).astype(np.uint8)
    img_color[:, :, 2] = np.random.normal(150, 60, size).astype(np.uint8)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[6]), img_color)
    
    # 8. Local Histogram
    img_local = np.zeros(size, dtype=np.uint8)
    img_local[size[0] // 4:3 * size[0] // 4, size[1] // 4:3 * size[1] // 4] = 200
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[7]), img_local)
    
    # 9. Thống kê Histogram
    img_stats = np.random.normal(128, 30, size).astype(np.uint8)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[8]), img_stats)
    
    # 10. Lọc Box
    img_box = np.ones(size, dtype=np.uint8) * 128
    img_box[size[0] // 4:3 * size[0] // 4, size[1] // 4:3 * size[1] // 4] = 255
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[9]), img_box)
    
    # 11. Lọc Gauss
    img_gauss = np.zeros(size, dtype=np.uint8)
    for i in range(5):
        x = np.random.randint(0, size[0])
        y = np.random.randint(0, size[1])
        cv2.circle(img_gauss, (x, y), 50, 255, -1)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[10]), img_gauss)
    
    # 12. Phân Ngưỡng
    img_threshold = np.zeros(size, dtype=np.uint8)
    img_threshold[:size[0] // 2, :] = 100
    img_threshold[size[0] // 2:, :] = 200
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[11]), img_threshold)
    
    # 13. Lọc Median
    img_median = np.zeros(size, dtype=np.uint8)
    noise = np.random.randint(0, 255, size=size, dtype=np.uint8)
    img_median = cv2.add(img_median, noise)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[12]), img_median)
    
    # 14. Sharpen
    img_sharpen = np.ones(size, dtype=np.uint8) * 128
    cv2.circle(img_sharpen, (size[0] // 2, size[1] // 2), 100, 255, -1)
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[13]), img_sharpen)
    
    # 15. Gradient
    img_gradient = np.zeros(size, dtype=np.uint8)
    for i in range(size[0]):
        img_gradient[i, :] = i * 255 // size[0]
    cv2.imwrite(os.path.join(SAVE_PATH, list_images_c3[14]), img_gradient)

def generate_all_samples():
    """Tạo tất cả ảnh mẫu và lưu vào thư mục."""
    create_sample_folder()
    print(f"Đang tạo các ảnh mẫu trong thư mục: {SAVE_PATH}")
    
    create_sample_images()
    
    print("Hoàn thành! Đã tạo tất cả các ảnh mẫu.")
    print("\nDanh sách ảnh đã tạo:")
    for filename in sorted(os.listdir(SAVE_PATH)):
        print(f"- {filename}")

if __name__ == "__main__":
    generate_all_samples()