import cv2
import numpy as np

# Thông số video
width = 640    # Độ phân giải ngang
height = 480   # Độ phân giải dọc
fps = 30       # Số khung hình/giây
duration = 5   # Thời lượng video (giây)
fourcc = 0     # Codec = 0 để không nén (rawvideo)

# Tạo đối tượng VideoWriter
output_file = "uncompressed_video.avi"
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

# Tạo khung hình mẫu (màu ngẫu nhiên)
for i in range(int(fps * duration)):
    # Tạo khung hình ngẫu nhiên (RGBವ

    # Khung hình màu RGB ngẫu nhiên
    frame = np.random.randint(0, 255, size=(height, width, 3), dtype=np.uint8)
    
    # Ghi khung hình vào video
    out.write(frame)

# Giải phóng đối tượng VideoWriter
out.release()

print(f"Video không nén đã được tạo: {output_file}")