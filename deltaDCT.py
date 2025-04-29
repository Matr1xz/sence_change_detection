import cv2
import numpy as np
from scipy.fft import dct

# Hàm tính DCT cho 1 frame
def compute_dct(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
    return dct(dct(frame_gray, axis=0, norm='ortho'), axis=1, norm='ortho')

# Hàm phát hiện chuyển cảnh bằng độ lệch DCT
def detect_scene_change(prev_dct, curr_dct, threshold=10000):
    diff = np.sum((curr_dct - prev_dct) ** 2)
    return diff > threshold

# ==== MAIN FUNCTION ====

def detect_scene_changes_in_video(video_path, threshold=10000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Không thể đọc frame đầu tiên.")
        return

    prev_dct = compute_dct(prev_frame)
    frame_index = 1

    print("Các frame chuyển cảnh:")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_dct = compute_dct(frame)

        if detect_scene_change(prev_dct, curr_dct, threshold=threshold):
            print(f"→ Chuyển cảnh tại frame: {frame_index}")

        prev_dct = curr_dct
        frame_index += 1

    cap.release()

# Ví dụ sử dụng:
detect_scene_changes_in_video("uncompressed_video.avi", threshold=10000)
