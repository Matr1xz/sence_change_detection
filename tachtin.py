import cv2
import numpy as np
from scipy.fft import dct
from bit2char import bit2char

def check_closer(value, Q=120):
    """
    Tách bit từ giá trị DCT dựa trên residue.
    Args:
        value: Giá trị C(2, v).
        Q: Ngưỡng quantization (mặc định 120).
    Returns:
        0 nếu residue ∈ [0, 0.25Q] ∪ [0.75Q, Q), 1 nếu residue ∈ (0.25Q, 0.75Q).
    """
    tmp = abs(value) / Q - abs(value) // Q
    if tmp < 0.25 or tmp > 0.75:
        return 0
    else:
        return 1

def extract_message_from_frame(video_path, frame_idx, channel_name, block_positions, Q=120):
    """
    Tách 8 bit từ khung tại frame_idx, kênh channel_name, dùng majority voting từ 8 nhóm x 5 khối.
    Args:
        video_path: Đường dẫn video.
        frame_idx: Chỉ số khung.
        channel_name: Kênh ('G' cho Green).
        block_positions: Danh sách 8 nhóm, mỗi nhóm 5 khối [[(row, col), ...], ...].
        Q: Ngưỡng quantization.
    Returns:
        message_bits: Chuỗi 8 bit.
        character: 1 ký tự nhúng.
    """
    # Kiểm tra block_positions
    if len(block_positions) != 8 or any(len(group) != 5 for group in block_positions):
        print("Lỗi: block_positions phải có 8 nhóm, mỗi nhóm 5 khối")
        return "", ""
    
    cap = cv2.VideoCapture(video_path)
    block_size = 8
    
    # Di chuyển đến khung frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Không đọc được khung {frame_idx}")
        cap.release()
        return "", ""
    
    # Chuyển sang float và tách kênh
    frame_float = frame.astype(float)
    channels = cv2.split(frame_float)
    channel_idx = {'B': 0, 'G': 1, 'R': 2}[channel_name]
    channel = channels[channel_idx]
    
    message_bits = ""
    
    # Tách bit từ mỗi nhóm 5 khối
    for group_idx, group in enumerate(block_positions):
        bits = []
        for block_idx, (row, col) in enumerate(group):
            i, j = row * block_size, col * block_size
            block = channel[i:i+block_size, j:j+block_size]
            if block.shape != (block_size, block_size):
                print(f"Khối ({row}, {col}) trong nhóm {group_idx} không hợp lệ")
                bits.append(None)
                continue
            
            # Tính DCT 2D
            dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            
            # Tìm C(2, v) đầu tiên có |C| >= 100
            row_2 = dct_block[2, :]
            candidates = np.where(np.abs(row_2) >= 100)[0]
            if len(candidates) == 0:
                print(f"Khối ({row}, {col}) trong nhóm {group_idx} không có C(2, v) >= 100")
                bits.append(None)
                continue
            
            v = candidates[0]
            value = dct_block[2, v]
            bit = check_closer(value, Q)
            bits.append(bit)
        
        # Majority voting
        valid_bits = [b for b in bits if b is not None]
        if len(valid_bits) >= 3:  # Cần ít nhất 3 residue hợp lệ
            bit = 1 if sum(valid_bits) >= len(valid_bits) / 2 else 0
            message_bits += str(bit)
        else:
            print(f"Nhóm {group_idx} không đủ khối hợp lệ để tách bit")
            message_bits += "0"  # Mặc định 0 nếu không đủ dữ liệu
    
    cap.release()
    
    # Chuyển bit thành ký tự
    if len(message_bits) != 8:
        print(f"Lỗi: Chỉ tách được {len(message_bits)} bit, cần 8 bit")
        character = ""
    else:
        character = bit2char(message_bits)
    
    return message_bits, character

# Nhập thông tin từ người dùng
video_path = input("nhap duong dan video")
for _ in range(11):
    frame_idx = int(input("nhap chi so khung(ví dụ: 2): "))
    channel_name = 'G'

    # Nhập block_positions (8 nhóm x 5 khối)
    block_positions_input = input("nhap block_positions (danh sach 8 nhom x 5 khoi, vi du: [[(0,0), (0,2), (0,5), (0,6), (0,10)], ...]): ")
    try:
        block_positions = eval(block_positions_input)
    except:
        print("Lỗi: block_positions không đúng định dạng")
        exit()

    # Tách tin và in kết quả
    print(f"tach tin tu khung {frame_idx}, kenh {channel_name}")
    message_bits, character = extract_message_from_frame(video_path, frame_idx, channel_name, block_positions, Q=120)
    print(f"chuoi bit nhung: {message_bits}")
    