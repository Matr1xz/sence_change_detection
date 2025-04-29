import cv2
import numpy as np
from scipy.fft import dct, idct

def embed_dct_8x8_quantization(block, bit, Q=120):
    """
    Nhúng bit vào hệ số DCT C(2, v) (|C| >= 100) trong hàng 2, ép đuôi 0 (bit 0) hoặc Q/2 (bit 1).
    Args:
        block: Khối 8x8 (float).
        bit: Bit nhúng (0 hoặc 1).
        Q: Ngưỡng quantization (mặc định 120).
    Returns:
        block_reconstructed: Khối tái tạo sau nhúng.
        embedded: True nếu nhúng thành công, False nếu không.
    """
    dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    row_2 = dct_block[2, :]
    candidates = np.where(np.abs(row_2) >= 100)[0]
    
    if len(candidates) == 0:
        return idct(idct(dct_block, axis=1, norm='ortho'), axis=0, norm='ortho'), False
    
    v = candidates[0]
    value = dct_block[2, v]
    
    k = np.round(value / Q)
    if bit == 0:
        target = k * Q
    else:
        target1 = (k - 1) * Q + Q / 2
        target2 = k * Q + Q / 2
        if abs(target2) >= 100:
            target = target2
        elif abs(target1) >= 100:
            target = target1
        else:
            return idct(idct(dct_block, axis=1, norm='ortho'), axis=0, norm='ortho'), False
    
    dct_block[2, v] = target
    block_reconstructed = idct(idct(dct_block, axis=1, norm='ortho'), axis=0, norm='ortho')
    return block_reconstructed, True

def embed_8bits_with_redundancy(frame, message_bits, Q=120):
    """
    Nhúng 8 bit vào 40 khối 8x8 đầu tiên của kênh G, mỗi bit nhúng vào 5 khối liên tiếp.
    Args:
        frame: Khung màu (H, W, 3, uint8).
        message_bits: Danh sách 8 bit cần nhúng.
        Q: Ngưỡng quantization.
    Returns:
        frame_reconstructed: Khung tái tạo (float).
        embedded_success: True nếu nhúng đủ 8 bit (40 khối), False nếu không.
        block_indices: Danh sách 8 nhóm, mỗi nhóm 5 chỉ số khối [(i,j), (i,j), (i,j), (i,j), (i,j)].
    """
    block_size = 8
    height, width, _ = frame.shape
    frame_float = frame.astype(float)
    frame_reconstructed = np.zeros_like(frame_float)
    block_indices = []  # Lưu 8 nhóm, mỗi nhóm 5 khối
    bits_embedded = 0
    
    channels = cv2.split(frame_float)
    
    for c, channel_name in enumerate(['B', 'G', 'R']):
        channel = channels[c]
        reconstructed_channel = np.zeros_like(channel)
        
        if channel_name == 'G' and len(message_bits) == 8:
            bit_index = 0
            current_group = []  # Lưu 5 khối cho mỗi bit
            block_count = 0  # Đếm tổng số khối đã duyệt
            
            # Duyệt khối theo thứ tự từ trái sang phải, trên xuống dưới
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    if bit_index >= 8:  # Đã nhúng đủ 8 bit
                        break
                    block = channel[i:i+block_size, j:j+block_size]
                    if block.shape != (block_size, block_size):
                        reconstructed_channel[i:i+block_size, j:j+block_size] = block
                        continue
                    
                    block_reconstructed, embedded = embed_dct_8x8_quantization(
                        block, message_bits[bit_index], Q)
                    reconstructed_channel[i:i+block_size, j:j+block_size] = block_reconstructed
                    block_count += 1
                    
                    if embedded:
                        current_group.append((i // block_size, j // block_size))
                        if len(current_group) == 5:  # Đủ 5 khối cho bit hiện tại
                            block_indices.append(current_group)
                            current_group = []
                            bit_index += 1
                            bits_embedded = bit_index
                    
                    # Nếu duyệt quá nhiều khối mà chưa đủ 8 bit
                    if block_count >= 100 and bits_embedded < 8:
                        return frame_float, False, block_indices
                
                if bit_index >= 8:
                    break
            
            # Nếu không đủ 8 bit (40 khối)
            if bits_embedded < 8:
                return frame_float, False, block_indices
        else:
            reconstructed_channel = channel.copy()
        
        frame_reconstructed[:, :, c] = reconstructed_channel
    
    return frame_reconstructed, True, block_indices

# Nhập chuỗi từ người dùng
message = input("nhap ma sinh vien (10 ky tu): ")
if len(message) < 10:
    print("Lỗi: Chuỗi phải có ít nhất 10 ký tự")
    exit()

# Chuyển 10 ký tự thành 80 bit (1 ký tự = 8 bit mỗi khung)
message_bits = []
for char in message[:10]:  # Lấy 10 ký tự đầu
    bits = [int(b) for b in format(ord(char), '08b')]
    message_bits.extend(bits)
print(f"tong so bit can nhung {len(message_bits)} (10 ky tu x 8 bit)")

# Chia thành 10 nhóm 8 bit
bit_groups = [message_bits[i:i+8] for i in range(0, 80, 8)]
if len(bit_groups) != 10:
    print("Lỗi: Không đủ bit cho 10 khung")
    exit()

# Nhập thông tin video
input_video = input("nhap duong dan video dau vao: ")
output_video = 'stego_video_msv.avi'
start_frame = int(input("nhap khun bat dau (mac dinh 2): ") or 2)

cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Lỗi: Không mở được video")
    exit()

# Lấy thông tin video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Kiểm tra số khung đủ cho 10 khung từ start_frame
if frame_count < start_frame + 10:
    print(f"loi video chi co {frame_count} khung, cần ít nhất {start_frame + 10} khung")
    cap.release()
    exit()

# Tạo video đầu ra
fourcc = 0
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Nhúng 8 bit vào 10 khung, mỗi bit vào 5 khối
frame_idx = 0
group_idx = 0
all_block_indices = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if start_frame <= frame_idx < start_frame + 10 and group_idx < 10:
        # Nhúng 8 bit của nhóm hiện tại
        frame_reconstructed, embedded_success, block_indices = embed_8bits_with_redundancy(
            frame, bit_groups[group_idx], Q=120)
        if embedded_success:
            #print(f"Khung {frame_idx}: Đã nhúng 8 bit (ký tự {group_idx+1})")
            if frame_idx == start_frame:
                with open('block_indices.txt', 'w') as f:
                    f.write(f"Khung {frame_idx}: {block_indices}\n")
            else:
                with open('block_indices.txt', 'a') as f:
                    f.write(f"Khung {frame_idx}: {block_indices}\n")
            print(f"{block_indices}")
            print('')
            all_block_indices[frame_idx] = block_indices
            group_idx += 1
        else:
            print(f"loi khung {frame_idx} khong du 40 khoi hop le de nhung 8 bit")
            cap.release()
            out.release()
            exit()
    else:
        frame_reconstructed = frame.astype(float)
    
    frame_reconstructed = np.clip(frame_reconstructed, 0, 255).astype(np.uint8)
    out.write(frame_reconstructed)
    
    frame_idx += 1

cap.release()
out.release()

if group_idx < 10:
    print(f"loi chi nhung duoc {group_idx} khung, khong du 10 khung")
else:
    print(f"da tao video nhung: {output_video}")
    print(f"vi tri khoi nhung: {all_block_indices}")