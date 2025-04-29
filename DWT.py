import cv2
import numpy as np
import pywt

def normalize_dwt_frame(original_frame, stego_frame, 
                       alpha_ll_g=0.3, beta_ll_g=0.7, alpha_detail_g=0.1, beta_detail_g=0.9,
                       alpha_ll_br=0.9, beta_ll_br=0.1, alpha_detail_br=0.7, beta_detail_br=0.3, 
                       wavelet='haar'):
    """
    Chuẩn hóa khung hình bằng DWT để tăng PSNR, ít ảnh hưởng đến tin giấu trong DCT (kênh G).
    Args:
        original_frame: Khung gốc (H, W, 3, uint8).
        stego_frame: Khung sau nhúng DCT (H, W, 3, uint8).
        alpha_ll_g, beta_ll_g: Hệ số LL cho kênh G.
        alpha_detail_g, beta_detail_g: Hệ số LH, HL, HH cho kênh G.
        alpha_ll_br, beta_ll_br: Hệ số LL cho kênh B, R.
        alpha_detail_br, beta_detail_br: Hệ số LH, HL, HH cho kênh B, R.
        wavelet: Loại wavelet (mặc định 'haar').
    Returns:
        normalized_frame: Khung chuẩn hóa (H, W, 3, uint8).
    """
    assert alpha_ll_g + beta_ll_g == 1, "alpha_ll_g + beta_ll_g must equal 1"
    assert alpha_detail_g + beta_detail_g == 1, "alpha_detail_g + beta_detail_g must equal 1"
    assert alpha_ll_br + beta_ll_br == 1, "alpha_ll_br + beta_ll_br must equal 1"
    assert alpha_detail_br + beta_detail_br == 1, "alpha_detail_br + beta_detail_br must equal 1"
    
    # Chuẩn hóa điểm ảnh về [0, 1]
    original_float = original_frame.astype(float) / 255.0
    stego_float = stego_frame.astype(float) / 255.0
    
    normalized_frame = np.zeros_like(original_float)
    
    # Xử lý từng kênh (B, G, R)
    for c in range(3):
        original_channel = original_float[:, :, c]
        stego_channel = stego_float[:, :, c]
        
        # Tính DWT
        coeffs_orig = pywt.dwt2(original_channel, wavelet)
        coeffs_stego = pywt.dwt2(stego_channel, wavelet)
        
        cA_orig, (cH_orig, cV_orig, cD_orig) = coeffs_orig
        cA_stego, (cH_stego, cV_stego, cD_stego) = coeffs_stego
        
        # Chọn hệ số alpha, beta dựa trên kênh
        if c == 1:  # Kênh G (nơi nhúng tin)
            alpha_ll, beta_ll = alpha_ll_g, beta_ll_g
            alpha_detail, beta_detail = alpha_detail_g, beta_detail_g
        else:  # Kênh B, R
            alpha_ll, beta_ll = alpha_ll_br, beta_ll_br
            alpha_detail, beta_detail = alpha_detail_br, beta_detail_br
        
        # Kết hợp hệ số
        cA_normalized = alpha_ll * cA_orig + beta_ll * cA_stego
        cH_normalized = alpha_detail * cH_orig + beta_detail * cH_stego
        cV_normalized = alpha_detail * cV_orig + beta_detail * cV_stego
        cD_normalized = alpha_detail * cD_orig + beta_detail * cD_stego
        
        # Tái tạo khung bằng IDWT
        coeffs_normalized = (cA_normalized, (cH_normalized, cV_normalized, cD_normalized))
        normalized_channel = pywt.idwt2(coeffs_normalized, wavelet)
        
        # Cắt bỏ phần dư
        normalized_channel = normalized_channel[:original_channel.shape[0], :original_channel.shape[1]]
        normalized_frame[:, :, c] = normalized_channel
    
    # Khôi phục về [0, 255]
    normalized_frame = np.clip(normalized_frame * 255.0, 0, 255).astype(np.uint8)
    return normalized_frame

def create_normalized_video(original_video_path, stego_video_path, output_video_path, 
                           alpha_ll_g=0.2, beta_ll_g=0.8, alpha_detail_g=0.05, beta_detail_g=0.95,
                           alpha_ll_br=0.9, beta_ll_br=0.1, alpha_detail_br=0.7, beta_detail_br=0.3, 
                           wavelet='haar'):
    """
    Tạo video chuẩn hóa từ video gốc và video sau nhúng DCT, ít ảnh hưởng đến tin giấu.
    Args:
        original_video_path: Đường dẫn video gốc.
        stego_video_path: Đường dẫn video sau nhúng DCT.
        output_video_path: Đường dẫn video chuẩn hóa.
        alpha_ll_g, beta_ll_g, alpha_detail_g, beta_detail_g: Hệ số cho kênh G.
        alpha_ll_br, beta_ll_br, alpha_detail_br, beta_detail_br: Hệ số cho kênh B, R.
        wavelet: Loại wavelet.
    """
    cap_orig = cv2.VideoCapture(original_video_path)
    cap_stego = cv2.VideoCapture(stego_video_path)
    
    if not cap_orig.isOpened() or not cap_stego.isOpened():
        print("Lỗi: Không mở được video")
        cap_orig.release()
        cap_stego.release()
        return
    
    width = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_orig.get(cv2.CAP_PROP_FPS))
    frame_count_orig = int(cap_orig.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_stego = int(cap_stego.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count_orig != frame_count_stego:
        print("Lỗi: Số khung không khớp")
        cap_orig.release()
        cap_stego.release()
        return
    
    fourcc = 0  # Định dạng không nén
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap_orig.isOpened() and cap_stego.isOpened():
        ret_orig, frame_orig = cap_orig.read()
        ret_stego, frame_stego = cap_stego.read()
        
        if not ret_orig or not ret_stego:
            break
        
        normalized_frame = normalize_dwt_frame(frame_orig, frame_stego, 
                                             alpha_ll_g, beta_ll_g, alpha_detail_g, beta_detail_g,
                                             alpha_ll_br, beta_ll_br, alpha_detail_br, beta_detail_br, 
                                             wavelet)
        out.write(normalized_frame)
        
        frame_idx += 1
        print(f"Đã xử lý khung {frame_idx}/{frame_count_orig}")
    
    cap_orig.release()
    cap_stego.release()
    out.release()
    print(f"Đã tạo video chuẩn hóa: {output_video_path}")

if __name__ == "__main__":
    original_video = input("Nhập đường dẫn video gốc: ")
    stego_video = input("Nhập đường dẫn video sau nhúng DCT: ")
    output_video = 'normalized_stego_video.avi'
    
    create_normalized_video(original_video, stego_video, output_video)