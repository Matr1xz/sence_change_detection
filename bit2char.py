def bits_to_string(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break  # bỏ qua nếu không đủ 8 bit
        byte_str = ''.join(str(b) for b in byte)
        ascii_code = int(byte_str, 2)
        chars.append(chr(ascii_code))
    return ''.join(chars)

# Ví dụ:
def bit2char(bit_sequence):
    bit_sequence = [int(bit) for bit in bit_sequence if bit in '01']  # Chỉ lấy 0 và 1
    result = bits_to_string(bit_sequence)
    print(f'ky tu:{result}')  # Kết quả: AB
