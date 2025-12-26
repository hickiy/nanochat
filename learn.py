# from nanochat.tokenizer import get_tokenizer
# tokenizer = get_tokenizer()

# encoded = tokenizer.encode("the world is a beautiful place")
# print(f"Encoded tokens: {encoded}")  # 打印编码后的 token 列表

# decoded = tokenizer.decode(encoded)
# print(f"Decoded text: {decoded}")  # 打印解码后的文本

from nanochat.common import autodetect_device_type
device_type = autodetect_device_type()  # 调用自动检测设备类型函数
print(f"Detected device type: {device_type}")  # 打印检测到的设备类型