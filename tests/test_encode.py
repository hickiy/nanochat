from nanochat.tokenizer import get_tokenizer
tokenizer = get_tokenizer()

encoded = tokenizer.encode("the world is a beautiful place")
print(f"Encoded tokens: {encoded}")  # 打印编码后的 token 列表

decoded = tokenizer.decode(encoded)
print(f"Decoded text: {decoded}")  # 打印解码后的文本