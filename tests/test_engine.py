"""
测试 Engine 类。运行示例：

python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import KVCache

def test_kv_cache_resize():
    """
    KV 缓存没有正确调整大小，更多信息请见：
    https://github.com/karpathy/nanochat/pull/186
    此测试复现了该问题，并将与修复一起合并。
    """

    batch_size = 2
    num_heads = 3
    seq_len = 4
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # 插入单个 token 并为所有层设置不同的填充值
    def insert_token(token_idx):
        for layer_idx in range(num_layers):
            k = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx), dtype=torch.float32)
            v = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx * 100), dtype=torch.float32)
            kv_cache.insert_kv(layer_idx, k, v)

    # 插入 4 个 token（填满初始的 seq_len=4）
    for i in range(4):
        insert_token(i)

    # 记录缓存的原始状态
    original_cache = kv_cache.kv_cache.clone()
    original_seq_len = original_cache.shape[4]

    # 插入第 5 个 token，这将触发调整大小
    insert_token(4)
    # 验证缓存确实调整了大小
    new_seq_len = kv_cache.kv_cache.shape[4]
    assert new_seq_len > original_seq_len, f"Cache did not resize: original seq_len={original_seq_len}, new seq_len={new_seq_len}"

    # 验证调整大小后原始的 4 个 token 仍然完整
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # 检查调整大小后的缓存是否与预期值匹配
            expected_k = float(token_idx)
            expected_v = float(token_idx * 100)
            actual_k = kv_cache.kv_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = kv_cache.kv_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == expected_k).all(), f"Layer {layer_idx}, token {token_idx}: key corrupted, expected {expected_k}"
            assert (actual_v == expected_v).all(), f"Layer {layer_idx}, token {token_idx}: value corrupted, expected {expected_v}"
            # 并且原始缓存与调整大小后的缓存匹配
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), f"Layer {layer_idx}, token {token_idx}: key doesn't match original"
            assert (actual_v == original_v).all(), f"Layer {layer_idx}, token {token_idx}: value doesn't match original"
