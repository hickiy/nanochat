"""
一些帮助评估基础模型的函数。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    此函数返回每字节比特数（bpb），而非朴素的"平均损失"。
    这是一个与分词器词汇表大小无关的指标，意味着即使你改变
    词汇表大小，你仍然是在比较同类项。其工作原理是：不是像往常
    那样计算平均损失，而是计算总损失，同时独立计算总字节数
    （所有目标 token 的），然后相除。这将损失按目标 token
    所代表的字节数进行归一化。

    增加的复杂性是为了：
    1) 所有"普通"token 按其字节长度归一化
    2) 指标中不包含特殊 token（例如 <|bos|>）- 它们被屏蔽掉。
    3) 指标中不包含主动屏蔽的 token（使用例如 -1 的 ignore_index）。

    除了 evaluate_loss，我们还需要 token_bytes 张量：
    它是形状为 (vocab_size,) 的 1D 张量，指示每个
    token id 的字节数，如果 token 不被计数则为 0（例如特殊 token）。
    """
    # 记录损失
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # 展平
        y = y.view(-1) # 展平
        if (y.int() < 0).any(): # mps 目前没有 int64 的 < 0 内核，只有 int32
            # 如果某些目标 token 是 ignore_index（例如 -1），则需要稍微复杂的代码路径
            # 任何 < 0 的目标 token 都将被忽略：不要用负数索引 token_bytes
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # 将有效目标映射到其字节长度；忽略的目标贡献 0 字节
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # 快速路径：没有忽略的目标，可以安全地直接索引
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # 在所有进程间求和归约
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # 将两者移到 cpu，计算 bpb 并返回
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb
