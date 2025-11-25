from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader_with_state(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, device="cuda", resume_state_dict=None):
    """
    从 parquet 文件流式读取预训练文本，分词，产出训练批次。

    这个实现变得有点复杂，因为我们希望支持近似恢复训练。
    我们选择不将其变成类，而是在每个批次中返回 state_dict，
    然后调用者可以传入 state_dict 以从期望的点恢复训练。
    注意，为了简单起见，这种恢复目前只是*近似*的。
    我们不会重复相同的文档，但可能会跳过一些。
    返回的 state_dict 可以稍后通过 `resume_state_dict` 传入此函数以近似恢复。

    完美的状态恢复是可能的，但会复杂很多，目前可能不值得。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    # 无限迭代文档批次（文本字符串列表）
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        pq_idx = resume_pq_idx # 从恢复索引启动 parquet 文件（默认从 0 开始）
        while True: # 无限迭代（多轮）
            while pq_idx < len(parquet_paths): # 遍历所有 parquet 文件
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # 如果在同一文件上恢复则从恢复点开始，否则从 DDP rank 开始
                # 我知道这个状态恢复有点棘手，有点 hacky... 唉。
                if resume_rg_idx is not None:
                    base_idx = resume_rg_idx // ddp_world_size # 以 ddp_world_size 为单位
                    base_idx += 1 # 前进 1 以确保恢复后不会重复数据
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    resume_rg_idx = None # 设为 None 因为我们只想执行一次
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # 每个批次是一个 parquet 组，例如 1024 行
                    # 分词器编码可能想要以更小的批次进行，例如 128 行
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # 前进到下一个行组（在 DDP 中）
                pq_idx += 1 # 前进到下一个 parquet 文件
    batches = document_batches()

    # 现在产出 token 批次。
    needed_tokens = B * T + 1 # +1 是因为我们还需要最后一个 token 的目标
    # 获取分词器和 bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # 暂存缓冲区保存一次迭代的 token
    token_buffer = deque() # 我们从右边流入 token，从左边弹出
    while True:
        # 在产出之前累积足够的 token 用于一次迭代。
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # 将 token 从 deque 移动到暂存缓冲区
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA 支持内存固定以实现 CPU 和 GPU 之间的异步传输
        use_cuda_optimizations = device == "cuda"
        scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations) # 在 PyTorch 中，long=int64
        # 创建作为 1D 张量的输入/目标
        inputs_cpu = scratch[:-1]
        targets_cpu = scratch[1:]
        # 重塑为 2D 并异步移动到 GPU
        inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # 如果我们希望近似恢复训练，需要这个
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # 只产出 inputs/targets 而不产出 state_dict 的辅助函数
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
