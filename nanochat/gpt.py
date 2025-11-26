"""
GPT 模型（重写版，简化了很多）
主要特性：
- 旋转位置编码（无位置嵌入）
- QK 归一化
- 词嵌入和 lm_head 权重不共享
- MLP 中使用 relu^2 激活函数
- 词嵌入后进行归一化
- RMSNorm 无可学习参数
- 线性层无偏置
- 支持分组查询注意力（GQA）以提高推理效率
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # 纯函数式 RMSNorm，无可学习参数
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # 多头注意力
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 将最后一维分成两半
    y1 = x1 * cos + x2 * sin # 旋转成对的维度
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # 重新组装
    out = out.to(x.dtype) # 确保输入/输出数据类型匹配
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # 投影输入以获得查询、键和值
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 对查询和键应用旋转嵌入以获得相对位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK 旋转嵌入
        q, k = norm(q), norm(k) # QK 归一化
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # 将头维度作为批次维度，即 (B, T, H, D) -> (B, H, T, D)

        # 应用 KV 缓存：将当前 k,v 插入缓存，获取到目前为止的完整视图
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # 本次前向传播的查询数量
        Tk = k.size(2) # 键/值的总数量（缓存中的 + 当前前向传播的）

        # 注意力：查询自回归地关注键/值。需要处理几种情况：
        enable_gqa = self.n_head != self.n_kv_head # 分组查询注意力（GQA）：如需要则复制键/值头以匹配查询头数量
        if kv_cache is None or Tq == Tk:
            # 训练时（无 KV 缓存），使用常规因果注意力
            # 即使有 KV 缓存，当 Tq == Tk 时也可以使用这个简单版本
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # 推理时但本次前向传播只有单个查询：
            # 查询需要关注缓存中的所有键/值
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # 推理时且本次前向传播有一批查询：
            # 首先，每个查询关注所有缓存的键/值（即完整前缀）
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = 保留, False = 屏蔽
            prefix_len = Tk - Tq
            if prefix_len > 0: # 不会为负但可能为零
                attn_mask[:, :prefix_len] = True
            # 然后，在这批查询内部使用因果注意力
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # 将各头并排重组并投影回残差流
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 为支持 meta 设备初始化，我们在这里初始化旋转嵌入，但这是假的
        # 关于 rotary_seq_len，这些旋转嵌入在内存中相当小/便宜，
        # 所以让我们过度计算它们，但如果达到该数量就触发断言失败。
        # 将来可以动态增长缓存，目前这样就够了。
        self.rotary_seq_len = config.sequence_len * 10 # 10 倍过度计算应该足够了，TODO 改进？
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False 表示不会保存到检查点
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # 将分类器权重初始化为零
        torch.nn.init.zeros_(self.lm_head.weight)
        # 将所有块中的 c_proj 权重初始化为零
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # 初始化旋转嵌入
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # 将嵌入从 fp32 转换为 bf16：优化器可以容忍这一点，而且可以节省内存：模型和激活值都能受益
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: 提高基础 theta，例如 100K 是最近更常用的值
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # 从模型嵌入自动检测设备
        if device is None:
            device = self.transformer.wte.weight.device
        # 对通道进行步进
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 对时间步进行步进
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个（时间，通道）对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # 保持为 bfloat16 格式
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # 添加批次和头维度以便后续广播
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ 返回模型每个 token 的估计 FLOPs。参考：https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # 将所有参数分为 3 组（矩阵、嵌入、lm_head）
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # 为嵌入和 lm_head 创建 AdamW 优化器
        # 按 ∝1/√dmodel 缩放 AdamW 参数的学习率（学习率已针对 768 维模型调优）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # 为线性层创建 Muon 优化器
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # 将两个优化器合并到一个列表中
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # 获取当前序列长度的旋转嵌入（形状为 (1, seq_len, 1, head_dim/2)）
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # 如果存在 KV 缓存，需要将旋转嵌入偏移到缓存中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # 将缓存截断到当前序列长度

        # 前向传播 Transformer 主干
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # 前向传播 lm_head（计算 logits）
        softcap = 15
        if targets is not None:
            # 训练模式：计算并返回损失
            # TODO: 尝试 Liger Kernels / 分块交叉熵等
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits 软上限
            logits = logits.float() # 使用 tf32/fp32 计算 logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理模式：计算并返回 logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits 软上限
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        朴素的自回归流式推理。
        为简单起见，假设：
        - 批次大小为 1
        - ids 和生成的 token 是简单的 Python 列表和整数
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # 添加批次维度
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
