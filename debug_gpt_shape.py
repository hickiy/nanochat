import torch
from nanochat.gpt import GPT, GPTConfig

# 小规模配置，便于快速运行与调试
cfg = GPTConfig(sequence_len=16, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

m = GPT(cfg).to(device)
# 初始化权重（与库中初始化逻辑一致）
m.init_weights()

print("Total params:", sum(p.numel() for p in m.parameters()))

shapes = {}

def hook(module, inp, out):
    # 记录输出 shape，方便跟踪数据流
    try:
        shapes[f"{module.__class__.__name__}_{id(module)}"] = getattr(out, 'shape', None)
    except Exception:
        shapes[f"{module.__class__.__name__}_{id(module)}"] = str(type(out))

# 注册前向 hook 到每个 Block，以及 lm_head
for i, blk in enumerate(m.transformer.h):
    blk.register_forward_hook(hook)
    # 也注册到子模块（attn / mlp）以便更细粒度观察
    blk.attn.register_forward_hook(hook)
    blk.mlp.register_forward_hook(hook)

m.lm_head.register_forward_hook(hook)

# 构造样本输入并前向
idx = torch.randint(0, cfg.vocab_size, (2, 8), dtype=torch.long, device=device)
logits = m.forward(idx)

print("Logits shape:", logits.shape)
print("Registered module output shapes:")
for k, v in shapes.items():
    print(" -", k, v)

# 打印 rotary buffer 信息
try:
    print("rotary cos device/dtype:", m.cos.device, m.cos.dtype)
    print("rotary sin device/dtype:", m.sin.device, m.sin.dtype)
except Exception:
    print("No rotary buffers found")

print("Done.")
