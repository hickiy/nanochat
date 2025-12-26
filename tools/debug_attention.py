import torch
import matplotlib.pyplot as plt
from nanochat.gpt import GPT, GPTConfig, apply_rotary_emb, norm

# 该脚本捕获第一个 Block 的 attention q/k/v，并绘制第一头的 attention scores heatmap
cfg = GPTConfig(sequence_len=32, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2, n_embd=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

m = GPT(cfg).to(device)
m.init_weights()

attention_data = {}

def attn_hook(module, inp, out):
    # inp: (x, cos_sin, kv_cache)
    try:
        x = inp[0]
        cos_sin = inp[1]
        # 重新计算 q/k/v（与模块内部逻辑一致）
        B, T, C = x.size()
        q = module.c_q(x).view(B, T, module.n_head, module.head_dim)
        k = module.c_k(x).view(B, T, module.n_kv_head, module.head_dim)
        v = module.c_v(x).view(B, T, module.n_kv_head, module.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, H, T, D)
        attention_data['q'] = q.detach().cpu()
        attention_data['k'] = k.detach().cpu()
        attention_data['v'] = v.detach().cpu()
    except Exception as e:
        print('hook error:', e)

# 仅注册第一个 Block 的 attention hook
m.transformer.h[0].attn.register_forward_hook(attn_hook)

# 构造输入并运行
idx = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long, device=device)
_ = m.forward(idx)

if 'q' not in attention_data:
    print('No attention data captured')
else:
    q = attention_data['q'][0]  # (H, T, D)
    k = attention_data['k'][0]
    # 计算 scores: (H, T, T)
    D = q.size(-1)
    scores = torch.einsum('htd,hsd->hts', q, k) / (D ** 0.5)
    scores = scores.numpy()
    # 绘制第一头的热力图
    plt.figure(figsize=(6, 5))
    plt.title('Attention scores (head 0)')
    plt.imshow(scores[0], aspect='auto', cmap='viridis')
    plt.colorbar()
    out_path = 'attention_heatmap_head0.png'
    plt.savefig(out_path)
    print('Saved attention heatmap to', out_path)
