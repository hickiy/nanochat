"""
训练模型。运行方式：

python base_train.py

或分布式运行：

torchrun --nproc_per_node=8 base_train.py

如果你只有 CPU/Macbook，你需要训练一个更小的 LLM。示例：
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# 用户设置
run = "dummy" # wandb 运行名称默认值（"dummy" 是特殊的 - 我们不会记录到 wandb）
# 运行时
device_type = "" # cuda|cpu|mps（空 => 自动检测良好的设备类型默认值，优先顺序：CUDA > MPS > CPU）
# 模型架构
depth = 20 # 要训练的 Transformer 模型深度，其余 kwargs 由此派生
max_seq_len = 2048 # 最大上下文长度
# 训练时长。以下 3 个中只有一个会被使用，按此优先顺序。
num_iterations = -1 # 显式的优化步数（-1 = 禁用）
target_flops = -1.0 # 计算 num_iterations 以达到 target_flops。用于缩放定律实验（-1 = 禁用）
target_param_data_ratio = 20 # 计算 num_iterations 以保持固定的数据:参数比（Chinchilla=20）（-1 = 禁用）
# 优化
device_batch_size = 32 # 每设备批次大小（设置为不会 OOM）
total_batch_size = 524288 # 期望的总批次大小，以 #tokens 为单位
embedding_lr = 0.2 # 嵌入参数的学习率（Adam）
unembedding_lr = 0.004 # 反嵌入参数的学习率（Adam）
weight_decay = 0.0 # 嵌入/反嵌入参数的权重衰减（Adam）
matrix_lr = 0.02 # 矩阵参数的学习率（Muon）
grad_clip = 1.0 # 梯度裁剪值（0.0 = 禁用）
warmup_ratio = 0.0 # 学习率预热的迭代比例
warmdown_ratio = 0.2 # 学习率衰减的迭代比例
final_lr_frac = 0.0 # 最终学习率是初始学习率的这个比例
resume_from_step = -1 # 从此优化步骤恢复训练（-1 = 禁用）
# 评估
eval_every = 250 # 每多少步评估一次验证 bpb
eval_tokens = 20*524288 # 评估验证损失的 token 数量
core_metric_every = 2000 # 每多少步评估一次核心指标（-1 = 禁用）
core_metric_max_per_task = 500 # 估计核心指标时每个任务的样本数
sample_every = 2000 # 每多少步从模型采样
save_every = -1 # 每多少步保存模型检查点（-1 = 禁用，仅在运行结束时保存）
# 输出
model_tag = "" # 可选覆盖输出检查点目录名的模型标签
# 现在允许 CLI 通过配置器覆盖设置，哈哈
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# 计算初始化
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # 此进程将执行日志记录、检查点保存等
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb 日志初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# 分词器将用于评估，我们也需要词汇表大小
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# 模型 kwargs 由期望的模型深度派生
num_layers = depth
model_dim = depth * 64 # 宽高比 64（通常随着模型大小增加从 64 变到 128）
num_heads = max(1, (model_dim + 127) // 128) # 头维度 128（这里的除法是向上取整）
num_kv_heads = num_heads # 默认是 1:1 GQA（分组查询注意力）比例（即禁用 GQA）
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# 优化器 / 数据 / 训练长度相关的超参数
# 计算达到期望总批次大小所需的梯度累积
tokens_per_fwdbwd = device_batch_size * max_seq_len # 单个进程每次迭代的 token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有进程每次迭代的总 token 数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# 初始化模型

# 使用随机权重创建新模型
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config)
model.to_empty(device=device)
model.init_weights()

# 如果我们正在恢复，用检查点的参数覆盖模型参数
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # 复制后释放此内存

orig_model = model # 原始的、未编译的模型，用于保存原始模型 state_dict 和推理/评估（因为形状可能会改变）
model = torch.compile(model, dynamic=False) # 模型的输入永远不会改变形状，所以 dynamic=False 是安全的
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 计算迭代次数。要么给定，要么从目标 flops 计算，要么从目标数据:参数比计算（按此顺序）
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # 从目标 flops 计算迭代次数
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # 从目标参数数据比计算迭代次数
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # 释放内存

# -----------------------------------------------------------------------------
# 初始化训练/验证数据加载器
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # 启动第一批数据的加载

# -----------------------------------------------------------------------------
# 设置超参数调度器

# 学习率调度器
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Muon 优化器的动量调度器
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# 循环状态（训练循环更新的变量）

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # 训练损失的 EMA
    total_training_time = 0 # 训练的总墙钟时间
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# 训练循环
while True:
    last_step = step == num_iterations # 循环运行 num_iterations+1 次，以便我们可以在最后评估/保存
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 定期：评估验证 bpb（所有进程参与）
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 定期：估计 CORE 指标（所有进程参与）
    # 使用原始未编译的模型，因为输入形状不断变化
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # 定期：从模型采样（仅在主进程上）
    # 使用原始未编译的模型，因为输入形状不断变化
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # 保存检查点：在运行结束时，或每 save_every 步，除了第一步或恢复步
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # 模型参数
            [opt.state_dict() for opt in optimizers], # 优化器状态
            { # 保存为 json 的元数据
                "step": step,
                "val_bpb": val_bpb, # 最后一步的损失
                "model_config": model_config_kwargs,
                "user_config": user_config, # 训练脚本的输入
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # 所有循环状态（除了 step），以便我们可以恢复训练
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # 终止条件（TODO：可能还要添加损失爆炸等）
    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单个训练步骤
    # 计算梯度
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # 用于日志记录
        loss = loss / grad_accum_steps # 每个 .backward() 是梯度求和 => 在这里归一化损失
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # 在 GPU 忙于前向/后向时预取下一批
    # 梯度裁剪
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU 张量 -> CPU 浮点数（注意：cpu-gpu 同步点）
    # 更新优化器
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 日志记录
    ema_beta = 0.9 # EMA 衰减因子，用于更平滑的日志记录
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA 训练损失
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 去偏 EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM 无 2:4 稀疏性
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # 以 % 表示
    if step > 10:
        total_training_time += dt # 只计算前 10 步之后的时间
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        wandb_run.log(log_data)

    # 状态更新
    step += 1

# 打印更多统计信息
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# 记录到报告
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI 参数
    { # 关于训练设置的统计信息
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # 关于训练结果的统计信息
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# 清理
wandb_run.finish() # wandb 运行完成
compute_cleanup()
