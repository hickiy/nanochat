"""
通过 "GRPO" 在 GSM8K 上进行强化学习。

我把 GRPO 加引号是因为我们最终得到的东西比它更简单，
更类似于 REINFORCE：

1) 删除信任区域，所以没有对参考模型的 KL 正则化
2) 我们是在策略上的，所以不需要 PPO 比率+裁剪。
3) 我们使用 GAPO 风格的归一化，是 token 级别的，而不是序列级别的。
4) 不使用 z-score 归一化 (r - mu)/sigma，只使用 (r - mu) 作为优势。

1 GPU:
python -m scripts.chat_rl

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl -- --run=default
"""

import os
import itertools
import re
import wandb
import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# RL 超参数
run = "dummy" # wandb 运行名称
source = "sft" # mid|sft
dtype = "bfloat16"
device_batch_size = 8 # 没有前向传播会超过这个以避免 OOM
examples_per_step = 16 # 总共且跨所有进程（注意：是样本/问题，不是采样/补全！）
num_samples = 16 # 每个样本/问题的采样数量
max_new_tokens = 256
temperature = 1.0
top_k = 50 # TODO: 尝试 None?
unembedding_lr = 0.004
embedding_lr = 0.2
matrix_lr = 0.02
weight_decay = 0.0
init_lr_frac = 0.05
num_epochs = 1 # 在 gsm8k 上训练多少轮
save_every = 60 # 每多少步保存模型
eval_every = 60 # 每多少步评估模型的验证 pass@k
eval_examples = 400 # 用于评估 pass@k 的样本数量
# 现在允许 CLI 通过配置器覆盖设置，哈哈
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# 初始化计算/精度
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0 # 此进程将执行日志记录、检查点保存等
dtype = torch.float32 if dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

# wandb 日志初始化
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=run, config=user_config)

# 初始化模型和分词器
model, tokenizer, meta = load_model(source, device, phase="eval")
engine = Engine(model, tokenizer) # 用于采样 rollouts

# -----------------------------------------------------------------------------
# Rollout / 采样生成器循环，产出用于训练的样本批次

train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // examples_per_step) * num_epochs
print0(f"Calculated number of steps: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # 可以使用此 token，它仅用于填充且不参与损失。
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # 每个进程负责训练数据中的不同样本
    for example_idx in itertools.cycle(rank_indices):

        # 首先获取用户和助手消息的完整对话
        conversation = train_task[example_idx]

        # 对对话进行分词，删除最后一条助手消息并准备助手进行补全
        # （即保留 <|assistant_start|>，但删除其后的所有内容）
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # 使用批量生成生成 num_samples 个样本，使用循环避免 OOM
        model.eval() # 确保模型处于评估模式
        generated_token_sequences = []
        masks = []
        num_sampling_steps = num_samples // device_batch_size # go sequentially to prevent OOMs
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
            with autocast_ctx:
                generated_token_sequences_batch, masks_batch = engine.generate_batch(
                    tokens,
                    num_samples=device_batch_size,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed, # 确保每个采样步骤改变种子
                )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # 计算每个样本的奖励
        rewards = []
        for sample_tokens in generated_token_sequences:
            # 只获取生成的 token（在提示之后）
            generated_tokens = sample_tokens[prefix_length:]
            # 将生成的响应解码为文本
            generated_text = tokenizer.decode(generated_tokens)
            # 计算奖励
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # 填充序列使其长度（在时间维度上）匹配
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # 将序列和掩码堆叠成 PyTorch 张量
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # 生成 Transformer 的自回归输入和目标
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone 以避免原地修改：
        targets[mask_ids[:, 1:] == 0] = -1 # <-- 这里是原地修改。-1 是忽略索引
        # 注意 Engine 对提示 token 和工具使用 token 都返回 mask=0。
        # 所以我们会（正确地）不在提示 token 或工具使用强制 token 上训练。
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # 通过简单减去均值计算优势（而不是 z-score (x-mu)/sigma）
        mu = rewards.mean()
        advantages = rewards - mu
        # 产出 inputs/targets 作为 (B, T) 的 id 和 rewards 作为 (B,) 的浮点数
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# GSM8K pass@k 的简单评估循环
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    评估 GSM8K 任务并返回评估结果的记录列表。
    在分布式环境中，所有进程协作但此函数不会
    跨进程进行归约。这是调用者的责任。
    因为评估可能需要一段时间，此函数会逐个产出记录。
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # 在 Engine 内部使用批量生成生成 k 个样本
        assert num_samples <= device_batch_size # 通常这是真的。如果不是我们可以添加循环...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # 检查每个样本是否正确
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # 有点臃肿因为我曾想做更复杂的日志记录。
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# 训练循环

# 初始化优化器
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay,
)

# 将初始学习率设为基础学习率的一个比例
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * init_lr_frac
        group["initial_lr"] = group["lr"] # 保存初始学习率以便后续轻松衰减

# 学习率调度器：在 num_steps 上简单线性衰减到零
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# 计算每个进程处理多少样本以达到期望的 examples_per_step
print0(f"Total sequences per step: {examples_per_step * num_samples}") # 每步的总批次大小（序列数）
assert examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
examples_per_rank = examples_per_step // ddp_world_size # per GPU
print0(f"Calculated examples per rank: {examples_per_rank}")

# Kick off the training loop
batch_iterator = get_batch()
for step in range(num_steps):

    # 定期评估模型并记录到 wandb
    if step % eval_every == 0:
        model.eval()
        passk = torch.zeros(device_batch_size, device=device) # pass@k for k=1..device_batch_size
        with autocast_ctx:
            records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=device_batch_size, max_examples=eval_examples, temperature=1.0)
            records = list(records_iter) # collect all records
        for k in range(1, device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # 按记录总数归一化
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # 在数据集中多个样本上进行前向/后向传播
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # 获取对应训练数据集中一个样本的一批
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # 评估损失和梯度
        model.train() # 确保模型处于训练模式
        # 我们还需要一个循环因为永远不能超过 device_batch_size
        assert inputs_all.size(0) % device_batch_size == 0
        num_passes = inputs_all.size(0) // device_batch_size
        for pass_idx in range(num_passes):
            # 提取此次传递的批次
            b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # 计算对数概率。注意损失计算 NLL = -logp，所以我们取负
            with autocast_ctx:
                logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # 计算 PG 目标。注意 ignore_index=-1 确保无效 token 的损失为 0。
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # 按有效 token 数量、传递次数和 examples_per_rank 归一化
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # 注意，不需要添加 PPO 比率+裁剪因为我们是在策略上的
            # 最后，将我们想要最大化的目标转换为我们想要最小化的损失
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # 用于日志记录
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # 记录此步 rollouts 的情况
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # 跨进程聚合
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # 更新模型参数
    lrm = get_lr_multiplier(step)
    for opt in optimizers: # 首先设置学习率
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    for opt in optimizers: # 然后更新优化器
        opt.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # 主进程定期保存模型。跳过第一步。保存最后一步。
    if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        model_tag = f"d{depth}" # 基于基础模型的深度命名模型标签
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
        model_config_kwargs = model.config.__dict__ # 有点不规范，滥用了 GPTConfig 的简单性，TODO 改进
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # 注意：我们不保存优化器状态
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# 记录到报告
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI 参数
])

wandb_run.finish() # wandb 运行完成
compute_cleanup()
