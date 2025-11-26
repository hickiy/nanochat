#!/bin/bash

# nanochat 的 1000 美元级别
# 设计为在 8XH100 节点上端到端运行 $1000/24 ~= 41.6 小时
# 注释较少，更多详情请见 speedrun.sh

# 所有设置内容
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 在约 4B 字符上训练分词器并开始下载其余预训练数据
python -m nanochat.dataset -n 16
# 开始下载其余分片，共计 800 个（见下方解释为什么是 800）
python -m nanochat.dataset -n 800 &
# todo: 下载剩余部分
python -m scripts.tok_train --max_chars=4000000000
python -m scripts.tok_eval

# 记录我确定此 run1000.sh 脚本超参数的过程：
# 我们想要约 1000 美元的预算 ~= 41.6 小时的 8XH100 计算
# 1) 我猜测这个模型大小约为 depth=32
# 2) 确定适合的 device_batch_size：
# 使用 --depth=32 运行 base_train.py 脚本，我发现 --device_batch_size=16
# 会耗尽内存，但 --device_batch_size=8 可以。在训练期间检查 `nvidia-smi`，
# 我看到所有 GPU 约在 78/80GB VRAM，所以刚好合适，我们有约 50% 的良好 MFU。
# 所以训练脚本运行正常并显示：
# Vocab size: 65,536
# num_layers: 32
# model_dim: 2048
# num_heads: 16
# num_kv_heads: 16
# Tokens / micro-batch / rank: 8 x 2048 = 16,384
# Tokens / micro-batch: 131,072
# Total batch size 524,288 => gradient accumulation steps: 4
# Number of parameters: 1,879,048,192
# Estimated FLOPs per token: 1.207960e+10
# Calculated number of iterations from target data:param ratio: 71,680
# Total number of training tokens: 37,580,963,840
# Tokens : Params ratio: 20.00
# Total training FLOPs estimate: 4.539628e+20
# step 00004/71680 (0.01%) | loss: 8.813754 | lrm: 1.00 | dt: 1571.88ms | tok/sec: 83,385 | mfu: 50.92 | total time: 0.00m
# step 00005/71680 (0.01%) | loss: 8.488074 | lrm: 1.00 | dt: 1572.76ms | tok/sec: 83,338 | mfu: 50.89 | total time: 0.00m
# ...
# 3) 验证运行时间是否符合我们的预算：
# 训练脚本使用 Chinchilla 缩放定律来计算最优化 #tokens = 20 * #params。具体来说：
# 脚本显示我们将训练 71,680 步，每步需要 1.574s 所以：
# 预估训练时间：71,680 * 1.574s / 60 / 60 = 31.3 小时。
# 这没问题，符合我们的预算，还剩约 10 小时用于中期训练、SFT、评估和可能的 RL。
# 可能我们甚至可以容纳 depth=33 或 depth=34，但现在先这样。
# 4) 最后要注意的是运行所需的训练数据量。
# 上面的脚本计算出 "Total number of training tokens: 37,580,963,840"
# tok_eval.py 脚本报告默认分词器设置下平均约 4.8 字符/token。
# 所以 ~38B tokens * ~4.8 字符/token = ~185B 字符。
# 每个数据分片约 2.5 亿字符，所以我们需要 ~185B / 250M ~= 740 个分片。
# 为安全起见，我把它提升到 800 个分片，这就是为什么上面预下载数据集分片时使用 -n 800。
# 如果我们没有足够的数据，训练脚本会循环并对同一数据进行多个 epoch，
# 这会降低模型性能。可能 2、3 个 epoch 左右是可以的，但肯定不理想，
# 而到 10+ 个 epoch 我们会严重过拟合。
# 5) 就这样，其他所有内容（例如学习率）都由训练脚本自动调整。

# 使用的进程/GPU 数量
NPROC_PER_NODE=8

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=32 --device_batch_size=8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# 中期训练
# 注意：确保我们这里使用与基础训练脚本相同的 device_batch_size。
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --device_batch_size=8 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# sft
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# 生成最终报告
python -m nanochat.report generate

# 与模型对话
python -m scripts.chat_web
