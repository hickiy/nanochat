#!/bin/bash

# 展示在 CPU（或 Macbook 上的 MPS）上运行一些代码路径的示例
# 运行方式：
# bash dev/cpu_demo_run.sh

# 注意：训练 LLM 需要 GPU 计算和 $$$。你不会在 Macbook 上走得很远。
# 把这个运行理解为教育/有趣的演示，而不是你应该期望运行良好的东西。
# 这也是为什么我把这个脚本藏在 dev/ 里的原因

# 所有设置内容
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 清空报告
python -m nanochat.report reset

# 在约 1B 字符上训练分词器
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

# 在 CPU 上训练一个非常小的 4 层模型
# 每个优化步骤处理单个 1024 token 的序列
# 我们只运行 50 步优化（增加这个值可以获得更好的结果）
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --core_metric_every=50 \
    --core_metric_max_per_task=12 \
    --sample_every=50 \
    --num_iterations=50
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# 中期训练
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024 \
    --num_iterations=100
# 评估结果会很糟糕，这只是为了执行代码路径。
# 注意我们将执行内存限制降低到 1MB 以避免在较小系统上出现警告
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --num_iterations=100 \
    --eval_steps=4 \
    --eval_metrics_max_problems=16

# Chat CLI
# python -m scripts.chat_cli -p "为什么天空是蓝色的？"

# Chat Web
# python -m scripts.chat_web

python -m nanochat.report generate
