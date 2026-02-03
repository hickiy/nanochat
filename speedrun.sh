#!/bin/bash

# 这个脚本是“100 美元能买到的最佳 ChatGPT 克隆”，
# 设计为在 8XH100 节点上以 3 美元/GPU/小时运行约 4 小时。

# 1) 启动示例（最简单）：
# bash speedrun.sh
# 2) 在 screen 会话中启动的示例（因为运行需要约 4 小时）：
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) 使用 wandb 日志记录启动的示例，但请先设置 wandb：
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 修改训练数据源
export HF_ENDPOINT="https://hf-mirror.com"

# 默认中间产物目录在 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 使用 uv 的 Python venv 设置

# 安装 uv（如果尚未安装）
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建 .venv 本地虚拟环境（如果不存在）
[ -d ".venv" ] || uv venv
# 安装仓库依赖
uv sync --extra gpu
# 激活 venv，使 `python` 使用项目的 venv 而不是系统 python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb 设置
# 如果你希望使用 wandb 进行日志记录（很不错！推荐）。
# 1) 首先确保登录 wandb，例如运行：
#    `wandb login`
# 2) 运行此脚本时设置 WANDB_RUN 环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"：它作为特殊情况处理，跳过记录到 wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 在运行过程中，我们将在基础目录的 report/ 目录中编写 markdown 报告。
# 此命令清空它并写入包含大量系统信息和标记运行开始时间戳的头部部分。
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器

# 安装 Rust / Cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# 构建 rustbpe 分词器
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 下载预训练数据集的前约 2B 字符
# 有关此数据准备方式的详细信息请查看 dev/repackage_data_reference.py
# 每个数据分片约 250M 字符
# 所以我们下载 2e9 / 250e6 = 8 个数据分片
# 每个分片约 100MB 文本（压缩后），所以约 800MB 磁盘数据
python -m nanochat.dataset -n 8
# 立即在后台开始下载更多分片，同时分词器训练
# 有关为什么是 240 的原因请见下方注释
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# 在约 2B 字符的数据上训练词汇表大小为 2**16 = 65536 的分词器
python -m scripts.tok_train --max_chars=2000000000
# 评估分词器（报告压缩比等）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基础模型（预训练）

# d20 模型有 5.61 亿参数。
# Chinchilla 说 #tokens = 20X #params，所以我们需要 561e6 * 20 = 112 亿 token。
# 假设我们的分词器是 4.8 字符/token，这是 112 亿 * 4.8 ~= 540 亿字符。
# 每个分片 2.5 亿字符，这是 540 亿 / 2.5 亿 ~= 216 个分片需要用于预训练。
# 四舍五入到 240 以确保安全。每个分片约 100MB，这下载约 24GB 数据到磁盘。
# （整个数据集可用的总分片数是 1822。）
echo "等待数据集下载完成..."
wait $DATASET_DOWNLOAD_PID

# 使用的进程/GPU 数量
NPROC_PER_NODE=1
DEVICE_BATCH_SIZE=4

# 预训练 d20 模型
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN --device_batch_size=$DEVICE_BATCH_SIZE
# 在更大的训练/验证数据块上评估模型并抽取一些样本
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# 在 CORE 任务上评估模型
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# 中期训练（教模型对话特殊 token、工具使用、多项选择）

# 下载 2.3MB 的合成身份对话，为 nanochat 赋予个性
# 有关此数据准备方式以及如何轻松调整的详细信息请见 dev/gen_sft_data.py
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# 运行中期训练并评估模型
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN --device_batch_size=$DEVICE_BATCH_SIZE
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# 有监督微调（针对每行每个序列单独进行领域适应）

# 训练 sft 并立即重新评估（应该会看到小幅提升）
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# 通过 CLI 与模型聊天！去掉 -p 可以交互式聊天
# python -m scripts.chat_cli -p "Why is the sky blue?"

# 更好的方式，通过漂亮的 ChatGPT 风格 WebUI 与你的模型聊天
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 强化学习。可选，目前仅在 GSM8K 上
# （可选）

# 运行强化学习
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# 仅在 GSM8K 上评估 RL 模型
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# 通过将所有部分组合在一起生成完整报告
# report.md 是输出，将被复制到当前目录以方便使用
python -m nanochat.report generate
