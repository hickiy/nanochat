# NanoChat 快速参考指南

## 📌 核心概念速查表

### 1. 模型规格速查

```
模型规格表:

属性              | d20 ($100)    | d26 ($300)    | d32 ($1000)
─────────────────────────────────────────────────────
参数数量          | 561M          | 1.3B          | 1.9B
层数 (depth)      | 20            | 26            | 32
注意力头数        | 6             | 8             | 8
KV 头数 (GQA)     | 6             | 8             | 8
嵌入维度          | 768           | 1024          | 1024
上下文长度        | 2048          | 2048          | 2048
训练令牌数        | 11.2B         | 26B           | 38B
预计训练时间      | ~4 小时       | ~12 小时      | ~41.6 小时
GPU 配置          | 8×H100        | 8×H100        | 8×H100
预计成本          | $100          | $300          | $1000

计算说明:
- Chinchilla 定律: tokens = 20 × params
- tokens = 2.5B 参数 × 20 = 11.2B
- 分词比率: 4.8 字符/令牌
- 字符数 = 11.2B × 4.8 = 53.7B 字符
```

### 2. 关键参数配置

```python
# nanochat/gpt.py 中的默认配置

GPTConfig 默认值:
├─ sequence_len = 1024          # 最大序列长度
├─ vocab_size = 50304           # 词汇表大小 (65536 for tok)
├─ n_layer = 12                 # Transformer 层数
├─ n_head = 6                   # 查询头数
├─ n_kv_head = 6                # 键值头数
└─ n_embd = 768                 # 嵌入维度

# scripts/base_train.py 中的优化参数

device_batch_size = 32          # 单 GPU 上的序列数
total_batch_size = 524288       # 全局批次大小（令牌数）
max_seq_len = 2048              # 最大序列长度

学习率:
├─ embedding_lr = 0.2           # Embedding 参数 (AdamW)
├─ matrix_lr = 0.02             # 矩阵参数 (Muon)
├─ unembedding_lr = 0.004       # Output projection (AdamW)
└─ init_lr_frac = 1.0           # 初始学习率比例

正则化:
├─ grad_clip = 1.0              # 梯度裁剪
├─ weight_decay = 0.0           # AdamW 权重衰减
└─ dropout = 0.0                # Dropout (未使用)

target_param_data_ratio = 20    # Chinchilla 定律
```

### 3. 文件执行时间参考

```
在 8×H100 节点上执行时间 (speedrun.sh 中):

阶段                          | 时间    | 累计
────────────────────────────────────────────
依赖安装 & 环境设置           | 10 分钟 | 0:10
分词器训练                    | 15 分钟 | 0:25
数据下载 (240 个分片)         | 30 分钟 | 0:55 (后台)
预训练 (base_train)           | 60 分钟 | 2:00
预训练评估 (base_eval/loss)   | 20 分钟 | 2:20
中间训练 (mid_train)          | 40 分钟 | 3:00
监督微调 (chat_sft)           | 30 分钟 | 3:30
最终评估                      | 10 分钟 | 3:40
报告生成                      | 5 分钟  | 3:45

总时间: ~4 小时

性能指标 (d20, 8×H100):
- 令牌吞吐量: ~200K tokens/sec
- GPU 利用率: 85-90%
- 显存占用: ~40 GB (总)/GPU
- 通信开销: ~15%
```

### 4. 数据流路径

```
原始文本数据流:
  Raw Text (互联网) 
    ↓ [Dataset.download()]
  .jsonl 文件 (~250MB/文件)
    ↓ [Parquet 转换]
  .parquet 文件 (行组优化)
    ↓ [DDP 分片读取]
  Text 字符串 (batch)
    ↓ [分词器]
  Token IDs [B, T]
    ↓ [模型]
  Logits [B, T, vocab_size]
    ↓ [损失计算]
  Loss (标量)
    ↓ [反向传播]
  Gradients
    ↓ [AllReduce (DDP)]
  Averaged Gradients
    ↓ [Optimizer Step]
  Updated Weights ✓
```

### 5. 任务类型速查

```python
# tasks/ 中可用的任务

对话类:
├─ SmolTalk()                # 一般对话数据
├─ CustomJSON()              # 自定义 JSON 格式
└─ <|特殊令牌|>             # 对话标记

评估类 (多项选择):
├─ ARC()                     # ARC Challenge (Science)
├─ MMLU()                    # 多领域知识
└─ 准确率评估

计算类:
├─ GSM8K()                   # 小学数学
└─ HumanEval()               # Python 编码

技能培养:
└─ SpellingBee()             # 拼写/计数任务

集合:
├─ TaskMixture()             # 随机混合多个任务
└─ TaskSequence()            # 顺序执行任务
```

---

## 🚀 常用命令速查

### 安装与环境

```bash
# 安装 Rust (用于 BPE)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# 创建虚拟环境
uv venv
source .venv/bin/activate

# 安装依赖
uv sync --extra gpu
```

### 训练命令

```bash
# 单 GPU 预训练 (小模型测试)
python -m scripts.base_train \
  --depth=12 \
  --device_batch_size=8 \
  --max_seq_len=1024 \
  --num_iterations=100

# 多 GPU 预训练 (完整训练)
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train \
  --depth=20 \
  --run=my_experiment

# 中间训练 (多任务学习)
torchrun --standalone --nproc_per_node=8 \
  -m scripts.mid_train \
  --device_batch_size=32

# 监督微调
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_sft \
  --num_iterations=500

# 强化学习 (可选)
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_rl \
  --num_iterations=200
```

### 评估命令

```bash
# 评估 CORE 得分 (基础能力)
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_eval

# 评估多个任务
torchrun --standalone --nproc_per_node=8 \
  -m scripts.chat_eval \
  --model_tag=sft \
  --all_tasks

# 只评估特定任务
python -m scripts.chat_eval \
  --model_tag=sft \
  -a GSM8K      # 只评估数学
```

### 交互命令

```bash
# CLI 聊天
python -m scripts.chat_cli \
  --model_tag=sft

# 带前缀的 CLI (自动回复)
python -m scripts.chat_cli \
  --model_tag=sft \
  -p "Hello, my name is"

# Web UI (推荐方式)
python -m scripts.chat_web \
  --model_tag=sft \
  --port=8000

# 访问 UI
open http://localhost:8000
# 或 http://<public_ip>:8000 (远程)
```

### 数据与分词

```bash
# 下载数据分片
python -m nanochat.dataset -n 240

# 训练分词器
python -m scripts.tok_train \
  --max_chars=2000000000 \
  --vocab_size=65536

# 评估分词器
python -m scripts.tok_eval

# 创建自定义数据集
python -c "
from tasks.customjson import CustomJSON
task = CustomJSON('my_data.jsonl')
batch = task.get_batch()
print(batch)
"
```

### Weights & Biases (W&B) 集成

```bash
# 登录 W&B
wandb login

# 带 W&B 日志的训练
WANDB_RUN=my_experiment bash speedrun.sh

# 禁用 W&B (默认)
WANDB_RUN=dummy bash speedrun.sh

# 查看日志
wandb sync
```

---

## 🔍 调试技巧

### 检查 GPU 状态

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 详细 GPU 信息
nvidia-smi -q

# 特定进程的显存使用
nvidia-smi pmon

# GPU 功耗
nvidia-smi --query-gpu=power.draw,power.limit \
  --format=csv,noheader
```

### 性能分析

```python
# 在脚本中添加性能分析
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, 
                profiler.ProfilerActivity.CUDA],
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
) as prof:
    for step in range(100):
        loss = model(batch)
        loss.backward()
        optimizer.step()
    
# 查看 TensorBoard
tensorboard --logdir ./logs
```

### 常见错误及解决

```
错误 1: CUDA Out of Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RuntimeError: CUDA out of memory

解决:
1. 减少 device_batch_size: 32 → 16 → 8 → 4
2. 减少 max_seq_len: 2048 → 1024 → 512
3. 启用梯度检查点: --grad_checkpoint=1
4. 混合精度: --dtype=float16


错误 2: 分布式训练不工作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RuntimeError: No resource available for device

解决:
1. 检查 NVIDIA NCCL: nvidia-smi topo -m
2. 检查 GPU 互连: nvidia-smi -q | grep Link
3. 使用 NCCL_DEBUG: NCCL_DEBUG=INFO torchrun ...


错误 3: 数据加载缓慢
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Training speed 很慢，GPU 利用率低

解决:
1. 增加 num_workers: dataloader(num_workers=4)
2. 预加载数据: python -m nanochat.dataset -n 300
3. 使用本地 SSD: cp data.parquet /ssd/


错误 4: 模型不收敛
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Loss 不下降或在振荡

解决:
1. 减小学习率: matrix_lr=0.01 (from 0.02)
2. 增加 grad_clip: grad_clip=0.5 (from 1.0)
3. 检查数据质量: 手动检查样本
4. 从检查点恢复: 减少学习率 50%
```

---

## 📊 性能基准

### 训练性能

```
硬件          | 模型    | Tokens/sec | GPU Util | 显存用
──────────────────────────────────────────────────────
2×A100-40GB   | d12     | 50K        | 75%      | 35GB
2×A100-80GB   | d20     | 100K       | 80%      | 60GB
8×H100        | d20     | 200K       | 85%      | 40GB
8×H100        | d26     | 120K       | 75%      | 50GB (OOM 可能)
8×H100        | d32     | 80K        | 65%      | 65GB (OOM 可能)
单 A100-40GB  | d12     | 6K         | 50%      | 35GB (梯度累积)
单 H100       | d12     | 15K        | 60%      | 40GB (梯度累积)
```

### 推理性能

```
配置              | 吞吐量        | 延迟（首token）| 显存用
──────────────────────────────────────────────────────
H100, 无 cache    | 100 tok/s     | 10ms           | 5GB
H100, 有 cache    | 400 tok/s     | 2.5ms/token    | 5GB
H100, batch=16    | 3000 tok/s    | 5ms (平均)     | 8GB
2×A100, batch=8   | 800 tok/s     | 10ms (平均)    | 8GB
CPU (8核)         | 0.1 tok/s     | 10000ms        | 15GB (慢)
MPS (M1)          | 1 tok/s       | 1000ms         | 8GB  (慢)
```

### 分词性能

```
操作              | 速度          | 说明
─────────────────────────────────────────────
分词训练 (RustBPE) | 500M chars/s  | 完整 2B 字符只需 4 秒
分词编码 (tiktoken)| 10M tokens/s  | 使用 GPU 加速
分词解码 (tiktoken)| 10M tokens/s  | 非常快
```

---

## 📈 扩展指南

### 添加新评估任务

```python
# 创建 tasks/mytask.py

from tasks.common import Task

class MyTask(Task):
    def __init__(self):
        self.data = []  # 加载你的数据
    
    def get_batch(self):
        """返回 (prompt, completion) 元组列表"""
        return [
            ("Q: What is 2+2?\nA:", " 4"),
            ("Q: What is 3+3?\nA:", " 6"),
        ]
    
    def evaluate(self, completions):
        """评估模型输出
        
        Returns: 准确率 (0-1)
        """
        correct = 0
        for pred, gt in zip(completions, self.get_batch()):
            if pred.strip() == gt[1].strip():
                correct += 1
        return correct / len(completions)

# 在 scripts/chat_eval.py 中注册
from tasks.mytask import MyTask
TASKS = {
    'mytask': MyTask,
    ...
}
```

### 自定义模型架构

```python
# 修改 nanochat/gpt.py 中的 GPT 类

class GPTCustom(GPT):
    def __init__(self, config):
        super().__init__(config)
        
        # 添加你的组件
        self.custom_layer = MyCustomLayer(config.n_embd)
    
    def forward(self, idx, kv_cache=None):
        # 调用父类前向传播
        x, cache = super().forward(idx, kv_cache)
        
        # 应用自定义层
        x = self.custom_layer(x)
        
        return x, cache

# 在训练脚本中使用
model = GPTCustom(config)
```

### 使用不同的优化器

```python
# 在 scripts/base_train.py 中修改优化器选择

if optimizer_choice == "adamw":
    param_groups = [
        {'params': embedding_params, 'lr': embedding_lr, 'weight_decay': weight_decay},
        {'params': matrix_params, 'lr': matrix_lr, 'weight_decay': weight_decay},
    ]
    optimizer = torch.optim.AdamW(param_groups)

elif optimizer_choice == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

elif optimizer_choice == "lion":
    # 需要安装: pip install lion-pytorch
    from lion_pytorch import Lion
    optimizer = Lion(model.parameters(), lr=0.001)
```

### 集成自定义数据

```python
# 方式 1: JSON 行格式
echo '{"prompt": "Hello", "completion": " world"}' > data.jsonl
python -m scripts.base_train --data_path=data.jsonl

# 方式 2: 使用 CustomJSON 任务
from tasks.customjson import CustomJSON
task = CustomJSON('data.jsonl')

# 方式 3: 创建自定义 Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
```

---

## 🎓 学习资源

### 必读论文

1. **Attention is All You Need** (2017)
   - Transformer 架构的原始论文
   - 必读：理解整个模型基础

2. **Rotary Position Embeddings** (2021)
   - RoPE 位置编码
   - 理解 NanoChat 的位置编码

3. **Grouped Query Attention** (2023)
   - GQA 推理优化
   - 理解 KV 缓存节省

4. **Training Compute-Optimal LLMs** (Chinchilla, 2022)
   - 参数和数据的最优缩放
   - 理解为什么 d20 需要 20B tokens

### 项目代码导航

**从这里开始：**
```
README.md                    ← 项目概述
speedrun.sh                  ← 完整训练流程
├→ PROJECT_MAP_CN.md         ← 项目地图（本文档）
└→ ADVANCED_PRINCIPLES_CN.md ← 深度原理
```

**逐个学习核心模块：**
```
1. nanochat/tokenizer.py     ← 分词系统
2. nanochat/gpt.py           ← 模型架构
3. nanochat/engine.py        ← 推理引擎
4. scripts/base_train.py     ← 训练循环
5. nanochat/dataloader.py    ← 数据加载
```

---

## 🐛 常见问题 (FAQ)

**Q: 我只有 1 个 GPU (A100-40GB)，能训练吗？**

A: 可以，但需要大幅降低配置：
```bash
python -m scripts.base_train \
  --depth=10 \
  --device_batch_size=4 \
  --total_batch_size=4096 \
  --num_iterations=1000
```
训练会很慢（~20 倍），但 code paths 是一样的。

**Q: 如何恢复中断的训练？**

A: NanoChat 自动保存检查点：
```bash
# 继续从最新检查点
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train \
  --resume=latest

# 指定特定步骤
torchrun --standalone --nproc_per_node=8 \
  -m scripts.base_train \
  --resume_step=10000
```

**Q: 如何自定义模型的"个性"？**

A: 编辑中间训练数据：
```bash
# 下载并修改身份对话
curl -o identity.jsonl https://...
# 编辑 identity.jsonl 来改变风格

# 混合到训练数据
python -m scripts.mid_train \
  --identity_data=identity.jsonl
```

**Q: 模型生成的质量很差，怎么办？**

A: 尝试这些优化：
1. 增加预训练数据：`python -m nanochat.dataset -n 500`
2. 增加模型大小：`--depth=26`
3. 增加 SFT 数据量和迭代
4. 调整采样参数：`--temperature=0.7 --top_p=0.9`

**Q: 能用 CPU/MPS 运行吗？**

A: 可以，参考 `dev/runcpu.sh`：
```bash
python -m scripts.base_train \
  --device_type=cpu \
  --depth=4 \
  --device_batch_size=1 \
  --max_seq_len=512 \
  --num_iterations=20
```

---

## 📞 获取帮助

- **GitHub Issues**: 项目问题追踪
- **Discussions**: 一般讨论和问题
- **DeepWiki**: 代码问答（deepwiki.com/karpathy/nanochat）

---

**祝你成功训练属于自己的 LLM！** 🚀
