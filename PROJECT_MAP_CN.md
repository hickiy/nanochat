# NanoChat 项目中文学习指南

> 一份完整的 ChatGPT 克隆实现，预算控制在 $100 以内，4小时内完成训练

**项目核心特点**：
- 🎯 从零到一的完整 LLM 全栈实现
- 💰 成本低廉：仅需 ~$100 在 8×H100 GPU 节点上运行
- ⏱️ 高效快速：4 小时完成从分词到推理的全流程
- 🔧 简洁易懂：代码量少（~8300 行），可读性强
- 📊 开箱即用：包含评估、推理、Web UI 等完整功能

---

## 📁 项目结构与文件导航

### 一、核心项目目录

```
nanochat/
├── 🤖 模型架构层
│   ├── gpt.py                    # ⭐ Transformer 架构核心
│   │   ├── GPTConfig             # 模型配置类
│   │   ├── CausalSelfAttention   # 因果注意力层（支持 GQA、旋转位置编码）
│   │   └── GPT                   # 完整模型定义
│   │
│   ├── 优化器与梯度
│   ├── adamw.py                  # 分布式 AdamW 优化器
│   └── muon.py                   # Muon 优化器（用于矩阵参数）
│
├── 📝 分词与文本处理
│   ├── tokenizer.py              # ⭐ BPE 分词器包装层
│   │   ├── HuggingFaceTokenizer  # HF 分词器实现
│   │   ├── RustBPETokenizer      # Rust 高性能 BPE 训练
│   │   └── 特殊令牌定义           # <|bos|>, <|user_start|> 等
│   │
│   └── rustbpe/                  # Rust 实现的 BPE 分词器（性能优化）
│       └── src/lib.rs
│
├── 💾 数据与加载
│   ├── dataset.py                # ⭐ 数据集下载与管理
│   │   ├── 在线数据源处理         # 从互联网下载预训练数据
│   │   └── Parquet 文件解析       # 高效的数据存储格式
│   │
│   └── dataloader.py             # ⭐ 分布式数据加载器
│       ├── tokenizing_distributed_data_loader  # 流式分词
│       ├── DDP 多卡同步           # 分布式数据并行
│       └── 断点恢复支持
│
├── 🔧 推理与生成
│   ├── engine.py                 # ⭐ 高效推理引擎
│   │   ├── KV 缓存管理            # 推理时的键值缓存优化
│   │   ├── 多令牌生成             # 批量生成
│   │   ├── 工具执行               # Python REPL 执行
│   │   └── 温度/采样控制
│   │
│   └── execution.py              # Python 代码执行工具
│
├── 🔌 检查点与恢复
│   └── checkpoint_manager.py      # ⭐ 模型保存/加载
│       ├── 完整模型检查点         # 完整权重保存
│       ├── 分布式训练恢复         # DDP 检查点处理
│       └── 版本管理
│
├── 📊 评估与损失计算
│   ├── loss_eval.py              # 比特每字节 (BPB) 评估
│   ├── core_eval.py              # CORE 任务评估
│   └── 其他评估工具
│
├── ⚙️ 工具与配置
│   ├── common.py                 # ⭐ 通用工具函数
│   │   ├── 分布式信息获取         # DDP 初始化
│   │   ├── 设备自动检测           # CUDA/MPS/CPU 检测
│   │   ├── 日志与输出             # 彩色日志记录
│   │   └── 文件下载与锁           # 多进程安全下载
│   │
│   ├── configurator.py           # ⭐ 配置系统
│   │   └── 比 argparse 更优的配置解析
│   │
│   └── report.py                 # 生成运行报告
│
└── 🌐 前端与交互
    └── ui.html                   # ChatGPT 风格的 Web UI


### 二、脚本与训练流程

scripts/
├── 🔤 分词训练
│   ├── tok_train.py              # BPE 分词器训练
│   └── tok_eval.py               # 分词评估（压缩率等）
│
├── 📚 预训练 (Base Model)
│   ├── base_train.py             # ⭐ 基础模型预训练
│   │   └── 自回归语言建模
│   │
│   ├── base_loss.py              # 预训练损失评估与采样
│   └── base_eval.py              # CORE 分数评估
│
├── 💬 中间训练 (Midtraining)
│   └── mid_train.py              # ⭐ 中间训练
│       ├── 多任务混合学习
│       ├── 对话特殊令牌学习
│       └── 工具使用学习
│
├── 🎯 监督微调 (SFT)
│   └── chat_sft.py               # ⭐ 监督微调
│       └── 对话质量改进
│
├── 🎲 强化学习 (RL)
│   └── chat_rl.py                # 强化学习微调（可选）
│
├── 🤖 推理与对话
│   ├── chat_cli.py               # 命令行聊天
│   ├── chat_web.py               # 🌐 Web 聊天界面
│   ├── chat_eval.py              # 各任务评估
│   └── 推理优化
│
└── 📋 任务评估

tasks/
├── common.py                     # ⭐ 任务框架
│   ├── TaskMixture              # 任务混合
│   ├── TaskSequence             # 任务序列
│   └── Task 基类
│
├── 多项选择题任务
│   ├── arc.py                    # ARC 科学竞赛题
│   ├── mmlu.py                   # 多领域选择题
│   └── spellingbee.py            # 拼写蜜蜂任务
│
├── 计算任务
│   ├── gsm8k.py                  # 小学数学题
│   └── humaneval.py              # Python 编码任务
│
├── 对话任务
│   ├── smoltalk.py               # 综合对话数据集
│   └── customjson.py             # 自定义 JSON 格式
│
└── 🔧 任务工具


### 三、配置与运行

speedrun.sh                        # ⭐ 快速运行脚本（主要入口）
│   ├── 第 1 步：分词器训练
│   ├── 第 2 步：数据下载
│   ├── 第 3 步：预训练
│   ├── 第 4 步：中间训练
│   ├── 第 5 步：监督微调
│   └── 第 6 步：Web UI 部署

run1000.sh                         # 更大的模型脚本（~$800）

dev/runcpu.sh                      # CPU/MPS 小规模运行示例
```

---

## 🔄 工作流程与数据流

### 完整训练流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    NanoChat 完整训练流程                          │
└─────────────────────────────────────────────────────────────────┘

1️⃣ 分词器阶段 (Tokenization)
   ├─ 下载原始文本数据 (~2B 字符)
   ├─ 训练 BPE 分词器 (vocab_size=65536)
   └─ 评估分词器性能 (压缩率)
       └─ 输出：tokenizer.model (分词词表)

2️⃣ 数据准备阶段 (Data Preparation)
   ├─ 下载预训练数据 (~240 个数据分片)
   ├─ 转换为 Parquet 格式
   └─ 流式分词处理
       └─ 输出：tokenized 数据集

3️⃣ 预训练阶段 (Base Training) ⭐ 核心
   ├─ 初始化 GPT 模型 (d20 = 561M 参数)
   ├─ 自回归语言建模
   │  └─ 目标：预测下一个词
   ├─ Chinchilla 定律：20×(参数数) = 训练令牌数
   └─ 输出：base.pt (预训练模型)

4️⃣ 中间训练阶段 (Midtraining)
   ├─ 加载预训练模型
   ├─ 多任务学习：
   │  ├─ 对话格式学习 (SmolTalk, 自定义数据)
   │  ├─ 选择题学习 (ARC, MMLU)
   │  ├─ 数学题学习 (GSM8K)
   │  └─ 编码学习 (HumanEval)
   └─ 输出：mid.pt (中间训练模型)

5️⃣ 监督微调阶段 (SFT)
   ├─ 加载中间训练模型
   ├─ 对话质量微调
   └─ 输出：sft.pt (微调模型)

6️⃣ 强化学习阶段 (RL) 🔄 可选
   ├─ 策略优化
   ├─ GSM8K 奖励模型
   └─ 输出：rl.pt (RL 模型)

7️⃣ 推理与部署阶段 (Inference)
   ├─ 启动 Web UI 服务
   ├─ KV 缓存优化推理
   └─ 用户交互聊天
```

### 数据流动

```
raw_text.jsonl
    ↓ [tokenizer] 
tokenized_ids (token sequences)
    ↓ [batch & shuffle]
mini_batches (B×T×vocab_size)
    ↓ [model forward]
logits (predictions)
    ↓ [cross entropy loss]
loss → [backward] → gradients
    ↓ [optimizer step: AdamW/Muon]
updated_weights
```

---

## 🧠 关键原理详解

### 1. 分词系统 (Tokenization)

**为什么需要分词？**
- LLM 只理解整数序列，不理解原始文本
- 需要将文本转换为令牌 ID 序列
- 分词效率影响训练效率和模型性能

**BPE (Byte Pair Encoding) 原理**

```
初始状态：每个字符是一个令牌
'hello world' → [h, e, l, l, o, space, w, o, r, l, d]

第 1 次迭代：找最常见的相邻字符对 'l','l'
'hello world' → [h, e, ll, o, space, w, o, r, l, d]

第 2 次迭代：找最常见的相邻字符对 'e','ll'  
'hello world' → [h, ell, o, space, w, o, r, l, d]

...重复 vocab_size 次...

最终：得到 65536 个令牌的词表
```

**NanoChat 的实现**
- **分词器训练** (`scripts/tok_train.py`)：在 ~2B 字符上训练
- **分词器推理** (`nanochat/tokenizer.py`)：使用 tiktoken（快速）
- **RustBPE**：用 Rust 实现训练部分以提高速度
- **特殊令牌**：`<|bos|>`, `<|user_start|>` 等用于结构化数据

---

### 2. 数据加载系统 (Data Loading)

**核心概念：流式分词数据加载器**

```
问题：如果先下载所有数据再分词，需要存储 ~24GB 文本
解决：边下载边分词边训练（流式处理）

流程：
├─ ParquetDataset (Parquet 文件)
│  ├─ Row Groups (分组，可并行读取)
│  └─ Text 列 (原始文本)
│
├─ DDP 分布式:
│  ├─ 8 个 GPU，每个处理不同数据分片
│  ├─ GPU_0 处理 shard 0
│  ├─ GPU_1 处理 shard 1
│  └─ ...
│
├─ 流式分词：
│  ├─ 读取文本 → 分词 → 拼接令牌
│  └─ 当累积足够令牌时，切割成 batch
│
└─ 训练 Batch:
   ├─ shape: [B, T] = [32, 2048]
   ├─ 32 个序列，每个 2048 令牌
   └─ 每 mini-batch 的令牌数 = 32 × 2048 = 65536
```

**断点恢复机制**
```python
# 保存状态
state = {
    'pq_idx': 25,           # 第 25 个 Parquet 文件
    'rg_idx': 512,          # 第 512 个 Row Group
}

# 恢复后，从这个位置继续读取数据
```

---

### 3. 模型架构 (GPT Model)

**Transformer 核心结构**

```
Input: token_ids [B, T]
  ↓
Token Embedding: [B, T, d_model]
  ↓
Position Encoding: Rotary Embeddings (RoPE)
  ↓
┌─── Transformer Block (×n_layer) ─────┐
│                                        │
│ 1. Causal Self-Attention              │
│    ├─ Query, Key, Value 投影          │
│    ├─ QK 正规化                        │
│    ├─ Rotary Position Embedding       │
│    ├─ 缩放点积注意力                   │
│    ├─ KV 缓存 (推理时优化)            │
│    └─ 组查询注意力 GQA                 │
│                                        │
│ 2. 前馈网络 (FFN)                    │
│    ├─ Linear(d → 4d)                  │
│    ├─ ReLU²激活                       │
│    └─ Linear(4d → d)                  │
│                                        │
│ 3. RMSNorm 残差连接                  │
│    └─ Layer Norm (无学习参数)        │
│                                        │
└────────────────────────────────────────┘
  ↓ (×n_layer)
RMSNorm
  ↓
Output Projection (LM Head): [B, T, vocab_size]
  ↓
Logits: [B, T, vocab_size]
```

**关键创新**

| 特性 | 作用 | 为什么 |
|-----|-----|-------|
| 旋转位置编码 (RoPE) | 相对位置信息 | 比绝对位置编码更优雅 |
| QK 正规化 | 稳定注意力 | 防止注意力权重过于尖锐 |
| 组查询注意力 (GQA) | 推理时节省显存 | KV 头数 < Q 头数 |
| RMSNorm | 更快的归一化 | 只有 RMS，无学习参数 |
| 无偏差线性层 | 减少参数 | 降低内存占用 |
| ReLU² 激活 | 更好的稳定性 | 比 GELU 更简单且稳定 |

---

### 4. 训练优化 (Training Optimization)

**双优化器策略**

```
参数分类：
├─ Embedding 参数 (token_embedding, lm_head)
│  └─ 使用 AdamW 优化 (自适应学习率)
│     └─ 学习率: embedding_lr = 0.2
│
└─ 矩阵参数 (线性层权重)
   └─ 使用 Muon 优化 (Orthogonal SGD)
      └─ 学习率: matrix_lr = 0.02

为什么分开？
- Embedding 层权重更新规律性差，需要 Adam 的自适应能力
- 矩阵参数权重更新可以正交化，Muon 更高效
```

**批大小计算**

```
Chinchilla 定律（LLM 扩展规律）：
  训练令牌数 = 20 × 模型参数数

d20 模型 (561M 参数):
  需要 561M × 20 = 11.2B 训练令牌
  
实际批大小计算：
  Total Batch Size = 524288 (总令牌数/步)
  Device Batch Size = 32 (每 GPU 的序列数)
  Seq Len = 2048 (序列长度)
  
  每步令牌数 = 8_GPUs × 32 × 2048 = 524288 ✓
  总步数 = 11.2B / 524288 ≈ 21,381 步
  
  总训练时间 ≈ 21,381 步 × (平均每步时间) ≈ 4 小时
```

**梯度累积 (Gradient Accumulation)**

```
问题：需要 Total Batch Size = 524288 令牌/步
     但显存只能容纳 Device Batch Size = 32 × 2048 = 65536 令牌

解决：梯度累积
  
  步长 1: forward(32 × 2048) → loss → backward → grad_accum_1
  步长 2: forward(32 × 2048) → loss → backward → grad_accum_2
  ...
  步长 8: forward(32 × 2048) → loss → backward → grad_accum_8
  
  更新: optimizer.step(sum(grad_accum_1...8))
  
  总令牌数 = 8 × (32 × 2048) = 524288 ✓
```

---

### 5. 推理引擎 (Inference Engine)

**KV 缓存机制**

```
训练时（没有 KV 缓存）：
  Step 1: 计算全部 seq 的注意力
    Input: [B, 1, d]
    Q = [B, 1, H, d_head]
    K = [B, 1, H, d_head]  
    V = [B, 1, H, d_head]
    Output: [B, 1, d]
  
  Step 2: 计算全部 seq 的注意力
    Input: [B, 2, d]
    计算新的 Q, K, V（包括 step1）
    Output: [B, 2, d]
  
  ❌ 问题：重复计算 step1 的 K, V

推理时（有 KV 缓存）：
  Step 1: 计算全部 seq 的注意力
    Q = [B, 1, H, d_head]
    K = [B, 1, H, d_head]  ← 缓存
    V = [B, 1, H, d_head]  ← 缓存
    Output: [B, 1, d]
  
  Step 2: 只计算新令牌
    Q_new = [B, 1, H, d_head]
    K = concat(K_cache, K_new)  ← 使用缓存 + 新 K
    V = concat(V_cache, V_new)  ← 使用缓存 + 新 V
    Output: [B, 1, d]
  
  ✓ 优势：计算量从 O(seq_len²) 降到 O(seq_len)
         推理速度快 ~100 倍
```

**批量推理**

```
支持一次生成多个令牌：

输入: [token_ids: [1, 2, 3]]

第 1 次前向：
  Input shape: [B=1, T=3, d]
  Output shape: [B=1, 3, vocab_size]
  返回: logits[:, -1, :] → 采样下一个令牌

如果 cache 为空，T=seq_len
如果 cache 有值，T=新增令牌数（推理加速）
```

---

### 6. 监督微调 (SFT) 与中间训练 (Midtraining)

**目的与区别**

| 阶段 | 目的 | 数据 | 方法 |
|-----|-----|------|------|
| **Midtraining** | 学习对话格式、多任务能力 | 多个任务数据混合 | 继续预训练 |
| **SFT** | 优化对话质量 | 高质量对话数据 | 标准微调 |

**任务混合系统**

```python
# 定义任务序列
tasks = TaskMixture([
    SmolTalk(),              # 一般对话
    GSM8K(),                 # 数学题
    ARC(),                   # 科学题
    MMLU(),                  # 多领域知识
    HumanEval(),             # 编程任务
    SpellingBee(),           # 拼写任务
])

# 每次迭代随机选择任务
for step in range(num_steps):
    task = tasks.sample()     # 随机选一个任务
    batch = task.get_batch()  # 从该任务获取数据
    # 继续训练...
```

**特殊令牌用途**

```
对话格式示例：
<|bos|>
<|user_start|>你好，今天天气怎么样？<|user_end|>
<|assistant_start|>今天晴空万里，天气很好！<|assistant_end|>
<|user_start|>谢谢！<|user_end|>
<|assistant_start|>不客气，有什么需要帮助的吗？<|assistant_end|>
<|python_start|>print("Hello")<|python_end|>
<|output_start|>Hello<|output_end|>

模型学会：
- 识别用户和助手的消息边界
- 知道何时执行代码
- 知道何时返回代码输出
```

---

### 7. 评估指标 (Evaluation Metrics)

**关键指标**

| 指标 | 含义 | 范围 | 更好 |
|-----|-----|------|------|
| **BPB** (Bits Per Byte) | 平均每字节需要多少比特来编码 | 0-8 | 更低 ↓ |
| **Loss** | 交叉熵损失 | 0-∞ | 更低 ↓ |
| **CORE** | 基础能力得分（DCLM 论文） | 0-1 | 更高 ↑ |
| **ARC** | 科学竞赛多项选择题准确率 | 0-1 | 更高 ↑ |
| **MMLU** | 多领域知识准确率 | 0-1 | 更高 ↑ |
| **GSM8K** | 小学数学题准确率 | 0-1 | 更高 ↑ |
| **HumanEval** | Python 编码任务通过率 | 0-1 | 更高 ↑ |

**评估流程**

```
预训练后:
  ├─ base_eval.py  → CORE 得分 (基础能力)
  └─ base_loss.py  → BPB (压缩能力)

中间训练后:
  └─ chat_eval.py  → ARC, MMLU, GSM8K, HumanEval

微调后:
  ├─ 重新评估所有任务
  └─ 生成完整报告 report.md
```

---

### 8. 分布式训练 (DDP)

**多 GPU 并行策略**

```
Master GPU (RANK=0)     GPU_1          GPU_2 ... GPU_7
   │                    │              │         │
   ├─ 读取数据分片 0     │              │         │
   │  [256M 字符]        │              │         │
   │                     ├─ 读取数据     ├─ 读取   │
   │                     │  分片 1       │ 分片 2  │
   │                     │              │         │
   ├─ 分词 → 令牌序列    ├─ 分词        ├─ 分词   │
   │                     │              │         │
   ├─ 模型副本 (同步)    ├─ 模型副本    ├─ 模型   │
   │                     │              │         │
   ├─ Forward Pass       ├─ Forward     ├─ For   │
   │  [32×2048]          │  [32×2048]   │ [32×2048]
   │                     │              │         │
   ├─ Backward & Grad    ├─ Backward    ├─ Back  │
   │                     │              │         │
   ├─ AllReduce (平均梯度) ├─ 通信        ├─ 通信  │
   │  (NCCL 通信)        │              │         │
   │                     │              │         │
   ├─ Optimizer Step     ├─ 同步更新    ├─ 同步  │
   │  (所有 GPU 同一点)   │              │         │
   │                     │              │         │
   └─ 模型参数完全同步   └─ 同步完成    └─ 同步  │
```

**通信开销**

```
AllReduce (对所有梯度求和然后平均)

简单方案：
  Master 接收所有 gradient
  求和
  广播回去
  
  通信量: n_gpu × (2 × gradient_size)  (send + receive)

NanoChat 使用：NCCL 后端
  - 优化的链式 AllReduce
  - GPU 直连通信 (NVLink)
  - 通信量: 2 × gradient_size × log(n_gpu)
```

---

## 📊 关键数学公式

### 模型缩放规律 (Chinchilla)

$$N = \text{参数数量}$$
$$D = \text{数据令牌数}$$

**Chinchilla 定律**：
$$D = 20 \times N$$

对于 d20 模型 (561M 参数)：
$$D = 20 \times 561M = 11.2B \text{ tokens}$$

### 交叉熵损失

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i | x_i)$$

其中：
- $P(y_i | x_i)$ 是模型预测正确令牌的概率
- $N$ 是令牌总数

### 比特每字节 (BPB)

$$\text{BPB} = \frac{\text{Total Loss (bits)}}{\text{Total Characters}}$$

$$\text{Total Loss (bits)} = \text{Loss} \times \log_2(e) \times \text{num_tokens}$$

其中 $\log_2(e) \approx 1.443$ 是从自然对数转换到对数 2

---

## 🚀 快速开始步骤

### 1. 环境准备

```bash
# 进入项目目录
cd e:\github\nanochat

# 安装 uv (包管理)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
uv venv

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 安装依赖
uv sync --extra gpu
```

### 2. 分词器阶段

```bash
# 安装 Rust (用于构建 RustBPE)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# 构建 RustBPE
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 下载数据 + 训练分词器
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
```

### 3. 预训练阶段

```bash
# 单 GPU
python -m scripts.base_train --depth=12 --device_batch_size=8

# 多 GPU (8 张 H100)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train --depth=20
```

### 4. 聊天交互

```bash
# CLI 聊天
python -m scripts.chat_cli -p "你好"

# Web UI (推荐)
python -m scripts.chat_web
# 访问 http://localhost:8000
```

---

## 💡 学习路径建议

### 初级（理解概念）
1. 阅读 `README.md` 了解项目整体目标
2. 研究 `speedrun.sh` 理解训练流程
3. 查看 `nanochat/gpt.py` 学习模型架构

### 中级（理解实现）
1. 深入 `nanochat/tokenizer.py` 理解分词
2. 学习 `nanochat/dataloader.py` 的数据流
3. 研究 `scripts/base_train.py` 的训练循环
4. 理解 `nanochat/engine.py` 的推理优化

### 高级（修改与扩展）
1. 修改 `nanochat/gpt.py` 中的模型架构
2. 在 `tasks/` 中添加新的评估任务
3. 自定义数据集 (见 `tasks/customjson.py`)
4. 实现新的优化器或损失函数

---

## 🔗 重要文件速查

| 功能 | 文件 | 关键函数/类 |
|-----|------|------------|
| 分词训练 | `scripts/tok_train.py` | train_bpe |
| 分词推理 | `nanochat/tokenizer.py` | get_tokenizer() |
| 模型定义 | `nanochat/gpt.py` | GPT 类 |
| 注意力机制 | `nanochat/gpt.py` | CausalSelfAttention |
| 数据加载 | `nanochat/dataloader.py` | tokenizing_distributed_data_loader |
| 预训练 | `scripts/base_train.py` | main() |
| 推理 | `nanochat/engine.py` | Engine 类 |
| Web UI | `scripts/chat_web.py` | app (FastAPI) |
| 评估 | `scripts/chat_eval.py` | evaluate 函数 |

---

## 📚 相关论文与参考

1. **Attention is All You Need** (Vaswani et al., 2017)
   - Transformer 架构基础

2. **RoFormer: Enhanced Transformer with Rotary Position Embedding** (Su et al., 2021)
   - 旋转位置编码 (RoPE)

3. **Training Compute-Optimal Large Language Models** (Hoffmann et al., 2022)
   - Chinchilla 缩放定律

4. **Grouped Query Attention** (Ainslie et al., 2023)
   - GQA 推理优化

5. **DCLM: The Data-Centric Language Modeling Project** 
   - CORE 评估指标

6. **Root Mean Square Layer Normalization** (Zhang & Sennrich, 2019)
   - RMSNorm 实现

---

## ❓ 常见问题

**Q: 为什么用 Rust 写 BPE？**
A: Python BPE 训练很慢。Rust 版本快 10-50 倍，训练 65K 词表只需几分钟。

**Q: KV 缓存为什么能加速推理？**
A: 避免重复计算历史令牌的 K、V。从 O(T²) 复杂度降到 O(T)。

**Q: 为什么用双优化器？**
A: Embedding 层权重更新规律差需要 Adam；矩阵参数可以正交化用 Muon 更高效。

**Q: 梯度累积的作用？**
A: 逻辑上累积 8 步梯度后更新，物理上分 8 次做，节省显存 8 倍。

**Q: 为什么 DDP 需要 AllReduce？**
A: 确保所有 GPU 的梯度相同，然后统一更新，保证收敛性。

**Q: 中间训练有什么特别的？**
A: 混合多个下游任务数据继续预训练，学习任务特定格式和能力。

**Q: 能在 CPU 上运行吗？**
A: 可以，但很慢。参考 `dev/runcpu.sh` 配置小参数模型。

---

## 📞 调试与优化建议

### 显存不足 (OOM)

```bash
# 减少 device_batch_size
python -m scripts.base_train --device_batch_size=8  # 从 32 改为 8

# 减少序列长度
python -m scripts.base_train --max_seq_len=1024  # 从 2048 改为 1024

# 梯度累积会自动调整
# 总 batch size 保持不变，分更多步累积
```

### 训练速度慢

```bash
# 检查 GPU 利用率
nvidia-smi  # 应该 >90%

# 检查数据加载是否瓶颈
# 在 dataloader 中加 debug 打印

# 用 torch.profiler 分析
torch.profiler.profile()
```

### 模型效果差

```bash
# 增加训练数据
python -m nanochat.dataset -n 300  # 增加数据分片数

# 增加模型大小
python -m scripts.base_train --depth=24  # 增加层数

# 调整学习率
python -m scripts.base_train --matrix_lr=0.03
```

---

## 🎓 总结

NanoChat 项目展示了如何在有限的计算预算下实现一个完整、高效的 LLM 系统。核心设计原则包括：

1. **简洁性**：代码量少但完整，易于理解和修改
2. **效率**：优化的推理、分布式训练、梯度累积等
3. **可扩展性**：模块化设计，易于添加新任务或修改架构
4. **可复现性**：清晰的配置系统和检查点管理

通过学习这个项目，你可以理解：
- ✅ 现代 LLM 的完整工作流程
- ✅ Transformer 架构的高效实现
- ✅ 分布式训练的最佳实践
- ✅ 推理优化（KV 缓存、GQA 等）
- ✅ 从预训练到微调的完整过程

祝你学习愉快！🚀
