# 分词器（Tokenizer）完全指南

## 第一部分：什么是分词器？

### 核心概念

分词器是 NLP（自然语言处理）中的关键组件，它的作用是将人类可读的文本转换为机器可以理解的数字序列（**Token IDs**）。

```
文本输入: "Hello, world!"
         ↓↓↓ [分词器处理] ↓↓↓
Token IDs: [7, 15, 23, 89, 45, ...]
```

### 为什么需要分词器？

神经网络（如 LLM）只能处理数字，不能直接理解文字。分词器充当了**文本与数字之间的桥梁**：

- **编码（Encoding）**：文本 → Token IDs
- **解码（Decoding）**：Token IDs → 文本

### 分词粒度的对比

```
原始文本: "Hello"

字符级分词:     H | e | l | l | o
                (5 个 token)

单词级分词:     Hello
                (1 个 token)

字节对编码(BPE): Hel | lo
                (2 个 token)  ← GPT 使用的方案
```

## 第二部分：字节对编码（BPE - Byte Pair Encoding）

### BPE 是什么？

BPE 是一种智能的分词算法，它**自动学习**如何将文本分割成最优的 token 序列。

**核心原理**：
1. 从最小的单位开始（单个字节，共 256 个）
2. 逐步合并高频出现的字节对
3. 重复这个过程直到达到目标词表大小

### BPE 训练过程示例

假设我们要构建一个词表大小为 258 的分词器（256 个字节 + 2 个合并）：

**初始状态**（词表大小 = 256）：
```
所有文本都被分解为单个字节：
"hello" → [h, e, l, l, o]  (5 个 token)
"world" → [w, o, r, l, d]  (5 个 token)
```

**第 1 次合并**（词表大小 = 257）：
```
统计所有字节对的出现频率：
- (l, l): 出现 2 次  ← 最高频
- (h, e): 出现 1 次
- (e, l): 出现 1 次
- ...

合并 (l, l) → 新 token ID 256
"hello" → [h, e, l, 256, o]  (4 个 token)
"hello" 中的两个 l 合并了
```

**第 2 次合并**（词表大小 = 258）：
```
重新统计字节对频率（考虑新的 token 256）：
- (e, l): 出现 1 次
- (e, 256): 出现 1 次
- 其他字节对
- ...

合并出现最高频的字节对 → 新 token ID 257
```

**最终结果**：
```
训练完成后，模型获得一个"合并规则表"（merges table）：
- (l, l) → 256
- (最高频字节对) → 257
- ...

推理时，使用这个规则表按顺序应用合并，将文本转换为 token IDs。
```

## 第三部分：rustbpe 的作用

### rustbpe 为什么存在？

在 nanochat 项目中，有多种分词器实现方案：

| 方案 | 优点 | 缺点 | 用途 |
|------|------|------|------|
| **tiktoken** | 快速、官方实现 | 仅推理，无训练功能 | LLM 推理（已训练的分词器） |
| **HuggingFace tokenizers** | 功能完整 | 复杂、臃肿、难以维护 | 通用 NLP 任务 |
| **minbpe** | 简洁、清楚 | Python 实现，训练慢 | 学习用 |
| **rustbpe** ⭐ | 高效、简洁、专用 | 仅支持 BPE | **nanochat 的最佳选择** |

### rustbpe 的核心优势

✅ **高效**：使用 Rust 编写，比 Python 快 10-100 倍
✅ **简洁**：代码简单易懂，专注于 BPE 算法
✅ **专用**：为 GPT 风格的分词器优化
✅ **无缝集成**：生成的词表可直接用于 tiktoken 推理

### rustbpe 的工作流

```
原始训练数据
     ↓
rustbpe.Tokenizer (Rust 实现)
  - 使用正则表达式分割文本
  - 执行 BPE 训练算法
  - 生成"合并规则表"
     ↓
词表 (Mergeable Ranks)
     ↓
tiktoken.Encoding (高效推理)
     ↓
LLM 推理
```

## 第四部分：rustbpe 的训练过程详解

### 算法步骤

#### 1️⃣ 文本预处理

使用 GPT-4 风格的正则表达式将文本分割成"块"（chunks）：

```rust
// GPT-4 分割模式
r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

// 例子
"Hello, world!" 
  ↓ [分割]
["Hello", ",", " ", "world", "!"]
```

**目的**：
- 智能分割文本（避免在标点符号、数字中间分割）
- 减少初始 token 数量
- 让 BPE 有更好的合并机会

#### 2️⃣ 字节转换

将每个块转换为字节序列：

```
块: "Hello"
  ↓ [转换为字节]
字节: [72, 101, 108, 108, 111]  (十进制)
      [H,  e,   l,   l,   o]    (字符)

块: ","
  ↓ [转换为字节]
字节: [44]  (逗号的 ASCII 码)
```

#### 3️⃣ 计算字节对频率

```python
# 例如有 1000 个 "Hello" 文本
块: "Hello" → [H, e, l, l, o]

字节对统计（在这 1000 个文本中）：
- (H, e):   1000 次  ← 最高频！
- (e, l):   1000 次
- (l, l):   1000 次
- (l, o):   1000 次

全局统计（合并所有文本的结果）：
- (H, e):   10000 次
- (e, l):   8000 次
- ...
```

#### 4️⃣ 贪心合并

使用最大堆（优先级队列）实现高效的贪心合并：

```
Iteration 1:
最高频字节对: (H, e) 出现 10000 次
新 token ID: 256
合并规则: (H, e) → 256

在所有文本中应用合并：
"Hello" → [256, l, l, o]  (4 个 token，之前是 5 个)

Iteration 2:
重新计算频率（考虑新 token 256）
最高频字节对: (l, l) 出现 9500 次
新 token ID: 257
合并规则: (l, l) → 257

继续这个过程...
```

#### 5️⃣ 优化技巧

rustbpe 使用了多个优化技巧来加速训练：

```rust
// 1. 并行计数 (Rayon)
// 多线程并行计算字节对频率
words
    .par_iter()  // 并行迭代每个块
    .map(|w| count_pairs(w))  // 并行计算该块的字节对
    .reduce(...)  // 合并结果

// 2. 懒刷新堆 (Lazy Heap Refresh)
// 不是每次都更新堆中的所有元素
// 只在弹出元素时检查其频率是否过期
if top.count != current_count {
    top.count = current_count;
    heap.push(top);  // 重新推入
    continue;
}

// 3. 高效的字对合并
// 在原地修改向量，避免不必要的复制
fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
    // 原地合并，返回变化的字节对
    // 这样只需要更新受影响的字节对
}

// 4. 紧凑的数据结构
// 使用 AHashMap (快速哈希表) 和 CompactString (节省内存)
```

#### 6️⃣ 参数说明

```python
tokenizer = rustbpe.Tokenizer()
tokenizer.train_from_iterator(
    text_iterator,           # 文本数据源（Python 迭代器）
    vocab_size=10000,        # 最终词表大小
    buffer_size=8192,        # 缓冲区大小（多少文本后开始处理）
    pattern=SPLIT_PATTERN    # 文本分割正则表达式
)
```

## 第五部分：推理过程（Encoding）

训练完成后，我们有了一个"合并规则表"。现在可以使用它来对新文本进行编码：

### 编码步骤

```python
text = "Hello, world!"

# 步骤 1: 正则分割
chunks = ["Hello", ",", " ", "world", "!"]

# 步骤 2: 转换为初始字节
ids = [
    [72, 101, 108, 108, 111],  # "Hello"
    [44],                        # ","
    [32],                        # " "
    [119, 111, 114, 108, 100],  # "world"
    [33]                         # "!"
]

# 步骤 3: 应用合并规则（逐个块）
# 假设合并规则有：(72, 101) → 256, (108, 108) → 257

# 块 1: [72, 101, 108, 108, 111]
for merge_rule in sorted_merges:
    pair, new_id = merge_rule
    # 找到最早出现的 pair，替换为 new_id
    while pair in ids:
        idx = find_first_occurrence(pair)
        if idx != -1:
            ids[idx:idx+2] = [new_id]
        else:
            break

# 块 1 结果: [256, 257, 111]  # 3 个 token

# 步骤 4: 合并所有块的结果
final_ids = [256, 257, 111, 44, 32, 119, 111, 114, 257, 100, 33]
# 总共 11 个 token

# 步骤 5: 添加特殊 token（可选）
final_ids = [<bos_token_id>] + final_ids  # 在开头添加文档起始 token
```

### tiktoken 推理

```python
# nanochat 使用 tiktoken 进行快速推理
from nanochat.tokenizer import RustBPETokenizer

tokenizer = RustBPETokenizer.from_directory("tokenizer/")

# 编码
text = "Hello, world!"
token_ids = tokenizer.encode(text)
# → [1, 256, 257, 111, 44, 32, 119, 111, 114, 257, 100, 33]

# 解码
text_back = tokenizer.decode(token_ids)
# → "Hello, world!"
```

## 第六部分：nanochat 中的实际应用

### 特殊 Token

nanochat 定义了 8 个特殊 token 用于对话格式化：

```python
SPECIAL_TOKENS = [
    "<|bos|>",              # 文档起始（Beginning of Sequence）
    "<|user_start|>",       # 用户消息开始
    "<|user_end|>",         # 用户消息结束
    "<|assistant_start|>",  # AI 助手消息开始
    "<|assistant_end|>",    # AI 助手消息结束
    "<|python_start|>",     # Python 工具调用开始
    "<|python_end|>",       # Python 工具调用结束
    "<|output_start|>",     # 输出开始
    "<|output_end|>",       # 输出结束
]
```

### 对话渲染示例

```python
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}

# 渲染为 token 序列
ids, mask = tokenizer.render_conversation(conversation)

# ids:   [BOS, USER_START, ..., USER_END, ASST_START, ..., ASST_END]
# mask:  [0,   0,           0,   0,       1,          1,  1]
#        (mask=0: 输入，mask=1: 需要训练的输出)
```

### 完整流程

```
原始训练数据（文本文件）
     ↓
[tok_train.py] 使用 rustbpe 训练分词器
     ↓
生成的文件：
- tokenizer.pkl (tiktoken 对象)
- token_bytes.pt (token 字节嵌入)
     ↓
[训练/推理] 使用 tokenizer.encode() / decode()
     ↓
模型输入/输出
```

## 第七部分：性能对比

| 操作 | minbpe (Python) | HuggingFace | rustbpe | 相对 rustbpe |
|------|-----------------|-------------|---------|-------------|
| 训练 50GB 文本 | ~2 小时 | ~1 小时 | ~2 分钟 | **60-300x** ⚡ |
| 编码（推理） | ~100k token/s | ~300k token/s | tiktoken: **1M+ token/s** | **3-10x** ⚡ |
| 代码行数 | ~200 行 | ~5000+ 行 | ~300 行 | 简洁 ✓ |
| 学习难度 | 简单 | 困难 | 中等 | 平衡 ✓ |

## 总结

### rustbpe 的核心价值

1. **高效**：Rust 实现 + 并行化 = 训练速度提升 100 倍
2. **简洁**：专注于 BPE，代码清晰易维护
3. **无缝集成**：直接与 tiktoken 配合，推理速度保持一流
4. **专用优化**：为 GPT 风格分词器量身定制

### 工作流总结

```
┌─────────────────┐
│  原始文本数据   │
└────────┬────────┘
         │
         ↓
┌─────────────────────────────────┐
│   rustbpe.train_from_iterator() │
│  （Rust 并行 BPE 训练）         │
└────────┬────────────────────────┘
         │
         ↓
┌──────────────────────────────┐
│  合并规则表 + tiktoken 编码  │
│ (Mergeable Ranks)            │
└────────┬─────────────────────┘
         │
         ↓
    ┌────┴────┐
    │          │
    ↓          ↓
┌─────────┐ ┌─────────┐
│ 编码    │ │ 解码    │
│(推理)   │ │(解码)   │
└─────────┘ └─────────┘
```

### 关键要点

- **分词器** = 文本 ↔ 数字的桥梁
- **BPE** = 智能的、可学习的分词方案
- **rustbpe** = 高效、简洁、专用的 BPE 实现
- **nanochat** = 使用 rustbpe 训练 + tiktoken 推理的完整流程
