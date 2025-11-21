# 📚 NanoChat 中文学习文档索引

欢迎使用 NanoChat 中文学习资源！这个文件帮助你快速找到需要的文档。

---

## 🎯 快速导航

### 初学者路线图 👶

如果你是 LLM 和 NanoChat 的新手，请按此顺序阅读：

1. **[PROJECT_MAP_CN.md](PROJECT_MAP_CN.md)** - 项目地图与入门指南
   - ✅ 项目结构概览
   - ✅ 完整工作流程
   - ✅ 关键概念解释
   - ✅ 学习路径建议

2. **[QUICK_REFERENCE_CN.md](QUICK_REFERENCE_CN.md)** - 快速参考手册
   - ✅ 常用命令
   - ✅ 配置参数速查
   - ✅ 调试技巧
   - ✅ 常见问题 FAQ

3. **README.md** - 官方英文版本
   - 了解项目背景和更新

### 进阶学习路线 🧠

如果你想深入理解 NanoChat 的实现细节：

1. **[ADVANCED_PRINCIPLES_CN.md](ADVANCED_PRINCIPLES_CN.md)** - 高级原理深度解析
   - ✅ Transformer 架构详解
   - ✅ 注意力机制优化
   - ✅ 位置编码数学原理
   - ✅ 分布式训练细节
   - ✅ 推理优化策略
   - ✅ 性能瓶颈分析

2. **代码仓库** - 直接阅读实现
   - `nanochat/gpt.py` - 模型架构
   - `nanochat/engine.py` - 推理引擎
   - `scripts/base_train.py` - 训练循环

---

## 📖 文档详细介绍

### 📄 PROJECT_MAP_CN.md

**适合人群**: 所有人（入门必读）

**主要内容**:
- 项目整体结构和文件说明
- 每个模块的功能和职责
- 完整的训练和推理流程图
- 关键概念的图解说明
- 8 个核心原理详解：
  1. 分词系统 (Tokenization)
  2. 数据加载 (Data Loading)
  3. 模型架构 (Model Architecture)
  4. 训练优化 (Training Optimization)
  5. 推理引擎 (Inference Engine)
  6. 微调方法 (Fine-tuning & Midtraining)
  7. 评估指标 (Evaluation Metrics)
  8. 分布式训练 (Distributed Training)

**关键部分**:
- 📊 工作流程与数据流
- 🔢 关键数学公式
- 🚀 快速开始步骤
- 🎓 学习路径建议

### 📄 ADVANCED_PRINCIPLES_CN.md

**适合人群**: 有一定 ML 基础的学习者

**主要内容**:
- Transformer 架构的完整前向传播过程
- QK Normalization 的作用原理
- 组查询注意力 (GQA) 的显存优化
- 旋转位置编码 (RoPE) 的数学原理
- All-Reduce 的优化算法
- KV 缓存的精妙设计
- 交叉熵损失的计算与梯度
- 性能瓶颈分析与缓解策略

**深入讨论**:
- 为什么使用 RoPE？
- GQA 如何节省显存？
- KV 缓存为何能 100 倍加速？
- 通信隐藏技巧

### 📄 QUICK_REFERENCE_CN.md

**适合人群**: 所有人（日常查阅）

**主要内容**:
- 速查表（模型规格、参数、执行时间）
- 文件执行时间参考
- 数据流路径说明
- 常用命令速查
- 调试技巧和常见错误解决
- 性能基准数据
- 扩展指南（添加任务、自定义架构）
- FAQ 常见问题

**快速定位**:
- 🚀 找不到命令？→ 常用命令速查
- 💥 遇到错误？→ 常见错误及解决
- 📈 想扩展？→ 扩展指南
- ❓ 有疑问？→ FAQ

---

## 🎯 按学习目标查阅

### "我想理解 NanoChat 的整体架构"

→ 读 **PROJECT_MAP_CN.md** 的：
- 项目结构与文件导航
- 工作流程与数据流
- 8 个关键原理详解的前 4 个

预计时间: 1 小时

### "我想学习如何运行 NanoChat"

→ 读 **QUICK_REFERENCE_CN.md** 的：
- 常用命令速查
- 文件执行时间参考
- 调试技巧

→ 然后读 **PROJECT_MAP_CN.md** 的：
- 快速开始步骤

预计时间: 30 分钟

### "我想深入理解 Transformer 和注意力机制"

→ 读 **ADVANCED_PRINCIPLES_CN.md** 的：
- Transformer 架构详解
- 注意力机制优化
- 位置编码

→ 然后阅读相关论文

预计时间: 2-3 小时

### "我想优化推理性能"

→ 读 **ADVANCED_PRINCIPLES_CN.md** 的：
- 推理优化策略
- 性能瓶颈分析

→ 读 **QUICK_REFERENCE_CN.md** 的：
- 性能基准

预计时间: 1 小时

### "我想修改模型架构或添加新功能"

→ 读 **QUICK_REFERENCE_CN.md** 的：
- 扩展指南

→ 读 **PROJECT_MAP_CN.md** 相关模块说明

→ 查看对应的源代码

预计时间: 取决于修改复杂度

### "我遇到了错误想调试"

→ 查 **QUICK_REFERENCE_CN.md** 的：
- 常见错误及解决

→ 使用：
- 调试技巧
- 检查 GPU 状态
- 性能分析

预计时间: 20-30 分钟

---

## 📚 核心概念速查

### 分词 (Tokenization)

| 问题 | 答案位置 |
|------|---------|
| 什么是 BPE？ | PROJECT_MAP_CN.md → 关键原理详解 → 1. 分词系统 |
| RustBPE vs HuggingFace？ | QUICK_REFERENCE_CN.md → 分数与分词 |
| 如何训练自定义分词器？ | QUICK_REFERENCE_CN.md → 常用命令速查 |

### 模型架构

| 问题 | 答案位置 |
|------|---------|
| GPT 模型的结构？ | PROJECT_MAP_CN.md → 关键原理详解 → 3. 模型架构 |
| QK Norm 的作用？ | ADVANCED_PRINCIPLES_CN.md → 注意力机制优化 |
| 为什么用 RoPE？ | ADVANCED_PRINCIPLES_CN.md → 位置编码 |
| GQA 如何工作？ | ADVANCED_PRINCIPLES_CN.md → 注意力机制优化 |

### 训练

| 问题 | 答案位置 |
|------|---------|
| 梯度累积是什么？ | PROJECT_MAP_CN.md → 关键原理详解 → 4. 训练优化 |
| Chinchilla 定律是什么？ | PROJECT_MAP_CN.md → 关键原理详解 → 关键数学公式 |
| 双优化器的好处？ | PROJECT_MAP_CN.md → 关键原理详解 → 4. 训练优化 |
| 如何解决显存不足？ | QUICK_REFERENCE_CN.md → 常见错误及解决 |

### 推理

| 问题 | 答案位置 |
|------|---------|
| KV 缓存怎么工作？ | ADVANCED_PRINCIPLES_CN.md → 推理优化策略 |
| 采样方法有哪些？ | ADVANCED_PRINCIPLES_CN.md → 推理优化策略 |
| 如何做批量推理？ | ADVANCED_PRINCIPLES_CN.md → 推理优化策略 |
| Web UI 怎么启动？ | QUICK_REFERENCE_CN.md → 常用命令速查 |

### 分布式训练

| 问题 | 答案位置 |
|------|---------|
| DDP 怎么工作？ | PROJECT_MAP_CN.md → 关键原理详解 → 8. 分布式训练 |
| All-Reduce 是什么？ | ADVANCED_PRINCIPLES_CN.md → 分布式训练深入 |
| 通信隐藏？ | ADVANCED_PRINCIPLES_CN.md → 分布式训练深入 |
| 多节点训练？ | 查看源代码中的 RANK, WORLD_SIZE 环境变量 |

---

## 🔗 外部资源

### 推荐论文

| 论文 | 相关章节 |
|-----|---------|
| Attention is All You Need | ADVANCED_PRINCIPLES_CN.md 全部 |
| Rotary Position Embeddings | ADVANCED_PRINCIPLES_CN.md → 位置编码 |
| Grouped Query Attention | ADVANCED_PRINCIPLES_CN.md → 注意力机制优化 |
| Training Compute-Optimal LLMs (Chinchilla) | PROJECT_MAP_CN.md → 关键原理详解 → 关键数学公式 |

### 代码导航

| 功能 | 文件 | 相关文档 |
|------|------|---------|
| 分词系统 | `nanochat/tokenizer.py` | PROJECT_MAP_CN.md → 关键原理详解 → 1 |
| 模型架构 | `nanochat/gpt.py` | ADVANCED_PRINCIPLES_CN.md → Transformer 架构详解 |
| 数据加载 | `nanochat/dataloader.py` | PROJECT_MAP_CN.md → 关键原理详解 → 2 |
| 推理引擎 | `nanochat/engine.py` | ADVANCED_PRINCIPLES_CN.md → 推理优化策略 |
| 训练循环 | `scripts/base_train.py` | PROJECT_MAP_CN.md → 关键原理详解 → 4 |

---

## ⏱️ 预计学习时间

| 目标 | 文档 | 时间 |
|-----|------|------|
| 理解项目结构 | PROJECT_MAP_CN.md | 1 小时 |
| 学会运行训练 | QUICK_REFERENCE_CN.md | 30 分钟 |
| 深入理解 Transformer | ADVANCED_PRINCIPLES_CN.md | 2-3 小时 |
| 修改代码 | 相关文档 + 源代码 | 1-2 小时 |
| **完整学习** | **全部** | **5-7 小时** |

---

## 🎓 建议学习路径

### 路径 A: 快速上手（2-3 小时）

```
1. README.md (英文) - 15 分钟
   ↓
2. PROJECT_MAP_CN.md 前 50% - 30 分钟
   ↓
3. QUICK_REFERENCE_CN.md 常用命令 - 20 分钟
   ↓
4. 运行 speedrun.sh 实践 - 4 小时 (可同时进行)
```

### 路径 B: 深度理解（5-7 小时）

```
1. PROJECT_MAP_CN.md - 1 小时
   ↓
2. QUICK_REFERENCE_CN.md - 1 小时
   ↓
3. ADVANCED_PRINCIPLES_CN.md - 2-3 小时
   ↓
4. 阅读相关源代码 - 1-2 小时
   ↓
5. 动手修改和实验 - 自由
```

### 路径 C: 实战型（1 周）

```
Day 1-2: 快速上手（路径 A）
Day 3-4: 深度学习（路径 B）
Day 5-6: 修改和扩展（按需求）
Day 7: 实验和优化
```

---

## 💡 学习技巧

### 有效学习建议

1. **边读边做**: 不要只读文档，实际运行命令
2. **记笔记**: 用自己的话总结关键概念
3. **对比理解**: 理解"为什么"，而不仅仅是"是什么"
4. **渐进深入**: 先理解整体，再深入细节
5. **动手实验**: 修改参数看效果，改进代码

### 遇到问题

1. **查相关文档**: 用目录快速定位
2. **搜索关键词**: 在文档中搜索
3. **阅读源代码**: 代码是最好的文档
4. **查看 GitHub Issues**: 看别人的问题
5. **实验**: 修改参数看会发生什么

---

## 📞 获取帮助

- **GitHub Issues**: 报告 bug 或提问
- **GitHub Discussions**: 一般讨论
- **DeepWiki**: 代码问答（https://deepwiki.com/karpathy/nanochat）
- **本地测试**: 修改参数实验

---

## 📝 文档维护

| 文档 | 最后更新 | 覆盖内容 |
|-----|--------|---------|
| PROJECT_MAP_CN.md | 2025-11 | 项目结构、基本原理 |
| ADVANCED_PRINCIPLES_CN.md | 2025-11 | 深度算法和优化 |
| QUICK_REFERENCE_CN.md | 2025-11 | 命令、参数、调试 |

---

## 🎯 总结

你现在已经有了：

✅ **PROJECT_MAP_CN.md** - 了解项目的全貌  
✅ **ADVANCED_PRINCIPLES_CN.md** - 理解底层原理  
✅ **QUICK_REFERENCE_CN.md** - 日常快速查阅  

下一步：选择适合你的学习路径，开始学习和实验！

**祝你学习愉快！** 🚀

---

**推荐开始阅读顺序**:

1️⃣ 本文件（了解文档结构）← 你现在这里  
2️⃣ [PROJECT_MAP_CN.md](PROJECT_MAP_CN.md)（获取全面理解）  
3️⃣ [QUICK_REFERENCE_CN.md](QUICK_REFERENCE_CN.md)（学会操作）  
4️⃣ [ADVANCED_PRINCIPLES_CN.md](ADVANCED_PRINCIPLES_CN.md)（深度学习）

---
