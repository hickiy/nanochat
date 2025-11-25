# nanochat

![nanochat logo](dev/nanochat.png)

> 100美元能买到的最佳ChatGPT。

这个仓库是一个完整的LLM（如ChatGPT）的全栈实现，包含在一个干净、最小化、可黑客、可依赖轻量的代码库中。nanochat设计为在单个8XH100节点上运行，通过像[speedrun.sh](speedrun.sh)这样的脚本，从头到尾运行整个管道。这包括分词、预训练、微调、评估、推理和通过简单UI进行网页服务，以便您可以像与ChatGPT对话一样与自己的LLM对话。nanochat将成为Eureka Labs开发的LLM101n课程的顶点项目。

## 与它对话

为了感受这个仓库的端点，您目前可以在[nanochat.karpathy.ai](https://nanochat.karpathy.ai/)上找到托管的[nanochat d32](https://github.com/karpathy/nanochat/discussions/8)。"d32"意味着这个模型在Transformer神经网络中有32层。这个模型有19亿参数，通过简单运行单个脚本[run1000.sh](run1000.sh)，在380亿个token上训练，总训练成本约为800美元（在8XH100 GPU节点上约33小时训练时间）。虽然今天这足以超越2019年的GPT-2，但与现代大型语言模型如GPT-5相比，它大大落后。与这些微模型对话时，您会看到它们犯了很多错误，有点天真和愚蠢，它们幻觉很多，有点像孩子。这有点有趣。但nanochat独特之处在于它是完全属于您的——完全可配置、可调整、可黑客，并由您从头到尾训练。要训练和与自己的对话，我们转向...

## 快速开始

感受魔力的最快方式是运行速度运行脚本[speedrun.sh](speedrun.sh)，它训练并推理100美元级别的nanochat。在24美元/小时的8XH100节点上，这总运行时间约为4小时。从您喜欢的提供商（例如，我使用并喜欢[Lambda](https://lambda.ai/service/gpu-cloud)）启动一个新的8XH100 GPU盒子，并启动训练脚本：

```bash
bash speedrun.sh
```

或者，由于脚本运行4小时，我喜欢在新的screen会话`speedrun`中启动它（并将输出记录到`speedrun.log`）：

```bash
screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
```

如果您不太熟悉，请查看[screen cheatsheet](https://gist.github.com/jctosta/af918e1618682638aa82)。您可以在screen会话中观看它运行，或使用`Ctrl-a d`分离并`tail speedrun.log`查看进度。现在等待4小时。一旦完成，您可以通过ChatGPT-like网页UI与您的LLM对话。确保您的本地uv虚拟环境处于活动状态（运行`source .venv/bin/activate`），并服务它：

```bash
python -m scripts.chat_web
```

然后访问显示的URL。确保正确访问，例如在Lambda上使用您所在节点的公共IP，后跟端口，例如[http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/)等。然后像正常与ChatGPT对话一样与您的LLM对话！让它写故事或诗歌。问它您是谁以看到幻觉。问它为什么天空是蓝色的。或为什么它是绿色的。速度运行是一个4e19 FLOPs能力的模型，所以有点像与幼儿园孩子对话 :)。

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

您也可以`cat report.md`文件，它出现在项目目录中，包含运行的"成绩单"，即一堆评估和指标。在最后，您会看到一个摘要表，例如：

---

- 字符：333,989
- 行：8,304
- 文件：44
- Token（约）：83,497
- 依赖（uv.lock行）：2,004

| 指标          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2219   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2875   | 0.2807   | -        |
| ARC-Easy        | -        | 0.3561   | 0.3876   | -        |
| GSM8K           | -        | 0.0250   | 0.0455   | 0.0758   |
| HumanEval       | -        | 0.0671   | 0.0854   | -        |
| MMLU            | -        | 0.3111   | 0.3151   | -        |
| ChatCORE        | -        | 0.0730   | 0.0884   | -        |

总时钟时间：3h51m

---

（您的表可能默认缺少RL数字）。有关速度运行脚本以及要查找和期望的内容的更多信息，请参考我在仓库讨论中发布的演练：["Introducing nanochat: The best ChatGPT that $100 can buy"](https://github.com/karpathy/nanochat/discussions/1)。

## 更大的模型

不出所料，100美元不足以训练高性能的ChatGPT克隆。实际上，LLM以其数百万美元的资本支出而闻名。对于我们的目的，我认为有两个更感兴趣的规模。首先是约300美元级别的d26模型（即depth=26），训练约12小时，略微超越GPT-2 CORE分数。其次是1000美元级别（约41.6小时），因为这是一个不错的整数。但这两个尚未完全支持，因此尚未在主分支中附加。

话虽如此，为了给出一个感觉，训练GPT-2级模型d26所需的[speedrun.sh](speedrun.sh)文件的示例更改仅涉及三个更改：

```bash
...
# 您需要下载更多预训练数据分片
# 获取参数数量，乘以20得到token，乘以4.8得到字符，
# 除以2.5亿得到分片数量。todo需要改进这个...
python -m nanochat.dataset -n 450 &
...
# 使用--depth增加模型大小。为了不oom，将设备批大小32 -> 16：
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=26 --device_batch_size=16
...
# 确保在midtraining期间使用相同的：
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train -- --device_batch_size=16
```

就是这样！要注意的最大事情是确保您有足够的数据分片来训练（否则代码会循环并在同一训练集上做更多epoch，稍微降低学习速度），并管理您的内存/VRAM，主要通过减少`device_batch_size`直到适合（脚本会自动补偿，通过增加梯度累积循环的数量，将并行计算转换为顺序计算）。

以及更多关于运行nanochat的计算环境：

- 代码在Ampere 8XA100 GPU节点上也能正常运行，但稍慢一些。
- 通过省略`torchrun`，所有代码在单个GPU上也能正常运行，并产生几乎相同的结果（代码会自动切换到梯度累积），但您需要等待8倍的时间。
- 如果您的GPU有少于80GB，您需要调整一些超参数，否则会OOM/耗尽VRAM。在脚本中查找`--device_batch_size`并减少它直到适合。例如从32（默认）到16、8、4、2，甚至1。少于这个您需要知道更多自己在做什么，并更有创意。
- 大部分代码是相当标准的PyTorch，所以它应该在任何支持它的东西上运行 - xpu、mps等，但我还没有开箱即用实现，所以可能需要一些调整。

## 在CPU / MPS上运行

nanochat可以在CPU或MPS（如果您在Macbook上）上运行，并会自动尝试检测最佳运行设备。您没有GPU不会走得太远，但至少您可以运行代码路径，也许有耐心训练一个微小的LLM。作为如何使所有运行命令更小的一个例子（随意调整！），您可以参考[dev/runcpu.sh](dev/runcpu.sh)文件。您会看到我基本上限制所有脚本训练更小的模型，运行更少的迭代等。这个功能是新的，稍微粗糙（触及了很多代码），并在2025年10月21日的[CPU|MPS PR](https://github.com/karpathy/nanochat/pull/88)中合并。

## 定制

要定制您的nanochat，请参阅讨论中的[指南：为您的nanochat注入身份](https://github.com/karpathy/nanochat/discussions/139)，它描述了如何通过合成数据生成并将该数据混合到midtraining和SFT阶段来调整您的nanochat个性。

此外，要为nanochat添加新能力，请参阅[指南：数草莓中的r（以及如何一般添加能力）](https://github.com/karpathy/nanochat/discussions/164)。

## 问题

nanochat设计为简短而甜美。一个大优势是我们可以将所有文件打包并复制粘贴到您喜欢的LLM中询问任意问题。作为一个例子，我喜欢使用[files-to-prompt](https://github.com/simonw/files-to-prompt)工具打包仓库，如下：

```bash
files-to-prompt . -e py -e md -e rs -e html -e toml -e sh --ignore "*target*" --cxml > packaged.txt
```

这包括所有py、rs、html、toml、sh文件，排除`rustbpe/target`文件夹，并选择cxml输出格式。一切写入`packaged.txt`文件，目前测量约330KB（即远低于100K token的先进LLM），以及45个文件中的约8K行代码。

或者，我推荐使用[DeepWiki](https://deepwiki.com/karpathy/nanochat)从Devin/Cognition询问这个仓库的问题。在这个仓库的URL中，只需将github.com更改为deepwiki.com，您就可以开始了。

## 测试

我在这里没有投入太多，但有一些测试存在，特别是对于分词器。例如运行：

```bash
python -m pytest tests/test_rustbpe.py -v -s
```

## 文件结构

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # 身份的示例合成数据
│   ├── generate_logo.html
│   ├── nanochat.png
│   ├── repackage_data_reference.py # 预训练数据分片生成
│   └── runcpu.sh                   # 在CPU/MPS上运行的小示例
├── nanochat
│   ├── __init__.py                 # 空
│   ├── adamw.py                    # 分布式AdamW优化器
│   ├── checkpoint_manager.py       # 保存/加载模型检查点
│   ├── common.py                   # 杂项小工具，质量生活
│   ├── configurator.py             # argparse的优越替代品
│   ├── core_eval.py                # 评估基础模型CORE分数（DCLM论文）
│   ├── dataloader.py               # 分词分布式数据加载器
│   ├── dataset.py                  # 预训练数据的下载/读取工具
│   ├── engine.py                   # 带KV缓存的高效模型推理
│   ├── execution.py                # 允许LLM作为工具执行Python代码
│   ├── gpt.py                      # GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # 评估每字节位（而不是损失）
│   ├── muon.py                     # 分布式Muon优化器
│   ├── report.py                   # 编写nanochat报告的工具
│   ├── tokenizer.py                # GPT-4风格的BPE分词器包装器
│   └── ui.html                     # nanochat前端的HTML/CSS/JS
├── pyproject.toml
├── run1000.sh                      # 训练约800美元的nanochat d32
├── rustbpe                         # 自定义Rust BPE分词器训练器
│   ├── Cargo.lock
│   ├── Cargo.toml
│   ├── README.md                   # 见为什么这个存在
│   └── src
│       └── lib.rs
├── scripts
│   ├── base_eval.py                # 基础模型：计算CORE分数
│   ├── base_loss.py                # 基础模型：计算每字节位，采样
│   ├── base_train.py               # 基础模型：训练
│   ├── chat_cli.py                 # 聊天模型（SFT/Mid）：通过CLI对话
│   ├── chat_eval.py                # 聊天模型（SFT/Mid）：评估任务
│   ├── chat_rl.py                  # 聊天模型（SFT/Mid）：强化学习
│   ├── chat_sft.py                 # 聊天模型：训练SFT
│   ├── chat_web.py                 # 聊天模型（SFT/Mid）：通过WebUI对话
│   ├── mid_train.py                # 聊天模型：midtraining
│   ├── tok_eval.py                 # 分词器：评估压缩率
│   └── tok_train.py                # 分词器：训练它
├── speedrun.sh                     # 训练约100美元的nanochat d20
├── tasks
│   ├── arc.py                      # 多选科学问题
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # 从任意jsonl对话制作任务
│   ├── gsm8k.py                    # 8K小学数学问题
│   ├── humaneval.py                # 误称；简单Python编码任务
│   ├── mmlu.py                     # 多选问题，广泛主题
│   ├── smoltalk.py                 # HF SmolTalk的综合数据集
│   └── spellingbee.py              # 教模型拼写/计数字母的任务
├── tests
│   └── test_engine.py
│   └── test_rustbpe.py
└── uv.lock
```

## 贡献

nanochat远未完成。目标是改善微模型的状态，这些模型在<1000美元的预算上可以端到端工作。可访问性不仅是总体成本，也是认知复杂性——nanochat不是一个详尽可配置的LLM"框架"；代码库中不会有巨大的配置对象、模型工厂或if-then-else怪物。它是一个单一、连贯、最小化、可读、可黑客、最大化可分叉的"强基线"代码库，设计为从头到尾运行并产生一个具体的ChatGPT克隆及其成绩单。

当前LLM政策：披露。在提交PR时，请声明任何有实质LLM贡献的部分，这些部分您没有写或不完全理解。

## 致谢

- 名称（nanochat）源自我之前的项目[nanoGPT](https://github.com/karpathy/nanoGPT)，它只涵盖预训练。
- nanochat也受[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)的启发，它用清晰的指标和排行榜游戏化了nanoGPT仓库，并借用了很多想法和一些预训练实现。
- 感谢[HuggingFace](https://huggingface.co/)的fineweb和smoltalk。
- 感谢[Lambda](https://lambda.ai/service/gpu-cloud)用于开发这个项目的计算。
- 感谢首席LLM耳语者🧙‍♂️ Alec Radford的建议/指导。
- 感谢仓库沙皇Sofie [@svlandeg](https://github.com/svlandeg)帮助管理nanochat的问题、拉取请求和讨论。

## 引用

如果您在研究中发现nanochat有帮助，请简单引用为：

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## 许可证

MIT
