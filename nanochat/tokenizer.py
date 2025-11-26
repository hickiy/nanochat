"""
GPT-4 风格的 BPE 分词器。

有两种实现可用：
1) HuggingFace Tokenizer，可以进行训练和推理，但相当令人困惑
2) 我们自己的 RustBPE Tokenizer 用于训练，tiktoken 用于高效推理
"""

import os
import copy
from functools import lru_cache

SPECIAL_TOKENS = [
    # 每个文档以 Beginning of Sequence (BOS) token 开始，用于分隔文档
    "<|bos|>",
    # 下面的 token 仅在微调期间用于将对话渲染为 token id
    "<|user_start|>", # 用户消息
    "<|user_end|>",
    "<|assistant_start|>", # 助手消息
    "<|assistant_end|>",
    "<|python_start|>", # 助手调用 python REPL 工具
    "<|python_end|>",
    "<|output_start|>", # python REPL 输出返回给助手
    "<|output_end|>",
]

# 注意：此分割模式与 GPT-4 的不同之处在于我们使用 \p{N}{1,2} 而非 \p{N}{1,3}
# 我这样做是因为不想为较小的词汇表大小"浪费"太多 token 在数字上。
# 我还没有验证这是否真的是个好主意，TODO。
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# 基于 HuggingFace Tokenizer 的通用 GPT-4 风格分词器
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """HuggingFace Tokenizer 的轻量级包装，提供一些实用功能"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # 从 HuggingFace 预训练分词器初始化（例如 "gpt2"）
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # 从磁盘上的本地目录初始化（例如 "out/tokenizer"）
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 从文本迭代器训练
        # 配置 HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # 归一化器：无
        tokenizer.normalizer = None
        # 预分词器：GPT-4 风格
        # GPT-4 用于在 BPE 之前将文本分割成组的正则表达式模式
        # 注意：该模式从 \p{N}{1,3} 改为 \p{N}{1,2}，因为我怀疑这对
        # 非常小的模型和较小的词汇表大小有害，因为在 token 空间上有点浪费。
        # （但我还没有验证这一点！TODO）
        gpt4_split_regex = Regex(SPLIT_PATTERN) # huggingface 要求你用 Regex 包装它！！
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # 解码器：ByteLevel（与 ByteLevel 预分词器配对使用）
        tokenizer.decoder = decoders.ByteLevel()
        # 后处理器：无
        tokenizer.post_processor = None
        # 训练器：BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # 无最小频率
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        # 启动训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)

    def _encode_one(self, text, prepend=None, append=None):
        # 编码单个字符串
        # prepend/append 可以是特殊 token 的字符串或直接是 token id。
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        # 通过精确匹配编码单个特殊 token
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        bos = self.encode_special("<|bos|>")
        return bos

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        # 将分词器保存到磁盘
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

# -----------------------------------------------------------------------------
# 基于 rustbpe + tiktoken 组合的分词器
import pickle
import rustbpe
import tiktoken

class RustBPETokenizer:
    """tiktoken 的轻量级包装（用于高效推理），但使用 rustbpe 训练"""

    def __init__(self, enc, bos_token):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) 使用 rustbpe 训练
        tokenizer = rustbpe.Tokenizer()
        # 特殊 token 稍后在 __init__ 中插入，这里不训练它们
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) 构建用于推理的关联 tiktoken 编码
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks, # dict[bytes, int] (token 字节 -> 合并优先级排名)
            special_tokens=special_tokens, # dict[str, int] (特殊 token 名称 -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken 将特殊文档分隔符 token 称为 "<|endoftext|>"
        # 是的，这很令人困惑，因为这个 token 几乎总是被前置到文档开头
        # 它最常用于在推理期间向 LLM 发出新序列开始的信号等。
        # 所以在 nanoChat 中我们总是使用 "<|bos|>"（"beginning of sequence"的缩写），
        # 但历史上它通常被称为 "<|endoftext|>"。
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        # text 可以是字符串或字符串列表

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id) # TODO: 这里效率稍低？:( 嗯
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id) # TODO: 同上
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        # 将编码对象保存到磁盘
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        对单个聊天对话（我们这里称为"doc"或"document"）进行分词。
        返回：
        - ids: list[int] 是此渲染对话的 token id 列表
        - mask: list[int] 相同长度，mask = 1 表示助手需要训练的 token。
        """
        # 我们将返回的 ids、masks 以及一个辅助函数来帮助构建它们。
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # 有时第一条消息是系统消息...
        # => 只需将其与第二条（用户）消息合并
        if conversation["messages"][0]["role"] == "system":
            # 这里需要进行一些对话手术...
            conversation = copy.deepcopy(conversation) # 避免修改原始对象
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # 获取我们需要的所有特殊 token
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # 现在我们可以对对话进行分词
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # 这里进行一些健全性检查，以防止踩坑
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content 可以是简单字符串或部分列表（例如包含工具调用）
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # 简单字符串 => 直接添加 token
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # 字符串部分 => 直接添加 token
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python 工具调用 => 在 <|python_start|> 和 <|python_end|> 之间添加 token
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python 输出 => 在 <|output_start|> 和 <|output_end|> 之间添加 token
                            # 这些 token 都不被监督，因为测试时这些 token 来自 Python
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # 截断到最大 max_tokens 个 token（帮助防止 OOM）
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """用于调试的小辅助函数：可视化 render_conversation 的分词结果"""
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        GRAY = '\033[90m'
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return '|'.join(tokens)

    def render_for_completion(self, conversation):
        """
        用于强化学习。在那种情况下，我们想要
        渲染对话以准备助手进行补全。
        与聊天 SFT 情况不同，我们不需要返回 mask。
        """
        # 我们需要进行一些手术：需要弹出最后一条消息（助手的）
        conversation = copy.deepcopy(conversation) # 避免修改原始对象
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # 原地移除最后一条消息（助手的）

        # 现在对对话进行分词
        ids, mask = self.render_conversation(conversation)

        # 最后，为了准备助手进行补全，追加助手开始 token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids

# -----------------------------------------------------------------------------
# nanochat 特定的便利函数

def get_tokenizer():
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)

def get_token_bytes(device="cpu"):
    import torch
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
