"""
比较以下训练实现：

1. （非常慢）Python 参考实现
2. 优化的 Python 实现
3. HuggingFace tokenizers 训练实现
4. 我们自己的 RustBPE 训练实现

所有这些都应该计算相同的合并并产生
相同的词汇表和分词结果。

最后，推理时我们将使用 tiktoken 以提高效率。
所以我们要确保可以将我们的 rustbpe 分词器
导出到 tiktoken 并使用它进行推理，得到相同的结果。

运行方式：
python -m pytest tests/test_rustbpe.py -v -s
-v 是详细输出，-s 是显示打印
"""

import regex as re
from collections import Counter, defaultdict
import time
import rustbpe
import tiktoken
import pytest

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# 参考分词器，基本上是从 minbpe 复制粘贴并精简的

def get_stats(ids, counts=None):
    """
    给定一个整数列表，返回连续对的计数字典
    示例：[1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    可选地允许更新现有的计数字典
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # 遍历连续元素
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    在整数列表（ids）中，将所有连续出现的 pair
    替换为新的整数 token idx
    示例：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # 如果不在最后一个位置且 pair 匹配，则替换它
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class RegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: 可选字符串，用于覆盖默认值（GPT-4 分割模式）
        - special_tokens: str -> int 的特殊 token 字典
          示例：{'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {} # (int, int) -> int
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # 跟踪训练过程中合并是否存在歧义（pair 计数不唯一）
        ambiguous = False

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # 计算每个连续 pair 出现的次数
            stats = {}
            for chunk_ids in ids:
                # 传入 stats 会原地更新它，累加计数
                get_stats(chunk_ids, stats)
            # 找到计数最高的 pair
            pair = max(stats, key=stats.get)
            # 检查合并是否存在歧义 - 即最大值不唯一
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                # 打印计数最高的前 10 个 pair
                # print(f"{i} Merge is ambiguous! {pair} has {pair_count} occurrences")
                # for print_pair, print_count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                #     print(f"{print_pair}: {print_count}")
                ambiguous = True
            # 铸造新 token：为其分配下一个可用的 id
            idx = 256 + i
            # 在 ids 中将所有 pair 替换为 idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # 保存合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # 打印信息
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        return ambiguous

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# 更快的 Python 分词器，参考分词器的优化版本

def fast_merge_inplace(ids, pair, idx):
    """
    在整数列表（ids）中，原地将所有连续出现的 pair
    替换为新的整数 token idx
    示例：ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # 找到 pair 出现的所有位置
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx
            ids.pop(i+1)
        else:
            i += 1
    return ids


class FastRegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        引入了多项优化：
        - 通过内联函数消除函数调用开销
        - 使用 .pop() 原地修改 id 列表而不是创建新列表
        - 将相同的块折叠为唯一的块
        - 更聪明地更新计数 - 只在受影响的块周围更新
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # 许多块是相同的，所以我们可以"折叠"它们为唯一的块
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # 输入文本预处理
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # 迭代合并最常见的 pair 以创建新 token
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # 初始计数：构建 stats 和位置跟踪
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> 包含此 pair 的块索引集合

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        for i in range(num_merges):
            if not stats:
                break

            # 找到计数最高的 pair
            pair = max(stats, key=stats.get)
            # 铸造新 token：为其分配下一个可用的 id
            idx = 256 + i

            # 获取包含此 pair 的块
            affected_chunks = positions[pair]

            # 跟踪增量更新的计数变化
            count_changes = defaultdict(int)

            # 只在受影响的块中替换所有 pair 出现
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # 跟踪正在移除/添加的 pair
                        # 移除：(prev, A), (A, B), (B, next)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # 合并的 pair 消失
                        count_changes[pair] -= chunk_count

                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # 应用合并
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)

                        # 添加：(prev, C), (C, next)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # 对 stats 和 positions 应用增量变化
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # 合并的 pair 应该完全消失
                    continue

                stats[changed_pair] += delta

                # 更新改变的 pair 的 positions - 只检查受影响的块
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # 完全移除合并的 pair
            del stats[pair]
            del positions[pair]

            # 保存合并
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens 是 str -> int 的字典
        # 示例：{"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # 给定 ids（整数列表），返回 Python 字符串
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# HuggingFace 分词器
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """HuggingFace Tokenizer 的轻量级包装，提供一些实用功能"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
        gpt4_split_regex = Regex(GPT4_SPLIT_PATTERN) # huggingface 要求你用 Regex 包装它！！
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
            special_tokens=[], # 无特殊 token
        )
        # 启动训练
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

# -----------------------------------------------------------------------------
# 测试以上所有实现

@pytest.fixture(scope="module")
def enwik8_path():
    """下载并缓存 enwik8 数据集的 fixture。"""
    import os
    import zipfile
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    # 将 enwik8 下载并解压到 .cache 目录
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """提供 100KB enwik8 用于快速测试的 fixture。"""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(100_000)

@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """提供 10MB enwik8 用于性能测试的 fixture。"""
    with open(enwik8_path, "r", encoding="utf-8") as f:
        return f.read(10**7)

def time_function(func, *args, **kwargs):
    """计时函数调用并返回结果和耗时"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed

def test_correctness(enwik8_small):
    """测试所有分词器实现产生相同的结果。"""
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 20 次合并

    # 训练慢速参考
    print("\nTraining slow reference...")
    slow_reference_tokenizer = RegexTokenizer()
    ambiguous_flag, slow_reference_train_time = time_function(slow_reference_tokenizer.train, text, vocab_size)
    slow_reference_ids, slow_reference_encode_time = time_function(slow_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Slow reference train time: {slow_reference_train_time:.4f}s")
    print(f"Slow reference encode time: {slow_reference_encode_time:.4f}s")
    print(slow_reference_ids[:20])

    if ambiguous_flag:
        print("‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size")
        print("The implementation could be correct but we might see different results below")
    else:
        print("✅ Merge order is NOT ambiguous")

    # 训练快速参考
    print("\nTraining fast reference...")
    fast_reference_tokenizer = FastRegexTokenizer()
    _, fast_reference_train_time = time_function(fast_reference_tokenizer.train, text, vocab_size)
    fast_reference_ids, fast_reference_encode_time = time_function(fast_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Fast reference train time: {fast_reference_train_time:.4f}s")
    print(f"Fast reference encode time: {fast_reference_encode_time:.4f}s")
    print(fast_reference_ids[:20])

    # 断言快速等于慢速
    assert fast_reference_ids == slow_reference_ids, "Fast reference should match slow reference"
    print("✅ Fast == Slow")

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace 有不同的字节顺序，所以我们需要自定义匹配
    def custom_match(ids1, ids2):
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    assert custom_match(hf_ids, fast_reference_ids), "HuggingFace should match fast reference"
    print("✅ HuggingFace == Fast")

    # 最后使用我们自己的 Rust 实现
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    rustbpe_ids, rustbpe_encode_time = time_function(rustbpe_tokenizer.encode, encode_text)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    print(f"RustBPE encode time: {rustbpe_encode_time:.4f}s")
    print(rustbpe_ids[:20])

    assert rustbpe_ids == fast_reference_ids, "RustBPE should match fast reference"
    print("✅ RustBPE == Fast")

    # 现在将 rustbpe 导出到 tiktoken 以进行更高效的推理
    print("\nTesting tiktoken export...")
    pattern = rustbpe_tokenizer.get_pattern()
    mergeable_ranks_list = rustbpe_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == rustbpe_ids, "Tiktoken should match RustBPE"
    print("✅ Tiktoken == RustBPE")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """使用更大的数据集并比较优化分词器（Python、Rust、HuggingFace）的训练速度。"""
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # 注释掉因为太慢
    # 训练优化的 python 版本
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # 训练 rustbpe
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    assert rustbpe_train_time > 0, "Training should take some time"

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # 打印比较结果
    print(f"\n📊 Performance comparison:")
    print(f"   RustBPE: {rustbpe_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/rustbpe_train_time:.2f}x")

def test_interface(enwik8_small):
    """测试 RustBPETokenizer 的训练、编码、解码和序列化接口。"""
    import tempfile
    from nanochat.tokenizer import RustBPETokenizer

    # 简单训练测试
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator([enwik8_small], vocab_size)
    assert tok.get_vocab_size() == vocab_size, f"Expected vocab size {vocab_size}, got {tok.get_vocab_size()}"
    print(f"✅ Trained tokenizer with vocab size {vocab_size}")

    # 编码/解码文本
    encode_text = "Hello world! How are you? 🙃"
    ids = tok.encode(encode_text)
    print(f"\nInput text: {encode_text}")
    print(f"IDs: {ids}")
    decoded = tok.decode(ids)
    print(f"Decoded: {decoded}")
    assert decoded == encode_text, f"Decoded text doesn't match: {decoded} != {encode_text}"
    print("✅ Encode/decode test passed")

    # 批量编码测试
    ids_new = tok.encode([encode_text, encode_text])
    assert all(x == ids for x in ids_new), "Batch encoding should produce identical results"
    print("✅ Encode batch OK")

    # append/prepend 功能
    ids_special = tok.encode(encode_text, prepend="<|bos|>", append="<|bos|>")
    bos_token_id = tok.encode_special("<|bos|>")
    assert ids_special == [bos_token_id] + ids + [bos_token_id], "Special tokens not correctly added"
    print("✅ append/prepend OK")

    # 通过临时目录保存/加载测试
    with tempfile.TemporaryDirectory() as tmp_dir:
        tok.save(tmp_dir)
        tok_reloaded = RustBPETokenizer.from_directory(tmp_dir)
        ids_reloaded = tok_reloaded.encode(encode_text)
        assert ids_reloaded == ids, "Reloaded tokenizer should produce same results"
        print("✅ Save/load through temporary directory OK")
