"""
我们模型的高效推理引擎。

一切都围绕 token 序列工作：
- 用户可以向引擎发送 token 序列
- 引擎返回下一个 token

注意：
- 引擎对分词一无所知，它纯粹处理 token id 序列。

整个设计尽可能高效。
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext 

# -----------------------------------------------------------------------------
# 计算器工具辅助函数
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # 忽略计算器使用错误是可以的
        return None

def use_calculator(expr):
    """
    安全地求值 Python 表达式。
    支持数学表达式和字符串操作（如 .count()）
    """
    # 移除数字中的逗号
    expr = expr.replace(",", "")

    # 检查是否为纯数学表达式（旧行为）
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # 禁止幂运算符
            return None
        return eval_with_timeout(expr)

    # 检查是否为我们支持的字符串操作
    # 允许：字符串（单/双引号）、.count()、字母、数字、空格、括号
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # 禁止危险模式
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # 目前只允许 .count() 方法（以后可以扩展）
    if '.count(' not in expr:
        return None

    # 带超时求值
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    与 GPT 模型配合使用以维护 KV 缓存。
    注意，.pos 在 Transformer 最后一层插入后会自动前进。
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # K/V 各自的形状为 (B, H, T, D)，每层 Transformer 有一个。
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # 缓存中的当前时间位置

    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        从另一个 KV 缓存预填充。可选地沿批次维度扩展。
        当我们进行批次 1 预填充，然后想要从那里并行生成
        多个样本时使用。
        """
        # 1) 验证形状
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers、k/v、num_heads、head_dim 必须匹配
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size 可以扩展
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len：self 必须比 other 长
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) 初始化缓存
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) 复制数据
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) 更新位置
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # 在这里延迟初始化缓存，因为需要知道 dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # 将新的键/值插入缓存并返回到目前为止的完整缓存
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # 如果需要则动态增长缓存
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # 所需数量加 1024 缓冲
            t_needed = (t_needed + 1023) & ~1023 # 然后向上取整到 1024 的倍数
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # 将 k, v 插入缓存
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # 返回到当前位置的完整缓存键/值（作为视图）
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # 在 Transformer 最后一层处理后递增 pos
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """从形状为 (B, vocab_size) 的 logits 中采样单个下一个 token。返回 (B, 1)。"""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # 生成过程中的每行状态跟踪
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # 此行的当前 token 序列
        self.forced_tokens = deque() # 强制注入的 token 队列
        self.in_python_block = False # 是否在 python 块内
        self.python_expr_tokens = [] # 当前 python 表达式的 token
        self.completed = False # 此行是否已完成生成

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # 工具使用时需要

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """与 generate 相同，但进行单次预填充然后克隆 KV 缓存。"""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # 获取协调工具使用状态机所需的特殊 token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # 如果采样到，结束该行
        bos = self.tokenizer.get_bos_token_id() # 如果采样到，结束该行

        # 1) 对提示 token 进行批次 1 预填充
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) 为每个样本/行复制 KV 缓存
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # 复制后释放此内存

        # 3) 为每个样本初始化状态
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) 主生成循环
        num_generated = 0
        first_iteration = True
        while True:
            # 停止条件：已达到最大 token 数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # 停止条件：所有行都已完成
            if all(state.completed for state in row_states):
                break

            # 获取采样的 token - 来自预填充或前向传播
            if first_iteration:
                # 使用我们从预填充中已采样的 token
                sampled_tokens = [sampled_tokens[0]] * num_samples  # 将第一个 token 广播到所有行
                # TODO: 我们应该为每行采样一个 token 而不是广播
                first_iteration = False
            else:
                # 前向传播模型并获取每行的下一个 token
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) 在最后一个时间步
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # 处理每行：选择下一个 token，更新状态，可选工具使用
            token_column = [] # 包含每行的下一个 token id
            token_masks = [] # 包含每行的掩码（采样(1)还是强制(0)?）
            for i, state in enumerate(row_states):
                # 选择此行的下一个 token
                is_forced = len(state.forced_tokens) > 0 # 队列中是否有等待强制的 token?
                token_masks.append(0 if is_forced else 1) # 强制为 0，采样为 1
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # 更新此行的状态以包含下一个 token
                state.current_tokens.append(next_token)
                # 遇到 <|assistant_end|> 或 <|bos|> 时，标记该行为已完成
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # 处理工具逻辑
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # 产出 token 列
            yield token_column, token_masks
            num_generated += 1
            # 为下一次迭代准备 ids
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        非流式批量生成，只返回最终的 token 序列。
        返回 token 序列列表（列表的列表，元素为整数）。
        终止 token（assistant_end、bos）不包含在结果中。
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # 如果所有行都已完成则停止
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    快速内联测试，确保朴素/慢速的 model.generate 函数
    与这里更快的 Engine.generate 函数等效。
    """
    import time
    # 初始化计算
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # 通用超参数
    kwargs = dict(max_tokens=64, temperature=0.0)
    # 设置起始提示
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # 使用 model.generate() 函数生成参考序列
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # 使用 Engine 生成 token
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # 注意：以 fp32 运行
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0] # 只打印第一行
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # 比较两个序列
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
