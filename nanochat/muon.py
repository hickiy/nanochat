"""
Keller 等人的 Muon 优化器。
也借鉴了 modded-nanogpt 的很多思路。
"""
import torch
from torch import Tensor
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz 迭代用于计算 G 的零次幂/正交化。我们选择使用
    五次迭代，其系数被选择为在零点处最大化斜率。为了最小化
    步数，事实证明即使在迭代不再在整个区间上完全收敛到 1 的
    点之后，继续增加零点处的斜率仍然是经验上有效的。因此，
    这个迭代不会产生 UV^T，而是产生类似 US'V^T 的东西，
    其中 S' 是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)，事实证明
    相对于 UV^T（其中 USV^T = G 是 SVD），这完全不会损害模型性能。
    """
    assert G.ndim >= 2 # @scottjmaddox 的批处理 Muon 实现，由 @YouJiacheng 在记录中付诸实践
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # 确保谱范数最多为 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # 执行 NS 迭代
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # 五次计算策略，采纳自 @jxbz、@leloykun 和 @YouJiacheng 的建议
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - 通过 Newton-Schulz 正交化的动量优化器

    https://kellerjordan.github.io/posts/muon/

    Muon 内部运行标准 SGD 动量，然后执行正交化后处理步骤，
    其中每个 2D 参数的更新被替换为最近的正交矩阵。为了高效地
    正交化每个更新，我们使用 Newton-Schulz 迭代，其优点是
    可以在 GPU 上稳定地以 bfloat16 运行。

    一些警告：
    - 此优化器不应用于嵌入层、最终全连接层或任何 {0,1}-D 参数；
      这些都应使用标准方法（例如 AdamW）进行优化。
    - 要将其用于 4D 卷积滤波器，只需展平其最后 3 个维度即可。

    参数：
        lr: 内部 SGD 使用的学习率。
        momentum: 内部 SGD 使用的动量。
        nesterov: 是否在内部 SGD 中使用 Nesterov 风格的动量。（推荐）
        ns_steps: Newton-Schulz 迭代的步数。
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)


class DistMuon(torch.optim.Optimizer):
    """
    Muon：SGD 动量 +（可选）Nesterov，然后通过 Newton-Schulz 正交化 2D 更新，
    最后应用宽高比缩放的步长。执行自己的分布式同步：
      - reduce_scatter(AVG) 用于梯度平均
      - all_gather 用于复制更新后的权重

    注意：
      * 专为 2D 参数设计（例如，重塑为 2D 的线性/卷积核）。不要用于 0D/1D
        参数，如嵌入或标量。
      * 动量缓冲区仅在每个参数的"所有者"进程上维护（进程由下面的
        块循环分配选择）。如果在单个进程上保存优化器状态检查点，
        请事先合并状态。

    参数：
        params: 可迭代的 Tensor
        lr: 学习率
        momentum: 动量系数，范围 [0,1)
        nesterov: 如果为 True，使用 Nesterov 风格更新（g <- lerp(g, buf, momentum)）；否则使用 buf
        ns_steps: 正交化的 Newton-Schulz 迭代步数
    """
    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        assert all(p.ndim == 2 for p in params), "Muon expects 2D parameters only"
        rank = dist.get_rank()
        # 按形状对所有参数进行分组
        shapes = sorted({p.shape for p in params}) # 排序以确保一致/确定性的顺序
        param_groups = []
        for shape in shapes:
            group_params = [p for p in params if p.shape == shape]
            device, dtype = group_params[0].device, group_params[0].dtype
            assert all(p.device == device for p in group_params)
            assert all(p.dtype == dtype for p in group_params)
            if rank == 0:
                print(f"Muon: Grouping {len(group_params)} params of shape {shape}, device {device}, dtype {dtype}")
            param_groups.append(dict(params=group_params, zero_buffer=torch.zeros_like(group_params[0])))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 确保所有梯度都存在
        assert all(p.grad is not None for group in self.param_groups for p in group["params"]), "All params must have grads"

        # 启动所有 reduce scatter 操作以跨所有进程平均梯度
        all_reduce_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # 以 world_size 为组遍历参数
            for base_i in range(0, len(params), world_size):
                # 每个参数的计算所有者是 rank i % world_size
                owner_idx = base_i + rank
                # 每个 rank 将其 world_size 个参数块堆叠成列表
                rs_input = [p.grad for p in params[base_i:base_i + world_size]]
                # 用零缓冲区填充 rs_input 以完成组
                rs_input.extend([zero_buffer] * (world_size - len(rs_input)))
                # 输出缓冲区根据 rank 跨组分布
                rs_output = params[owner_idx].grad if owner_idx < len(params) else torch.empty_like(zero_buffer)
                # 在这组 world_size 个参数内 reduce scatter 梯度
                work = dist.reduce_scatter(rs_output, rs_input, op=dist.ReduceOp.AVG, async_op=True).get_future()
                all_reduce_futures.append(work)

        # 现在每个 rank 计算更新并 gather
        future_idx = 0
        all_gather_futures = []
        for group in self.param_groups:
            params = group["params"]
            zero_buffer = group["zero_buffer"]
            # 以 world_size 为组遍历参数
            for base_i in range(0, len(params), world_size):
                # 每个参数的计算所有者是 rank i % world_size
                owner_idx = base_i + rank # 计算此 rank 拥有的参数索引
                # 等待 reduce scatter 完成
                all_reduce_futures[future_idx].wait() # 可能以后可以使用 wait_any 轮询代替
                future_idx += 1
                # 所有者计算 Muon 更新，结果在其参数中
                if owner_idx < len(params):
                    p = params[owner_idx]
                    g = p.grad  # 现在已跨 rank 平均
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    scale = (max(1.0, p.size(-2) / p.size(-1)) ** 0.5)
                    p.add_(g, alpha=-group["lr"] * scale)
                # 将更新后的参数复制到所有 rank
                ag_input = params[owner_idx] if owner_idx < len(params) else zero_buffer
                ag_output = params[base_i:base_i + world_size]
                ag_output.extend([torch.empty_like(zero_buffer) for _ in range(world_size - len(ag_output))]) # 填充
                work = dist.all_gather(ag_output, ag_input, async_op=True).get_future()
                all_gather_futures.append(work)

        # 等待所有工作完成
        torch.futures.collect_all(all_gather_futures).wait()
