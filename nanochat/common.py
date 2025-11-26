"""
nanochat 的通用工具函数。
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

class ColoredFormatter(logging.Formatter):
    """为日志消息添加颜色的自定义格式化器。"""
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 洋红色
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # 为日志级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # 格式化消息
        message = super().format(record)
        # 为消息的特定部分添加颜色
        if levelname == 'INFO':
            # 高亮数字和百分比
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # 默认将 nanochat 的中间文件与其他缓存数据一起存放在 ~/.cache 目录下
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    从 URL 下载文件到基础目录中的本地路径。
    使用锁文件防止多个进程同时下载。
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # 只有单个进程能获取此锁
        # 其他所有进程会阻塞直到锁被释放

        # 获取锁后重新检查
        if os.path.exists(file_path):
            return file_path

        # 下载内容为字节流
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # 字节流

        # 写入本地文件
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # 如果提供了后处理函数则执行
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # 使用 https://manytools.org/hacker-tools/ascii-banner/ 制作的炫酷 DOS Rebel 字体 ASCII 横幅
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp():
    # TODO 有没有更好的方法
    return int(os.environ.get('RANK', -1)) != -1

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # 优先使用 CUDA（如果可用），否则使用 MPS，最后回退到 CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """我们反复执行的基础初始化，因此将其抽取为公共函数。"""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # 可复现性
    # 注意这里我们设置了全局种子，但大部分代码使用显式的 rng 对象。
    # 全局 rng 可能被使用的唯一地方是 nn.Module 初始化模型权重时。
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # 暂时跳过完全可复现性，以后可能会研究其带来的性能下降
    # torch.use_deterministic_algorithms(True)

    # 精度
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # 矩阵乘法使用 tf32 而非 fp32

    # 分布式设置：分布式数据并行（DDP），可选，需要 CUDA
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # 将 "cuda" 默认指向此设备
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps 或 cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """compute_init 的配套函数，用于脚本退出前的清理工作"""
    if is_ddp():
        dist.destroy_process_group()

class DummyWandb:
    """当我们不想使用 wandb 但又想保持相同接口签名时很有用"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass
