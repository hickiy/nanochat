"""
基础/预训练数据集是一组 parquet 文件。
此文件包含以下功能：
- 遍历 parquet 文件并从中产出文档
- 如果文件不在磁盘上则按需下载

关于数据集准备的详细信息，请参见 `repackage_data_reference.py`。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool
from nanochat.common import get_base_dir

def get_proxy_dict():
    """PROXIES 从仓库根目录的 config.toml 读取 [proxy] 节；如果未读取到则使用默认值。"""
    _config_path = os.path.join(os.getcwd(), "config.toml")
    try:
        import toml as _toml_pkg  # toml
        _cfg = _toml_pkg.load(_config_path)
        _proxy_section = _cfg.get("proxy", None)
        if isinstance(_proxy_section, dict):
            return _proxy_section
    except FileNotFoundError:
        print(f"Proxy config file not found: {_config_path}")
    except PermissionError:
        print(
            f"Permission denied when accessing proxy config: {_config_path}")
    except Exception as e:
        print(f"An error occurred while reading proxy config: {e}")
    return None   

# -----------------------------------------------------------------------------
# 当前预训练数据集的具体信息

# 数据托管和按需下载的互联网 URL
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
MAX_SHARD = 1822  # 最后一个数据分片是 shard_01822.parquet
proxies = get_proxy_dict() # 用于 requests 的代理字典

# 本地数据目录
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 这些函数是对其他模块有用的工具，可以/应该被导入
def index_to_filename(index): return f"shard_{index:05d}.parquet"  # 文件名格式

def list_parquet_files(data_dir=None):
    """ 查看数据目录并返回所有 parquet 文件的完整路径。 """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def parquets_iter_batched(split, start=0, step=1):
    """
    遍历数据集，以底层 row_groups 为批次以提高效率。
    - split 可以是 "train" 或 "val"。最后一个 parquet 文件将是 val。
    - start/step 用于在 DDP 中跳过行。例如 start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-
                                  1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------


def download_single_file(index):
    """ 下载单个文件索引，带退避重试 """

    # 构建此文件的本地路径，如果已存在则跳过
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # 构建此文件的远程 URL
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # 带重试的下载
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(
                url, stream=True, timeout=30, proxies=proxies)
            response.raise_for_status()
            # 先写入临时文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                # 1MB 块
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            # 将临时文件移动到最终位置
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(
                f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理任何部分文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 使用指数退避重试几次：2^attempt 秒
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(
                    f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False

if __name__ == "__main__":
    print(f"Using proxies: {proxies}")

    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1,
                        help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4,
                        help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == - \
        1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(
        f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(
        f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
