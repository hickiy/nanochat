"""
将 FinewebEdu-100B 数据集重新打包为分片：

- 每个分片压缩后约 100MB 大小（zstd 压缩后）
- parquet 文件以 1000 行为行组大小写入
- 打乱数据集

这将上传到 HuggingFace 进行托管。
重要的是，我们的 DataLoader 将能够流式读取数据并
沿途缓存到磁盘，减少训练延迟。

注意：此文件仅作为数据集准备的参考/文档，
在项目运行时不会使用。
"""
import os
import time

from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

# 源数据集
dataset_kwargs = {
    "path": "HuggingFaceFW/fineweb-edu",
    "split": "train",
    "name": "sample-100BT", # ~100B GPT-2 token，约每 3 个字符一个 token => 总共约 ~300B 字符
}
ds = load_dataset(**dataset_kwargs)

# 打乱以打乱顺序
ds = ds.shuffle(seed=42)
ndocs = len(ds) # total number of documents to process
print(f"Total number of documents: {ndocs}")

# 重新打包为 parquet 文件
output_dir = "/home/ubuntu/.cache/nanochat/base_data"
os.makedirs(output_dir, exist_ok=True)

# 写入 parquet 文件
chars_per_shard = 250_000_000
row_group_size = 1024 # HF 使用 1000 但我们使用 2 的倍数，后续分布式数据加载器更好用
shard_docs = []
shard_index = 0
shard_characters = 0
total_docs_processed = 0
total_time_spent = 0
t0 = time.time()
for doc in ds:
    text = doc['text']
    shard_docs.append(text)
    shard_characters += len(text)
    collected_enough_chars = shard_characters >= chars_per_shard
    docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0
    if collected_enough_chars and docs_multiple_of_row_group_size: # 导致 ~100MB 的文本（压缩后）
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False, # this is usually used for categorical data
            compression="zstd", # Valid values: {‘NONE’, ‘SNAPPY’, ‘GZIP’, ‘BROTLI’, ‘LZ4’, ‘ZSTD’}
            compression_level=3,
            write_statistics=False, # not needed for text
        )
        t1 = time.time()
        dt = t1 - t0 # for this shard alone
        t0 = t1
        total_docs_processed += len(shard_docs)
        total_time_spent += dt
        remaining_docs = ndocs - total_docs_processed
        avg_time_per_doc = total_time_spent / total_docs_processed
        remaining_time = remaining_docs * avg_time_per_doc
        remaining_time_hours = remaining_time / 3600
        print(f"Wrote {shard_path}. #documents: {len(shard_docs)} | #characters: {shard_characters} | time: {dt:.2f}s | remaining time: {remaining_time_hours:.2f}h")
        shard_docs = []
        shard_characters = 0
        shard_index += 1

# 演示后续如何将数据上传到 HuggingFace
def upload():
    import os
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id="karpathy/fineweb-edu-100b-shuffle",
        repo_type="dataset",
    )
# upload()
