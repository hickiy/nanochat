"""
评估给定模型的 CORE 指标。

在单个 GPU 上运行：
python -m scripts.base_eval

使用 torchrun 在例如 8 个 GPU 上运行：
torchrun --nproc_per_node=8 -m scripts.base_eval

脚本将在控制台打印 CORE 指标。
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanochat 特定函数，处理 I/O 等

# 评估 CORE 指标所需的约 162MB 数据
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # 这里 file_path 是 eval_bundle.zip 文件的路径
    # 我们需要解压它并放置在基础目录中
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1):
    """
    在 CORE 基准测试上评估基础模型。
    - max_per_task: 将数据截断到每个任务这么多样本，用于测试（-1 = 禁用）
    """
    # 加载配置和任务元数据
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # 将评估包下载到磁盘（如果需要则解压）
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # 从评估元数据加载随机基线值
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    # 评估每个任务
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # 加载此任务的数据
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # 打乱数据，因为在很多情况下它看起来是有序的，但我们希望
        # 能够只运行数据子集用于调试目的等。
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # 运行此任务的评估
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace 加载工具和模型的轻量级包装

class ModelWrapper:
    """HuggingFace 模型的轻量级包装"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # 加载模型
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # 加载分词器
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    args = parser.parse_args()

    # 分布式 / 精度设置
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # 从命令行或文件系统加载模型和分词器
    if args.hf_path is not None:
        # 目前假设如果给定路径，它是 huggingface 模型路径
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # 用于输出 csv 文件
    else:
        # 从文件系统加载本地模型
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})" # 仅用于日志记录
        model_slug = f"base_model_{meta['step']:06d}" # 用于输出 csv 文件

    # 评估模型
    with autocast_ctx:
        out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task)

    # 将结果写入 csv 文件
    core_metric = None
    centered_results = {}
    if ddp_rank == 0:
        base_dir = get_base_dir()
        output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        results = out["results"]
        centered_results = out["centered_results"]
        core_metric = out["core_metric"]
        with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
            f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
            for label in results:
                f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
            f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
        # 也将 csv 文件内容打印到控制台
        print0("="*80)
        print0(f"Model: {model_name}")
        print0("="*80)
        with open(output_csv_path, 'r', encoding='utf-8') as f:
            print0(f.read())

    # 记录到报告
    from nanochat.report import get_report
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": core_metric,
        },
        centered_results, # 完整表格
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
