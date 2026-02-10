#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, copy, sys, yaml, json, multiprocessing as mp
from src.training.trainer_baseline import train_baseline

# 每个 job 指定它要用的 GPU 物理编号（如 5/6/7/3）
DATASET_JOBS = [
    {
        "tag": "imdb", "cuda_id": 0,
        "overrides": {
            "data": {"dataset": "AlignmentResearch/IMDB", "split_file": "src/data/imdb_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
            "out": {"dir": "outputs/imdb"},
        }
    },
    {
        "tag": "enron", "cuda_id": 1,
        "overrides": {
            "data": {"dataset": "AlignmentResearch/EnronSpam", "split_file": "src/data/enron_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
            "out": {"dir": "outputs/enron"},
        }
    },
    {
        "tag": "harmless", "cuda_id": 0,
        "overrides": {
            "data": {"dataset": "AlignmentResearch/Harmless", "split_file": "src/data/harmless_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
            "out": {"dir": "outputs/harmless"},
        }
    },
    {
        "tag": "helpful", "cuda_id": 1,
        "overrides": {
            "data": {"dataset": "AlignmentResearch/Helpful", "split_file": "src/data/helpful_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
            "out": {"dir": "outputs/helpful"},
        }
    },
]

def deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def _worker(job, base_cfg, q):
    # 0) 关键：防止 transformers 找不到模型，强制走镜像（国内服务器必加）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    # 1) 只暴露一张 GPU 给该进程
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job["cuda_id"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 2) 合并配置
    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, job["overrides"])
    cfg["device"] = "cuda:0"

    # 3) 准备目录
    out_dir = cfg["out"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    # 4) 日志处理：先在终端打印，再重定向
    print(f"--> [Job {job['tag']}] Started. Logging to {out_dir}/stdout.log")
    
    # 记录原始 stdout 以便出错时能看到
    original_stdout = sys.stdout 
    
    try:
        # 重定向后的输出将进入文件
        with open(os.path.join(out_dir, "stdout.log"), "w", buffering=1) as f_out, \
             open(os.path.join(out_dir, "stderr.log"), "w", buffering=1) as f_err:
            sys.stdout = f_out
            sys.stderr = f_err
            
            stats = train_baseline(cfg)
            stats["tag"] = job["tag"]
            q.put(stats)
    except Exception as e:
        # 恢复 stdout 打印错误
        sys.stdout = original_stdout
        print(f"!!! [Job {job['tag']}] Error: {e}")
        q.put({"tag": job["tag"], "error": repr(e)})

def main():
    # 运行前检查：如果当前目录下有叫 'gpt2' 的文件夹，先删掉它或改名
    if os.path.isdir("gpt2") and not os.path.exists("gpt2/baseline_summary.json"):
        import shutil
        print("Warning: Found directory named 'gpt2', renaming to 'gpt2_old' to avoid conflict.")
        if os.path.exists("gpt2_old"): shutil.rmtree("gpt2_old")
        os.rename("gpt2", "gpt2_old")

    with open("configs/baseline.yaml") as f:
        base_cfg = yaml.safe_load(f)

    q = mp.Queue()
    procs = []
    for job in DATASET_JOBS:
        p = mp.Process(target=_worker, args=(job, base_cfg, q))
        p.start()
        procs.append(p)

    summary = []
    for _ in DATASET_JOBS:
        summary.append(q.get())

    for p in procs:
        p.join()

    # 汇总文件改名，不要放在名为 'gpt2' 的文件夹里
    save_dir = "all_results"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/baseline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"All jobs finished. Summary saved to {save_dir}/")

if __name__ == "__main__":
    main()
