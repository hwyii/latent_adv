#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-job runner for SLURM/HPCC.

On SLURM, each job is allocated a specific GPU by the scheduler — no need for
the nvidia-smi polling loop in run_stage0.py. This script runs exactly one
(model, dataset) training job using CUDA_VISIBLE_DEVICES=0 (the first GPU
in the allocation).

Usage:
  python run_one_job.py --model gemma2-2b --dataset enron
  python run_one_job.py --model llama3-8b --dataset imdb

The MODELS / DATASETS tables here must stay in sync with run_stage0.py.
Output:  outputs/stage0/{model_short}/{dataset}/
Summary: all_results/baseline_summary_{model_short}.json  (appended per job)
"""

import os, sys, copy, yaml, json, argparse
from src.training.trainer_baseline import train_baseline

# ── Model / dataset tables (mirror of run_stage0.py) ─────────────────────────

MODELS = [
    {
        "name": "meta-llama/Meta-Llama-3-8B",
        "short": "llama3-8b",
        "overrides": {
            "model": {"torch_dtype": "bfloat16"},
            "data":  {"train_bsz": 4},
            "train": {"grad_accum_steps": 4},
        },
    },
    {
        "name": "Qwen/Qwen2.5-7B",
        "short": "qwen25-7b",
        "overrides": {
            "model": {"torch_dtype": "bfloat16"},
            "data":  {"train_bsz": 4},
            "train": {"grad_accum_steps": 4},
        },
    },
    {
        "name": "EleutherAI/pythia-6.9b",
        "short": "pythia-6.9b",
        "overrides": {
            "model": {"torch_dtype": "bfloat16"},
            "data":  {"train_bsz": 4},
            "train": {"grad_accum_steps": 4},
        },
    },
    {
        "name": "mistralai/Mistral-7B-v0.1",
        "short": "mistral-7b",
        "overrides": {
            "model": {"torch_dtype": "bfloat16"},
            "data":  {"train_bsz": 4},
            "train": {"grad_accum_steps": 4},
        },
    },
    {
        "name": "google/gemma-2-2b",
        "short": "gemma2-2b",
        # float32: Gemma-2 attention mask overflows bfloat16
        "overrides": {
            "data":  {"train_bsz": 8},
            "train": {"grad_accum_steps": 2},
        },
    },
]

DATASETS = [
    {
        "tag": "imdb",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/IMDB",
                      "split_file": "src/data/imdb_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "enron",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/EnronSpam",
                      "split_file": "src/data/enron_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "harmless",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/Harmless",
                      "split_file": "src/data/harmless_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "helpful",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/Helpful",
                      "split_file": "src/data/helpful_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "passwordmatch",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/PasswordMatch",
                      "split_file": "src/data/passwordmatch_splits.json",
                      "max_length": 64},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "wordlength",
        "overrides": {
            "data":  {"dataset": "AlignmentResearch/WordLength",
                      "split_file": "src/data/wordlength_splits.json",
                      "max_length": 64},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True,
                   help="Model short name, e.g. 'llama3-8b', 'gemma2-2b'.")
    p.add_argument("--dataset", required=True,
                   help="Dataset tag, e.g. 'imdb', 'enron', 'harmless'.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = parse_args()

    model_cfg = next((m for m in MODELS if m["short"] == args.model), None)
    ds_cfg    = next((d for d in DATASETS if d["tag"] == args.dataset), None)

    if model_cfg is None:
        sys.exit(f"Unknown model short name: '{args.model}'. "
                 f"Valid: {[m['short'] for m in MODELS]}")
    if ds_cfg is None:
        sys.exit(f"Unknown dataset tag: '{args.dataset}'. "
                 f"Valid: {[d['tag'] for d in DATASETS]}")

    # Build config
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    overrides = copy.deepcopy(model_cfg["overrides"])
    deep_update(overrides, ds_cfg["overrides"])
    overrides.setdefault("model", {})["name"] = model_cfg["name"]
    deep_update(cfg, overrides)

    out_dir = os.path.join("outputs", "stage0", model_cfg["short"], ds_cfg["tag"])
    cfg["device"] = "cuda:0"   # SLURM gives us GPU 0 in the allocation
    cfg["out"]    = {"dir": out_dir}
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{model_cfg['short']}/{ds_cfg['tag']}"
    print(f"=== [{tag}]  out → {out_dir} ===", flush=True)

    stats = train_baseline(cfg)
    stats["tag"] = tag

    # Append to per-model summary (safe even if multiple jobs finish at once
    # because each model's file is written by only one job at a time in practice)
    save_dir = "all_results"
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, f"baseline_summary_{model_cfg['short']}.json")

    existing = []
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(stats)
    with open(summary_path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"=== [{tag}]  Done. Summary → {summary_path} ===", flush=True)


if __name__ == "__main__":
    main()
