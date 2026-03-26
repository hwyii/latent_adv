#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-0: Fine-tune baseline classifiers for all (model, dataset) combinations.

GPU scheduling: each job polls nvidia-smi and claims the GPU with the most
free memory that meets the model's requirement. No manual GPU assignment needed.

Output layout:
  outputs/stage0/{model_short}/{dataset}/
      best_{Dataset}.pt
      best_loss_{Dataset}.pt
      flop_stats_{Dataset}.json
      stdout.log  /  stderr.log

Summary:
  all_results/baseline_summary_{model_short}.json

Usage:
  python run_stage0.py                         # scan all GPUs automatically
  python run_stage0.py --gpus 0,1,4,5         # restrict to these GPU IDs
  python run_stage0.py --models gemma2-2b      # subset of models
  python run_stage0.py --datasets imdb,enron   # subset of datasets
  python run_stage0.py --dry-run              # print jobs without running
"""

import os, copy, sys, yaml, json, time, argparse, subprocess
import multiprocessing as mp
from src.training.trainer_baseline import train_baseline

# ─────────────────────────── MODEL / DATASET CONFIG ────────────────────────

# min_free_gb: minimum free VRAM this model needs before a job starts.
#
# Memory breakdown (params + AdamW m+v + grads + activations, seq=512):
#   7B models  (bfloat16): ~14 + 28 + 14 + 5 = ~61 GB → H100 80GB ✓  H200 141GB ✓  A6000 48GB ✗
#   Gemma-2-2b (float32) : ~10 + 21 + 10 + 2 = ~43 GB → H100 ✓  H200 ✓  A6000 48GB ✗
#     (Gemma-2 uses float32 to avoid bfloat16 attention mask overflow)
#
# Neither model fits on A6000 (48 GB). A6000 is not usable for any of these models.

MODELS = [
    {
        "name": "meta-llama/Meta-Llama-3-8B",
        "short": "llama3-8b",
        "min_free_gb": 62,   # H100 / H200 only
        "overrides": {"model": {"torch_dtype": "bfloat16"}, "data": {"train_bsz": 4}, "train": {"grad_accum_steps": 4}},
    },
    {
        "name": "Qwen/Qwen2.5-7B",
        "short": "qwen25-7b",
        "min_free_gb": 62,
        "overrides": {"model": {"torch_dtype": "bfloat16"}, "data": {"train_bsz": 4}, "train": {"grad_accum_steps": 4}},
    },
    {
        "name": "EleutherAI/pythia-6.9b",
        "short": "pythia-6.9b",
        "min_free_gb": 62,
        "overrides": {"model": {"torch_dtype": "bfloat16"}, "data": {"train_bsz": 4}, "train": {"grad_accum_steps": 4}},
    },
    {
        "name": "mistralai/Mistral-7B-v0.1",
        "short": "mistral-7b",
        "min_free_gb": 62,
        "overrides": {"model": {"torch_dtype": "bfloat16"}, "data": {"train_bsz": 4}, "train": {"grad_accum_steps": 4}},
    },
    {
        "name": "google/gemma-2-2b",
        "short": "gemma2-2b",
        "min_free_gb": 44,   # H100 / H200 only (float32: ~43 GB needed)
        # float32: Gemma-2 attention mask uses finfo(float32).min which overflows bfloat16
        "overrides": {"data": {"train_bsz": 8}, "train": {"grad_accum_steps": 2}},
    },
]

# Natural-language datasets: sequences 100–1000 tokens → max_length=512
# Short-text datasets (PasswordMatch / WordLength): sequences 5–50 tokens → max_length=64
#   (reduces wasted padding compute; doesn't affect min_free_gb since
#    the bottleneck is params+optimizer, not activations)

DATASETS = [
    {
        "tag": "imdb",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/IMDB",          "split_file": "src/data/imdb_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "enron",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/EnronSpam",     "split_file": "src/data/enron_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "harmless",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/Harmless",      "split_file": "src/data/harmless_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "helpful",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/Helpful",       "split_file": "src/data/helpful_splits.json"},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "passwordmatch",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/PasswordMatch", "split_file": "src/data/passwordmatch_splits.json",
                     "max_length": 64},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
    {
        "tag": "wordlength",
        "overrides": {
            "data": {"dataset": "AlignmentResearch/WordLength",    "split_file": "src/data/wordlength_splits.json",
                     "max_length": 64},
            "train": {"epochs": 3, "lr": 1e-5},
        },
    },
]

# ─────────────────────────── GPU SCHEDULER ─────────────────────────────────

def query_free_memory_mb(candidate_ids=None):
    """Return {gpu_id: free_mb} via nvidia-smi."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    info = {}
    for line in result.stdout.strip().split("\n"):
        parts = line.strip().split(", ")
        if len(parts) == 2:
            gid, free = int(parts[0]), int(parts[1])
            if candidate_ids is None or gid in candidate_ids:
                info[gid] = free
    return info


def acquire_gpu(min_free_mb, claimed: list, lock, candidate_ids=None, poll_sec=60, tag="job"):
    """
    Block until a GPU with >= min_free_mb free memory is available.
    Marks the GPU as claimed so concurrent workers don't race for it.
    Returns the physical GPU id.
    """
    while True:
        with lock:
            free = query_free_memory_mb(candidate_ids)
            # exclude GPUs already claimed by our own jobs
            eligible = [
                (gid, mb) for gid, mb in free.items()
                if mb >= min_free_mb and gid not in claimed
            ]
            if eligible:
                best_gpu = max(eligible, key=lambda x: x[1])[0]
                claimed.append(best_gpu)
                free_gb = free[best_gpu] / 1024
                print(f"[GPU] [{tag}] Acquired GPU {best_gpu} ({free_gb:.1f} GB free)", flush=True)
                return best_gpu

        # Pretty-print current state while waiting
        free = query_free_memory_mb(candidate_ids)
        status = ", ".join(f"GPU{g}={mb/1024:.0f}GB" for g, mb in sorted(free.items()))
        print(
            f"[GPU] [{tag}] Need {min_free_mb/1024:.0f} GB free. "
            f"Currently: {status} | claimed={claimed} | retry in {poll_sec}s",
            flush=True,
        )
        time.sleep(poll_sec)


def release_gpu(gpu_id, claimed: list, lock, tag="job"):
    with lock:
        if gpu_id in claimed:
            claimed.remove(gpu_id)
    print(f"[GPU] [{tag}] Released GPU {gpu_id}", flush=True)


# ─────────────────────────── HELPERS ───────────────────────────────────────

def deep_update(base: dict, upd: dict):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _worker(job: dict, base_cfg: dict, claimed, lock, result_queue: mp.Queue):
    """One (model, dataset) training job."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tag = job["tag"]
    out_dir = job["out_dir"]
    min_free_mb = job["min_free_gb"] * 1024
    candidate_ids = job.get("candidate_ids", None)

    os.makedirs(out_dir, exist_ok=True)

    poll_sec = job.get("poll_sec", 300)

    # Acquire a suitable GPU (blocks here if none available)
    gpu_id = acquire_gpu(min_free_mb, claimed, lock, candidate_ids, poll_sec=poll_sec, tag=tag)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cfg = copy.deepcopy(base_cfg)
    deep_update(cfg, job["overrides"])
    cfg["device"] = "cuda:0"
    cfg["out"] = {"dir": out_dir}

    print(f"--> [{tag}] Starting on GPU {gpu_id}. Logs → {out_dir}/stdout.log", flush=True)

    try:
        with open(os.path.join(out_dir, "stdout.log"), "w", buffering=1) as fout, \
             open(os.path.join(out_dir, "stderr.log"), "w", buffering=1) as ferr:
            sys.stdout = fout
            sys.stderr = ferr
            stats = train_baseline(cfg)
            stats["tag"] = tag
            stats["gpu_id"] = gpu_id
            result_queue.put(stats)
    except Exception as e:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        print(f"!!! [{tag}] Error on GPU {gpu_id}: {e}", flush=True)
        result_queue.put({"tag": tag, "gpu_id": gpu_id, "error": repr(e)})
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        release_gpu(gpu_id, claimed, lock, tag=tag)
        print(f"<-- [{tag}] Done.", flush=True)


# ─────────────────────────── MAIN ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--gpus", default=None,
        help="Restrict to these physical GPU IDs, e.g. '0,1,4,5'. "
             "Default: all GPUs visible to nvidia-smi.",
    )
    p.add_argument(
        "--models", default=None,
        help="Comma-separated model shorts, e.g. 'llama3-8b,gemma2-2b'. Default: all.",
    )
    p.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset tags, e.g. 'imdb,enron'. Default: all.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print all jobs and GPU availability without running anything.",
    )
    p.add_argument(
        "--poll", type=int, default=300,
        help="Seconds between GPU availability checks when waiting. Default: 300.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    candidate_ids = (
        [int(g) for g in args.gpus.split(",")] if args.gpus else None
    )
    model_filter   = set(args.models.split(","))   if args.models   else None
    dataset_filter = set(args.datasets.split(",")) if args.datasets else None

    active_models   = [m for m in MODELS   if model_filter   is None or m["short"]  in model_filter]
    active_datasets = [d for d in DATASETS if dataset_filter is None or d["tag"]    in dataset_filter]

    # Show current GPU state
    free_now = query_free_memory_mb(candidate_ids)
    print("=" * 60)
    print("Current GPU free memory:")
    for gid, mb in sorted(free_now.items()):
        print(f"  GPU {gid}: {mb/1024:.1f} GB free")
    print("=" * 60)

    # Build job list
    jobs = []
    for model in active_models:
        for ds in active_datasets:
            tag = f"{model['short']}/{ds['tag']}"
            out_dir = os.path.join("outputs", "stage0", model["short"], ds["tag"])
            overrides = {}
            deep_update(overrides, model["overrides"])
            deep_update(overrides, ds["overrides"])
            overrides.setdefault("model", {})["name"] = model["name"]
            jobs.append({
                "tag": tag,
                "out_dir": out_dir,
                "min_free_gb": model["min_free_gb"],
                "candidate_ids": candidate_ids,
                "overrides": overrides,
                "poll_sec": args.poll,
            })

    print(f"Total jobs: {len(jobs)}")
    for j in jobs:
        print(f"  {j['tag']:35s}  needs {j['min_free_gb']} GB  →  {j['out_dir']}")
    print("=" * 60)

    if args.dry_run:
        print("[dry-run] Exiting without launching.")
        return

    with open("configs/baseline.yaml") as f:
        base_cfg = yaml.safe_load(f)

    manager = mp.Manager()
    claimed      = manager.list()   # GPU IDs currently held by our jobs
    lock         = manager.Lock()
    result_queue = manager.Queue()

    procs = []
    for job in jobs:
        p = mp.Process(target=_worker, args=(job, base_cfg, claimed, lock, result_queue))
        p.start()
        procs.append(p)
        # Small stagger so workers don't all slam nvidia-smi at exactly t=0
        time.sleep(2)

    all_results = [result_queue.get() for _ in jobs]
    for p in procs:
        p.join()

    # Save per-model summaries
    save_dir = "all_results"
    os.makedirs(save_dir, exist_ok=True)
    for model in active_models:
        model_results = [r for r in all_results if r.get("tag", "").startswith(model["short"] + "/")]
        out_path = os.path.join(save_dir, f"baseline_summary_{model['short']}.json")
        with open(out_path, "w") as f:
            json.dump(model_results, f, indent=2)
        print(f"Summary → {out_path}")

    errors = [r for r in all_results if "error" in r]
    print(f"All jobs finished. {len(all_results) - len(errors)}/{len(all_results)} succeeded.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
