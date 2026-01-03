# analysis/aggregate_gcg_summaries.py
import os
import re
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import math


CFG_RE = re.compile(
    r"eps(?P<eps>[0-9.]+)_k(?P<k>\d+)_(?P<norm>l2|linf)_mix(?P<mix>[0-9.]+)_seed(?P<seed>\d+)"
)

@dataclass
class Record:
    path: str
    run_dir: str
    dataset: str
    split: str
    eps: float
    k: int
    norm: str
    mix: float
    seed: int
    asr_overall: float
    clean_acc: float


def parse_config_from_run_dir(run_dir: str) -> Tuple[float, int, str, float, int]:
    """
    从 run_dir 里解析 eps/k/norm/mix/seed
    例如: baseline/continuous_at/runs_harmless/eps0.05_k10_l2_mix0.5_seed27
    """
    m = CFG_RE.search(run_dir)
    if not m:
        raise ValueError(f"Cannot parse config from run_dir: {run_dir}")
    eps = float(m.group("eps"))
    k = int(m.group("k"))
    norm = m.group("norm")
    mix = float(m.group("mix"))
    seed = int(m.group("seed"))
    return eps, k, norm, mix, seed


def safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def load_one_summary(path: str) -> Record:
    with open(path, "r") as f:
        j = json.load(f)

    run_dir = j.get("run_dir", os.path.dirname(path))
    eps, k, norm, mix, seed = parse_config_from_run_dir(run_dir)

    dataset = j.get("dataset", "UNKNOWN_DATASET")
    split = j.get("split", "UNKNOWN_SPLIT")

    asr_overall = float(j["asr_overall"])
    clean_acc = float(j.get("clean_acc", float("nan")))

    return Record(
        path=path,
        run_dir=run_dir,
        dataset=dataset,
        split=split,
        eps=eps,
        k=k,
        norm=norm,
        mix=mix,
        seed=seed,
        asr_overall=asr_overall,
        clean_acc=clean_acc,
    )


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 0:
        return float("nan"), float("nan")
    mu = sum(xs) / len(xs)
    if len(xs) == 1:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)  # sample std
    return mu, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--roots",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Directories to scan, e.g. "
            "baseline/continuous_at/runs_harmless baseline/continuous_at/runs_imdb baseline/continuous_at/runs_helpful"
        ),
    )
    ap.add_argument(
        "--split_filter",
        type=str,
        default="attack",
        help="Only include summaries whose 'split' matches this string (default: attack).",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default="analysis/continuous_at_gcg_aggregate.json",
        help="Where to save aggregated results (json).",
    )
    args = ap.parse_args()

    # 1) collect all summaries
    summary_paths: List[str] = []
    for root in args.roots:
        summary_paths += glob.glob(os.path.join(root, "**", "gcg_eval_summary.json"), recursive=True)

    if not summary_paths:
        raise FileNotFoundError(f"No gcg_eval_summary.json found under roots: {args.roots}")

    # 2) load & group
    groups: Dict[Tuple[str, float, int, str, float], List[Record]] = {}
    skipped = 0

    for p in sorted(summary_paths):
        try:
            rec = load_one_summary(p)
        except Exception as e:
            print(f"[skip] {p} (load/parse error: {e})")
            skipped += 1
            continue

        if args.split_filter and rec.split != args.split_filter:
            continue

        key = (rec.dataset, rec.eps, rec.k, rec.norm, rec.mix)
        groups.setdefault(key, []).append(rec)

    # 3) aggregate
    out: Dict[str, Any] = {
        "roots": args.roots,
        "split_filter": args.split_filter,
        "n_found": len(summary_paths),
        "n_skipped": skipped,
        "groups": [],
    }

    # pretty print
    print("\n==== Aggregated GCG summary stats ====\n")
    for (dataset, eps, k, norm, mix), recs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])):
        asrs = [r.asr_overall for r in recs]
        accs = [r.clean_acc for r in recs if not math.isnan(r.clean_acc)]

        asr_mu, asr_sd = mean_std(asrs)
        acc_mu, acc_sd = mean_std(accs) if accs else (float("nan"), float("nan"))

        seeds = sorted([r.seed for r in recs])
        paths = [r.path for r in recs]

        group_obj = {
            "dataset": dataset,
            "config": {"eps": eps, "k": k, "norm": norm, "mix_adv_frac": mix},
            "n": len(recs),
            "seeds": seeds,
            "asr_overall": {"mean": asr_mu, "sd": asr_sd, "values": asrs},
            "clean_acc": {"mean": acc_mu, "sd": acc_sd, "values": accs},
            "paths": paths,
        }
        out["groups"].append(group_obj)

        print(f"[{dataset}] eps={eps} k={k} norm={norm} mix={mix}  (n={len(recs)} seeds={seeds})")
        print(f"  ASR_overall: mean={asr_mu:.4f}  sd={asr_sd:.4f}  values={['%.4f'%v for v in asrs]}")
        if accs:
            print(f"  clean_acc:   mean={acc_mu:.4f}  sd={acc_sd:.4f}  values={['%.4f'%v for v in accs]}")
        print("")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {args.out_json}")


if __name__ == "__main__":
    main()

"""
python baseline/stat_baseline.py \
  --roots baseline/continuous_at/runs_harmless baseline/continuous_at/runs_imdb baseline/continuous_at/runs_helpful \
  --split_filter attack \
  --out_json baseline/continuous_at/continuous_at_gcg_aggregate.json

"""