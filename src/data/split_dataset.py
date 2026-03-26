"""
Preprocess AlignmentResearch classification datasets into train/val splits
with token-length filtering.

Pipeline per dataset:
  HF train split  → length filter [MIN_TOKENS, MAX_TOKENS] → random sample MAX_TRAIN → ft_train
  HF val   split  → length filter [MIN_TOKENS, MAX_TOKENS] → keep all               → val

NOTE: `attack` key is set equal to `val` for backward-compat with eval_token_gcg.py,
      but those eval scripts must load the HF "validation" split (not "train") when
      using these indices.

Usage:
    python -m src.data.split_dataset                    # all 6 datasets
    python -m src.data.split_dataset --dataset Helpful  # single dataset
    python -m src.data.split_dataset --model_name gpt2  # custom tokenizer
"""

import argparse, json, os, random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_MIN = 100    # natural-language datasets: keep 100–1000 tokens
DEFAULT_MAX = 1000

# (hf_path, out_path, min_tok_override, max_tok_override)
# None → use DEFAULT_MIN / DEFAULT_MAX
DATASETS = {
    "Helpful":       ("AlignmentResearch/Helpful",       "src/data/helpful_splits.json",       None, None),
    "Harmless":      ("AlignmentResearch/Harmless",      "src/data/harmless_splits.json",       None, None),
    "IMDB":          ("AlignmentResearch/IMDB",          "src/data/imdb_splits.json",           None, None),
    "EnronSpam":     ("AlignmentResearch/EnronSpam",     "src/data/enron_splits.json",          None, None),
    # Algorithmically-generated short-text tasks: all samples are 12–22 tokens
    "PasswordMatch": ("AlignmentResearch/PasswordMatch", "src/data/passwordmatch_splits.json",  5,    50  ),
    "WordLength":    ("AlignmentResearch/WordLength",    "src/data/wordlength_splits.json",     5,    50  ),
}
MAX_TRAIN   = 20_000   # max ft_train samples (from HF train)
MAX_ATTACK  = 500      # max attack samples   (from remaining HF train, disjoint from ft_train)
SEED        = 42
INPUT_FIELD = "content"
LABEL_FIELD = "clf_label"
BATCH_SIZE  = 512


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_str(val) -> str:
    """Flatten list-type content fields (e.g. Helpful stores two convs as a list)."""
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x is not None)
    return str(val) if val is not None else ""


def compute_lengths(ds, tokenizer) -> list[int]:
    """Return token length for every row in ds (in order), without truncation."""
    lengths = []
    for start in range(0, len(ds), BATCH_SIZE):
        batch = ds[start : start + BATCH_SIZE]
        texts = [_to_str(v) for v in batch[INPUT_FIELD]]
        enc = tokenizer(texts, truncation=False, padding=False, add_special_tokens=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def filter_by_length(lengths: list[int], min_tok: int, max_tok: int) -> list[int]:
    return [i for i, L in enumerate(lengths) if min_tok <= L <= max_tok]


def label_dist(ds, indices: list[int]) -> dict:
    c: dict[str, int] = {}
    for i in indices:
        y = str(ds[int(i)][LABEL_FIELD])
        c[y] = c.get(y, 0) + 1
    return dict(sorted(c.items()))


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "N/A"


# ── Core ─────────────────────────────────────────────────────────────────────

def process_dataset(name: str, hf_path: str, out_path: str, tokenizer,
                    min_tok: int, max_tok: int) -> dict:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  Dataset : {name}  ({hf_path})")
    print(sep)

    ds_train = load_dataset(hf_path, split="train")
    ds_val   = load_dataset(hf_path, split="validation")
    N_tr, N_va = len(ds_train), len(ds_val)
    print(f"  Loaded   train={N_tr:,}   validation={N_va:,}")

    # ── Length filter ──
    print(f"  Tokenizing train ({N_tr:,} rows)...")
    train_lens = compute_lengths(ds_train, tokenizer)

    print(f"  Tokenizing validation ({N_va:,} rows)...")
    val_lens = compute_lengths(ds_val, tokenizer)

    train_keep = filter_by_length(train_lens, min_tok, max_tok)
    val_keep   = filter_by_length(val_lens,   min_tok, max_tok)

    # ── Split filtered train indices into ft_train + attack (disjoint) ──
    rng = random.Random(SEED)
    shuffled = list(train_keep)
    rng.shuffle(shuffled)

    ft_train = sorted(shuffled[:MAX_TRAIN])
    attack_pool = shuffled[MAX_TRAIN:]          # everything not in ft_train
    if len(attack_pool) > MAX_ATTACK:
        attack = sorted(rng.sample(attack_pool, MAX_ATTACK))
    else:
        attack = sorted(attack_pool)

    val_indices = sorted(val_keep)

    # ── Print stats ──
    print(f"\n  [Train split  →  ft_train + attack, source: HF 'train']")
    print(f"    Total rows          : {N_tr:>8,}")
    print(f"    After length filter : {len(train_keep):>8,}  ({pct(len(train_keep), N_tr):>6})  "
          f"[{min_tok}–{max_tok} tokens]")
    print(f"    ft_train            : {len(ft_train):>8,}  (first {MAX_TRAIN} after shuffle, seed={SEED})")
    print(f"    attack              : {len(attack):>8,}  (from remaining {len(attack_pool)}, "
          f"capped at {MAX_ATTACK})")
    print(f"    overlap check       : {'PASS — 0 overlap' if not set(ft_train) & set(attack) else 'FAIL'}")
    print(f"    ft_train label dist : {label_dist(ds_train, ft_train)}")
    print(f"    attack   label dist : {label_dist(ds_train, attack)}")

    print(f"\n  [Validation split  →  val, source: HF 'validation']")
    print(f"    Total rows          : {N_va:>8,}")
    print(f"    After length filter : {len(val_indices):>8,}  ({pct(len(val_indices), N_va):>6})  "
          f"[{min_tok}–{max_tok} tokens]")
    print(f"    val label dist      : {label_dist(ds_val, val_indices)}")

    # ── Save ──
    out = {
        # Indices into HF split="train"
        "ft_train": ft_train,
        "attack":   attack,
        # Indices into HF split="validation"
        "val":      val_indices,
        "seed":     SEED,
        "stats": {
            "train_total":        N_tr,
            "train_after_filter": len(train_keep),
            "ft_train_size":      len(ft_train),
            "attack_pool_size":   len(attack_pool),
            "attack_size":        len(attack),
            "val_total":          N_va,
            "val_after_filter":   len(val_indices),
            "min_tokens":         min_tok,
            "max_tokens":         max_tok,
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"\n  Saved → {out_path}")
    return out["stats"]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Build length-filtered train/val splits")
    ap.add_argument("--dataset", "-d",
                    choices=list(DATASETS.keys()) + ["all"],
                    default="all",
                    help="Dataset to process (default: all)")
    ap.add_argument("--model_name", default="gpt2",
                    help="Tokenizer for length computation (default: gpt2)")
    args = ap.parse_args()

    print(f"Tokenizer : {args.model_name}")
    print(f"Length    : [{DEFAULT_MIN}, {DEFAULT_MAX}] tokens (default; short-text datasets use per-dataset bounds)")
    print(f"Max train : {MAX_TRAIN:,}  |  Max attack : {MAX_ATTACK}  (seed={SEED})")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    targets = (DATASETS if args.dataset == "all"
               else {args.dataset: DATASETS[args.dataset]})

    all_stats: dict[str, dict] = {}
    for name, (hf_path, out_path, min_ovr, max_ovr) in targets.items():
        min_tok = min_ovr if min_ovr is not None else DEFAULT_MIN
        max_tok = max_ovr if max_ovr is not None else DEFAULT_MAX
        all_stats[name] = process_dataset(name, hf_path, out_path, tokenizer, min_tok, max_tok)

    # ── Final summary table ──
    print(f"\n{'='*62}")
    print(f"  SUMMARY")
    print(f"{'='*62}")
    hdr = f"  {'Dataset':<14}  {'Train: total→filtered→sampled':>32}   {'Val: total→filtered':>22}"
    print(hdr)
    print(f"  {'-'*60}")
    for name, s in all_stats.items():
        tr = f"{s['train_total']:,} → {s['train_after_filter']:,} → {s['ft_train_size']:,}"
        atk = f"{s['attack_size']:,}"
        va = f"{s['val_total']:,} → {s['val_after_filter']:,}"
        print(f"  {name:<14}  {tr:>32}   atk={atk:<6}  val={va}")
    print()


if __name__ == "__main__":
    main()

"""
information:
==============================================================
  Dataset : Helpful  (AlignmentResearch/Helpful)
==============================================================
  Loaded   train=41,815   validation=2,243
  Tokenizing train (41,815 rows)...
  Tokenizing validation (2,243 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   41,815
    After length filter :   39,701  ( 94.9%)  [100–1000 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 19701, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 10012, '1': 9988}
    attack   label dist : {'0': 246, '1': 254}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :    2,243
    After length filter :    2,117  ( 94.4%)  [100–1000 tokens]
    val label dist      : {'0': 1056, '1': 1061}

  Saved → src/data/helpful_splits.json

==============================================================
  Dataset : Harmless  (AlignmentResearch/Harmless)
==============================================================
  Loaded   train=41,087   validation=2,217
  Tokenizing train (41,087 rows)...
  Tokenizing validation (2,217 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   41,087
    After length filter :   37,595  ( 91.5%)  [100–1000 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 17595, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 10092, '1': 9908}
    attack   label dist : {'0': 255, '1': 245}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :    2,217
    After length filter :    2,035  ( 91.8%)  [100–1000 tokens]
    val label dist      : {'0': 1022, '1': 1013}

  Saved → src/data/harmless_splits.json

==============================================================
  Dataset : IMDB  (AlignmentResearch/IMDB)
==============================================================
  Loaded   train=24,365   validation=24,401
  Tokenizing train (24,365 rows)...
  Tokenizing validation (24,401 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   24,365
    After length filter :   22,446  ( 92.1%)  [100–1000 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 2446, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 10142, '1': 9858}
    attack   label dist : {'0': 259, '1': 241}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :   24,401
    After length filter :   22,327  ( 91.5%)  [100–1000 tokens]
    val label dist      : {'0': 11293, '1': 11034}

  Saved → src/data/imdb_splits.json

==============================================================
  Dataset : EnronSpam  (AlignmentResearch/EnronSpam)
==============================================================
  Loaded   train=29,290   validation=1,852
  Tokenizing train (29,290 rows)...
  Tokenizing validation (1,852 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   29,290
    After length filter :   20,733  ( 70.8%)  [100–1000 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 733, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 9752, '1': 10248}
    attack   label dist : {'0': 243, '1': 257}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :    1,852
    After length filter :    1,289  ( 69.6%)  [100–1000 tokens]
    val label dist      : {'0': 626, '1': 663}

  Saved → src/data/enron_splits.json

==============================================================
  Dataset : PasswordMatch  (AlignmentResearch/PasswordMatch)
==============================================================
  Loaded   train=25,000   validation=25,000
  Tokenizing train (25,000 rows)...
  Tokenizing validation (25,000 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   25,000
    After length filter :   25,000  (100.0%)  [5–50 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 5000, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 10011, '1': 9989}
    attack   label dist : {'0': 269, '1': 231}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :   25,000
    After length filter :   25,000  (100.0%)  [5–50 tokens]
    val label dist      : {'0': 12500, '1': 12500}

  Saved → src/data/passwordmatch_splits.json

==============================================================
  Dataset : WordLength  (AlignmentResearch/WordLength)
==============================================================
  Loaded   train=25,000   validation=25,000
  Tokenizing train (25,000 rows)...
  Tokenizing validation (25,000 rows)...

  [Train split  →  ft_train + attack, source: HF 'train']
    Total rows          :   25,000
    After length filter :   25,000  (100.0%)  [5–50 tokens]
    ft_train            :   20,000  (first 20000 after shuffle, seed=42)
    attack              :      500  (from remaining 5000, capped at 500)
    overlap check       : PASS — 0 overlap
    ft_train label dist : {'0': 11292, '1': 8708}
    attack   label dist : {'0': 294, '1': 206}

  [Validation split  →  val, source: HF 'validation']
    Total rows          :   25,000
    After length filter :   25,000  (100.0%)  [5–50 tokens]
    val label dist      : {'0': 13981, '1': 11019}

  Saved → src/data/wordlength_splits.json

==============================================================
  SUMMARY
==============================================================
  Dataset            Train: total→filtered→sampled      Val: total→filtered
  ------------------------------------------------------------
  Helpful                 41,815 → 39,701 → 20,000   atk=500     val=2,243 → 2,117
  Harmless                41,087 → 37,595 → 20,000   atk=500     val=2,217 → 2,035
  IMDB                    24,365 → 22,446 → 20,000   atk=500     val=24,401 → 22,327
  EnronSpam               29,290 → 20,733 → 20,000   atk=500     val=1,852 → 1,289
  PasswordMatch           25,000 → 25,000 → 20,000   atk=500     val=25,000 → 25,000
  WordLength              25,000 → 25,000 → 20,000   atk=500     val=25,000 → 25,000
"""