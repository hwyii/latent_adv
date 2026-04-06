#!/bin/bash
# Script 2: GCG eval + BEAST eval on all round checkpoints (~3.5h on A100-80GB)
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash baseline/scaling/run_gcg_beast_eval.sh \
#     AlignmentResearch/Harmless \
#     src/data/harmless_splits.json \
#     baseline/scaling/advpool_runs/qwen7b_harmless \
#     Qwen/Qwen2.5-0.5B        # optional: BEAST base model

set -e

DATASET=$1                                  # HF dataset id
SPLITS_JSON=$2                              # splits JSON path
OUTPUT_DIR=$3                               # same OUTPUT_DIR used during training
BEAST_BASE_MODEL=${4:-"EleutherAI/pythia-14m"}   # small causal LM for BEAST

if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: bash run_gcg_beast_eval.sh DATASET SPLITS_JSON OUTPUT_DIR [BEAST_BASE_MODEL]"
  exit 1
fi

# Collect existing round dirs
ROUND_DIRS=()
for r in 001 002 003 004 005; do
  d="$OUTPUT_DIR/round_$r"
  [ -d "$d" ] && ROUND_DIRS+=("$d")
done

if [ ${#ROUND_DIRS[@]} -eq 0 ]; then
  echo "No round_* dirs found under $OUTPUT_DIR. Run Script 1 first."
  exit 1
fi

echo "=== GCG eval on ${#ROUND_DIRS[@]} rounds (~2h) ==="
python -m baseline.scaling.eval_all_attacks_scaling \
  --attack_type gcg \
  --round_dirs "${ROUND_DIRS[@]}" \
  --dataset "$DATASET" \
  --splits_json "$SPLITS_JSON" \
  --max_eval 100 \
  --rounds 20 \
  --n_candidates_per_it 32 \
  --beam_k 128 \
  --n_attack_tokens 10 \
  --attack_mode suffix \
  --out_root "$OUTPUT_DIR" \
  --seed 42

echo "=== BEAST eval on ${#ROUND_DIRS[@]} rounds (~1.5h) ==="
python -m baseline.scaling.eval_all_attacks_scaling \
  --attack_type beast \
  --beast_base_model "$BEAST_BASE_MODEL" \
  --beast_n_tokens 25 \
  --beast_beam_size 7 \
  --round_dirs "${ROUND_DIRS[@]}" \
  --dataset "$DATASET" \
  --splits_json "$SPLITS_JSON" \
  --max_eval 100 \
  --out_root "$OUTPUT_DIR" \
  --seed 42

echo "Done: GCG + BEAST eval -> $OUTPUT_DIR"
