#!/bin/bash
# Script 1: AdvPool adversarial training + RandomToken eval (~3.5–4h on A100-80GB)
#
# 用法:
#   CUDA_VISIBLE_DEVICES=0 bash baseline/scaling/run_train_and_randomtoken_eval.sh \
#     Qwen/Qwen2.5-7B \
#     path/to/best_Harmless.pt \
#     AlignmentResearch/Harmless \
#     src/data/harmless_splits.json \
#     baseline/scaling/advpool_runs/qwen7b_harmless

set -e

MODEL_PATH=$1      # HF model id or local path, e.g. Qwen/Qwen2.5-7B
MODEL_PT=$2        # finetuned .pt checkpoint
DATASET=$3         # HF dataset id, e.g. AlignmentResearch/Harmless
SPLITS_JSON=$4     # path to splits JSON, e.g. src/data/harmless_splits.json
OUTPUT_DIR=$5      # output dir, e.g. baseline/scaling/advpool_runs/qwen7b_harmless

if [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: bash run_train_and_randomtoken_eval.sh MODEL_PATH MODEL_PT DATASET SPLITS_JSON OUTPUT_DIR"
  exit 1
fi

# ---- 训练 (R=5 rounds, ~2–2.5h) ----
python -m baseline.scaling.run_adv_pool_at \
  --base_model_path "$MODEL_PATH" \
  --model_pt "$MODEL_PT" \
  --dataset "$DATASET" \
  --splits_json "$SPLITS_JSON" \
  --output_dir "$OUTPUT_DIR" \
  --torch_dtype bfloat16 \
  --R 5 \
  --kstart 8 \
  --kend 32 \
  --n_new_adv 100 \
  --naug 500 \
  --n_candidates_per_it 32 \
  --beam_k 128 \
  --max_steps 100 \
  --lr 2e-5 \
  --batch_size 8 \
  --seed 42

# ---- RandomToken eval (逐 round, ~1–1.5h 合计) ----
for r in 001 002 003 004 005; do
  ROUND_DIR="$OUTPUT_DIR/round_$r"
  [ -d "$ROUND_DIR" ] || continue
  echo "=== RandomToken eval: $ROUND_DIR ==="
  python -m baseline.scaling.eval_all_attacks_scaling \
    --attack_type random \
    --round_dirs "$ROUND_DIR" \
    --dataset "$DATASET" \
    --splits_json "$SPLITS_JSON" \
    --max_eval 100 \
    --random_rounds 200 \
    --n_attack_tokens 10 \
    --attack_mode suffix \
    --out_root "$OUTPUT_DIR" \
    --seed 42
done

echo "Done: training + RandomToken eval -> $OUTPUT_DIR"
