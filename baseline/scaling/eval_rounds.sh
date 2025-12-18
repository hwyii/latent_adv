#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=1

# -------------------------
# paths
# -------------------------
DATASET="AlignmentResearch/Harmless"
SPLITS_JSON="src/data/harmless_splits.json"
BASE_DIR="baseline/scaling/advpool_runs/harmless_gcg_suffix"

OUT_ROOT="baseline/scaling/advpool_runs/harmless_gcg_suffix"
EVAL_SCRIPT="baseline.scaling.eval_gcg_scaling"

# -------------------------
# eval config (MUST match other baselines)
# -------------------------
ATTACK_MODE="suffix"
ROUNDS=20
BEAM_K=256
N_CANDIDATES=128
N_ATTACK_TOKENS=10
ATTACK_START=5

MAX_EVAL=100
SEED=42

# -------------------------
# collect round dirs
# -------------------------
ROUND_DIRS=()
for i in $(seq 1 8); do
  ROUND_DIRS+=("${BASE_DIR}/round_$(printf "%03d" $i)")
done

echo "Evaluating rounds:"
for d in "${ROUND_DIRS[@]}"; do
  echo "  - $d"
done

# -------------------------
# run eval
# -------------------------
python -m ${EVAL_SCRIPT} \
  --round_dirs "${ROUND_DIRS[@]}" \
  --dataset ${DATASET} \
  --splits_json ${SPLITS_JSON} \
  --split attack \
  --max_eval ${MAX_EVAL} \
  --seed ${SEED} \
  --out_root ${OUT_ROOT} \
  --attack_mode ${ATTACK_MODE} \
  --rounds ${ROUNDS} \
  --beam_k ${BEAM_K} \
  --n_candidates_per_it ${N_CANDIDATES} \
  --n_attack_tokens ${N_ATTACK_TOKENS} \
  --attack_start ${ATTACK_START} \
  --save_text

echo "==== All rounds 001–008 evaluated ===="
echo "Results saved under: ${OUT_ROOT}"
