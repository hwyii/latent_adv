#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3

BASE_MODEL="EleutherAI/pythia-410m"
MODEL_PT="out/pythia410m/imdb/best_IMDB.pt"
SPLITS_JSON="src/data/imdb_splits.json"
OUT_DIR="baseline/continuous_at/runs_imdb"
DATASET="AlignmentResearch/IMDB"

SEEDS=(42 27 56)

for SEED in "${SEEDS[@]}"; do
  echo "==== Running seed ${SEED} ===="
  python -m baseline.continuous_at.run_continuous_at \
    --base_model_path ${BASE_MODEL} \
    --model_pt ${MODEL_PT} \
    --dataset ${DATASET} \
    --splits_json ${SPLITS_JSON} \
    --output_dir ${OUT_DIR} \
    --seed ${SEED} \
    --mix_adv_frac 0.5 \
    --batch_size 8 \
    --max_steps 1500
done

echo "==== All runs finished ===="
