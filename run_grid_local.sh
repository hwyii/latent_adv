#!/bin/bash
set -u

CONFIG="configs/train_reft.yaml"
GRID="param_grid_half.tsv"
NUM_TASKS=15          # 你的 tsv 行数
NUM_GPUS=4            # 机器上 GPU 数量

LOG_ROOT="logs/reft_latent_adv"
mkdir -p "${LOG_ROOT}"

IDX=1

while [ ${IDX} -le ${NUM_TASKS} ]; do
  echo "=== New wave starting at task ${IDX} ==="

  # 每一“波”最多起 NUM_GPUS 个任务
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    if [ ${IDX} -gt ${NUM_TASKS} ]; then
      break
    fi

    RUN_TAG="idx${IDX}_gpu${GPU_ID}"
    LOG_FILE="${LOG_ROOT}/${RUN_TAG}.log"

    echo "[Launch] task ${IDX} on GPU ${GPU_ID}, log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    python -m src.training.train_reft_adv \
      --config "${CONFIG}" \
      --param_grid "${GRID}" \
      --array_idx "${IDX}" \
      >"${LOG_FILE}" 2>&1 &

    IDX=$((IDX + 1))
  done

  # 等这一波所有 GPU 上的任务跑完，再起下一波
  wait
done

echo "All ${NUM_TASKS} runs finished."
