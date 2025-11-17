#!/usr/bin/env bash
set -e

# 你的训练输出根目录
RUN_ROOT="out/pythia410m/harmless/reft_latent_adv"
BASELINE_CKPT="out/pythia410m/harmless/best_Harmless.pt"
LOG_DIR="logs/gcg_eval"
mkdir -p "${LOG_DIR}"

# 可用 GPU 列表：按需改
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

# 1) 收集所有 run_dir（含 final_intervention 的目录的上一级）
RUNS=()
while IFS= read -r final_dir; do
  RUN_DIR=$(dirname "${final_dir}")
  RUNS+=("${RUN_DIR}")
done < <(find "${RUN_ROOT}" -type d -name "final_intervention" | sort)

NUM_RUNS=${#RUNS[@]}
echo "[Eval-GCG] 共发现 ${NUM_RUNS} 个 run 需要评估"

if [[ ${NUM_RUNS} -eq 0 ]]; then
  echo "[Eval-GCG] 没有找到任何 final_intervention，退出"
  exit 0
fi

# 2) 定义一个函数：在指定 GPU 上串行跑一组 RUN_DIR
run_on_gpu() {
  local GPU_ID="$1"
  shift
  local RUN_LIST=("$@")

  echo "[Worker][GPU ${GPU_ID}] 接到 ${#RUN_LIST[@]} 个任务"

  for RUN_DIR in "${RUN_LIST[@]}"; do
    local BASE_NAME
    BASE_NAME=$(basename "${RUN_DIR}")

    # 从 BASE_NAME 解析 layer_idx，例如 seed42_Rrandom_L16_A11_r4_atk...
    local LAYER_IDX
    LAYER_IDX=$(echo "${BASE_NAME}" | sed -n 's/.*_L\([0-9]\+\)_A.*/\1/p')

    if [[ -z "${LAYER_IDX}" ]]; then
      echo "[Worker][GPU ${GPU_ID}] [WARN] 无法从 ${BASE_NAME} 解析 layer_idx，跳过"
      continue
    fi

    local LOG_FILE="${LOG_DIR}/${BASE_NAME}_gpu${GPU_ID}.log"

    echo "[Worker][GPU ${GPU_ID}] 开始评估 ${RUN_DIR}, layer_idx=${LAYER_IDX}"
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python -m src.attack.eval_token_gcg \
      --model_name EleutherAI/pythia-410m \
      --baseline_ckpt "${BASELINE_CKPT}" \
      --layer_idx "${LAYER_IDX}" \
      --rank_r 4 \
      --run_dir "${RUN_DIR}" \
      --max_eval_samples 100 \
      --attack_start 0 \
      --n_attack_tokens 10 \
      --beam_k 512 \
      --rounds 20 \
      >"${LOG_FILE}" 2>&1

    echo "[Worker][GPU ${GPU_ID}] 完成 ${RUN_DIR}，log 写入 ${LOG_FILE}"
  done

  echo "[Worker][GPU ${GPU_ID}] 所有任务完成，退出"
}

# 3) 把 RUNS 按 GPU 拆组
#    GPU0: indices 0, 0+NUM_GPUS, 0+2*NUM_GPUS, ...
#    GPU1: indices 1, 1+NUM_GPUS, ...
GPU_RUNS=()
for ((g=0; g<NUM_GPUS; g++)); do
  GPU_RUNS[g]=""
done

for ((i=0; i<NUM_RUNS; i++)); do
  gpu_idx=$(( i % NUM_GPUS ))
  GPU_RUNS[gpu_idx]+="${RUNS[i]}|"
done

# 4) 为每个 GPU 启动一个 worker（每个 worker 内部串行）
for ((g=0; g<NUM_GPUS; g++)); do
  IFS='|' read -r -a THIS_GPU_RUNS <<< "${GPU_RUNS[g]}"
  # 过滤掉空字符串
  CLEANED=()
  for r in "${THIS_GPU_RUNS[@]}"; do
    [[ -n "${r}" ]] && CLEANED+=("${r}")
  done

  if [[ ${#CLEANED[@]} -eq 0 ]]; then
    echo "[Eval-GCG] GPU ${GPUS[g]} 没有任务，跳过 worker"
    continue
  fi

  run_on_gpu "${GPUS[g]}" "${CLEANED[@]}" &
done

echo "[Eval-GCG] 所有 GPU worker 已启动，等待全部评估结束..."
wait
echo "[Eval-GCG] 全部评估完成。"
