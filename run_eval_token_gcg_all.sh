#!/usr/bin/env bash
set -u

# --- 配置区 ---
RUN_ROOT="outputs_gpt2_full_l15/Helpful"

BASELINE_CKPT="outputs_gpt2_baseline/helpful/best_Helpful.pt"
LOG_DIR="logs_test/logs_helpful/debugl15_2_eval_last_loc_lambda10" 
QUEUE_DIR="${LOG_DIR}/queue"        # 任务队列目录
PROCESSING_DIR="${LOG_DIR}/processing" # 正在运行目录
INCLUDE_KEYS=("") # 只运行包含这些关键字的任务，空则不过滤
TARGET_JSON_NAME="debugl15_last_suffix_20rounds_10token_token_gcg_100_last_token_gcgcoord_results.json"


mkdir -p "${LOG_DIR}"
mkdir -p "${QUEUE_DIR}"
mkdir -p "${PROCESSING_DIR}"

# 可用 GPU 列表
GPUS=(1) 
NUM_GPUS=${#GPUS[@]}

# --- 1. 准备任务队列 (智能增量模式) ---
echo "[Init] 正在扫描任务..."

# 强制清空 queue 和 processing，因为我们要重新计算哪些需要跑
# 这能解决你担心的“残留任务”问题，把脏状态洗掉
rm -f "${QUEUE_DIR}"/*
rm -f "${PROCESSING_DIR}"/*

JOB_ID=0
SKIP_COUNT=0

# 查找所有 run 并生成任务文件
while IFS= read -r final_dir; do
  RUN_DIR=$(dirname "${final_dir}")
  
  # ---- 1. INCLUDE 过滤 (保持你原有的逻辑) ----
  if [[ ${#INCLUDE_KEYS[@]} -gt 0 ]]; then
    keep=0
    for key in "${INCLUDE_KEYS[@]}"; do
      if [[ "${RUN_DIR}" == *"${key}"* ]]; then
        keep=1
        break
      fi
    done
    if [[ $keep -eq 0 ]]; then
      continue
    fi
  fi

  # ---- 2. [新增] 检查结果是否存在 (断点续传核心) ----
  # 检查 RUN_DIR 下是否已经生成了结果 JSON 文件
  if [[ -f "${RUN_DIR}/${TARGET_JSON_NAME}" ]]; then
    # 如果文件存在，说明跑过了，跳过
    ((SKIP_COUNT++))
    continue
  fi
  # ------------------------------------------------

  # 生成任务文件 (只有没跑过的才会走到这里)
  JOB_FILE=$(printf "%s/job_%04d" "${QUEUE_DIR}" "${JOB_ID}")
  echo "${RUN_DIR}" > "${JOB_FILE}"
  
  ((JOB_ID++))
done < <(find "${RUN_ROOT}" -type d -name "final_intervention" | sort)

TOTAL_JOBS=${JOB_ID}
echo "[Init] 扫描结束：已跳过 ${SKIP_COUNT} 个已完成任务。"
echo "[Init] 剩余 ${TOTAL_JOBS} 个任务需执行，已生成到 ${QUEUE_DIR}"

if [[ ${TOTAL_JOBS} -eq 0 ]]; then
  echo "所有任务均已完成，无需执行，退出。"
  exit 0
fi

# --- 2. 定义 Worker 函数 ---
# 每个 GPU 作为一个 Worker，不断去 QUEUE_DIR 里抢文件
run_worker() {
  local GPU_ID="$1"
  
  echo "[Worker-GPU${GPU_ID}] 启动，开始抢任务..."
  
  while true; do
    # 1. 获取队列中的第一个文件 (ls 默认按字母序)
    # 2. 尝试将它 mv 到 processing 目录
    #    mv 是原子的，如果两个 GPU 同时抢同一个文件，只有一个会成功，另一个报错
    
    # 这里的 head -n 1 拿到文件名
    local TASK_FILE
    TASK_FILE=$(ls -1 "${QUEUE_DIR}"/job_* 2>/dev/null | head -n 1)
    
    # 如果没拿到文件名，说明队列空了，退出循环
    if [[ -z "${TASK_FILE}" ]]; then
      echo "[Worker-GPU${GPU_ID}] 队列为空，下班了！"
      break
    fi
    
    local BASE_FILENAME
    BASE_FILENAME=$(basename "${TASK_FILE}")
    
    # 尝试原子移动 (抢占)
    if mv "${TASK_FILE}" "${PROCESSING_DIR}/${BASE_FILENAME}" 2>/dev/null; then
      # === 抢到了！执行任务 ===
      
      # 读取任务内容 (RUN_DIR)
      local RUN_DIR
      RUN_DIR=$(cat "${PROCESSING_DIR}/${BASE_FILENAME}")
      
      # 解析参数
      local BASE_NAME
      BASE_NAME=$(basename "${RUN_DIR}")
      local LAYER_IDX
      LAYER_IDX=$(echo "${BASE_NAME}" | sed -n 's/.*_L\([0-9]\+\)_A.*/\1/p')
      
      if [[ -z "${LAYER_IDX}" ]]; then
        echo "[Worker-GPU${GPU_ID}] [WARN] 解析 Layer 失败: ${BASE_NAME}"
      else
        local LOG_FILE="${LOG_DIR}/${BASE_NAME}_gpu${GPU_ID}.log"
        echo "[Worker-GPU${GPU_ID}] 执行: ${BASE_NAME} (L${LAYER_IDX})"
        
        # --- 核心执行命令 ---
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        python -m src.attack.eval_token_gcg \
          --model_name gpt2 \
          --baseline_ckpt "${BASELINE_CKPT}" \
          --data_dir src/data/helpful_splits.json \
          --layer_idx "${LAYER_IDX}" \
          --rank_r 64 \
          --run_dir "${RUN_DIR}" \
          --dataset "AlignmentResearch/Helpful" \
          --max_eval_samples 100 \
          --seed 42 \
          --attack_start 10 \
          --n_attack_tokens 10 \
          --beam_k 256 \
          --rounds 20 \
          --output_json "${RUN_DIR}/${TARGET_JSON_NAME}" \
          --attack_mode suffix \
          --n_candidates_per_it 128 \
          --reft_loc_mode last_n \
          --n_reft_positions 15 \
          --position l15 \
          > "${LOG_FILE}" 2>&1
        # -------------------
      fi
      
      # 做完后删除任务文件，保持清洁
      rm "${PROCESSING_DIR}/${BASE_FILENAME}"
      
    else
      # mv 失败，说明被别的 GPU 抢先了，继续下一次循环
      continue
    fi
  done
}

# --- 3. 并行启动 Worker ---
for GPU in "${GPUS[@]}"; do
  run_worker "${GPU}" &
done

# --- 4. 等待所有 Worker 结束 ---
wait
echo "[Done] 所有任务处理完毕。"
# 清理空目录
rmdir "${QUEUE_DIR}" "${PROCESSING_DIR}" 2>/dev/null