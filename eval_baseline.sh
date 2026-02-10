#!/bin/bash

# 基础配置
MODEL_NAME="gpt2"
BASE_DIR="outputs"
SPLIT_DIR="src/data"
GPU_ID=2

# 数据集列表：tag | dataset_name | split_file | ckpt_name
# 格式：标签;数据集路径;索引文件;权重文件名
JOBS=(
    #"imdb;AlignmentResearch/IMDB;imdb_splits.json;best_IMDB.pt"
    "harmless;AlignmentResearch/Harmless;harmless_splits.json;best_Harmless.pt"
    "helpful;AlignmentResearch/Helpful;helpful_splits.json;best_Helpful.pt"
    "enron;AlignmentResearch/EnronSpam;enron_splits.json;best_EnronSpam.pt"
)

for job in "${JOBS[@]}"; do
    # 解析变量
    IFS=";" read -r TAG DS SPLIT CKPT <<< "${job}"
    
    echo "============================================"
    echo "Starting GCG Attack for: ${TAG}"
    echo "GPU: ${GPU_ID} | Dataset: ${DS}"
    echo "============================================"

    # 创建输出目录
    mkdir -p "${BASE_DIR}/${TAG}"

    # 执行 Python 命令
    # 2>&1 | tee 会将结果同时显示在屏幕并写入 log 文件
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m src.attack.eval_token_gcg_baseline \
      --model_name ${MODEL_NAME} \
      --baseline_ckpt "${BASE_DIR}/${TAG}/${CKPT}" \
      --dataset "${DS}" \
      --split "${SPLIT_DIR}/${SPLIT}" \
      --max_eval_samples 100 \
      --n_attack_tokens 10 \
      --beam_k 256 \
      --rounds 20 \
      --attack_mode suffix \
      --n_candidates_per_it 128 \
      --output_json "${BASE_DIR}/${TAG}/suffix_baseline_eval_10tokens_20rounds.json" \
      2>&1 | tee "${BASE_DIR}/${TAG}/gcg_attack.log"

    echo -e "\nFinished ${TAG}. Results in ${BASE_DIR}/${TAG}/\n"
done