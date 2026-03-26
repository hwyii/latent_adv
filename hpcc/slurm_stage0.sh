#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM job template for one (model, dataset) training run.
# Called by submit_all.sh — do not submit this file directly.
#
# Required environment variables (set by submit_all.sh via --export):
#   MODEL_SHORT   e.g.  llama3-8b
#   DATASET_TAG   e.g.  imdb
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=s0_${MODEL_SHORT}_${DATASET_TAG}
#SBATCH --output=outputs/stage0/%x/slurm_%j.log
#SBATCH --error=outputs/stage0/%x/slurm_%j.err

# ── Time & resources ──────────────────────────────────────────────────────────
#   7B bfloat16 models:  ~10-14 h per dataset
#   Gemma-2-2b float32:  ~4-6 h per dataset
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# ── GPU request ───────────────────────────────────────────────────────────────
# Check available GPU partitions with:  sinfo -o "%P %G" | grep gpu
# Common MSU HPCC GPU partitions / constraints:
#   --partition=gpu  --gres=gpu:a100:1          (A100 40/80 GB)
#   --partition=gpu  --gres=gpu:v100:1          (V100 32 GB — too small for 7B)
# All our models need ≥ 44 GB → request A100 80 GB if possible.
# Adjust the lines below to match your HPCC's partition names.

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
# Uncomment if you need to specifically target 80 GB cards:
##SBATCH --constraint=a100_80g

# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "======================================"
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Model:    ${MODEL_SHORT}"
echo "Dataset:  ${DATASET_TAG}"
echo "Started:  $(date)"
echo "======================================"

# ── Environment setup ─────────────────────────────────────────────────────────
# Redirect HuggingFace cache to scratch (models can be 10-30 GB each)
export HF_HOME=/mnt/scratch/heweiyi/hf_cache
export TRANSFORMERS_CACHE=/mnt/scratch/heweiyi/hf_cache
export HF_DATASETS_CACHE=/mnt/scratch/heweiyi/hf_cache/datasets

# Mirror (optional, remove if you have direct HF access)
export HF_ENDPOINT=https://hf-mirror.com

# Suppress tokenizer parallelism warnings inside DataLoader workers
export TOKENIZERS_PARALLELISM=false

# ── Conda environment ─────────────────────────────────────────────────────────
# Adjust env name and conda path to match your HPCC setup
module purge
module load CUDA/12.4           # check: module avail CUDA
module load Anaconda3           # check: module avail Anaconda

conda activate latent_adv       # created with setup_env.sh

# ── Working directory ─────────────────────────────────────────────────────────
cd /mnt/home/heweiyi/Documents/latent_adv

# ── Run ───────────────────────────────────────────────────────────────────────
mkdir -p outputs/stage0/${MODEL_SHORT}/${DATASET_TAG}

python run_one_job.py \
    --model   "${MODEL_SHORT}" \
    --dataset "${DATASET_TAG}" \
    2>&1 | tee outputs/stage0/${MODEL_SHORT}/${DATASET_TAG}/stdout.log

echo "======================================"
echo "Finished: $(date)"
echo "======================================"
