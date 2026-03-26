#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time environment setup on HPCC.
# Run this ONCE from login node after syncing the code.
#
# Usage:
#   bash hpcc/setup_env.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="latent_adv"
SCRATCH="/mnt/scratch/heweiyi"
CODE_DIR="/mnt/home/heweiyi/Documents/latent_adv"

echo "=== [1/4] Loading modules ==="
module purge
module load CUDA/12.4       # adjust to what your HPCC provides: module avail CUDA
module load Anaconda3       # adjust: module avail Anaconda

echo "=== [2/4] Creating conda environment: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.10 -y
conda activate "${ENV_NAME}"

echo "=== [3/4] Installing packages ==="
# Install PyTorch first (with CUDA 12.4 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install the rest
pip install -r "${CODE_DIR}/requirements_hpcc.txt"

echo "=== [4/4] Creating scratch directories ==="
mkdir -p "${SCRATCH}/hf_cache/datasets"
mkdir -p "${SCRATCH}/checkpoints"

echo ""
echo "=== Done ==="
echo "HuggingFace cache will be stored at: ${SCRATCH}/hf_cache"
echo ""
echo "Next steps:"
echo "  1. Log in to HuggingFace (needed for gated models Llama-3, Gemma-2):"
echo "       conda activate ${ENV_NAME}"
echo "       huggingface-cli login"
echo "  2. Submit jobs:"
echo "       cd ${CODE_DIR}"
echo "       bash hpcc/submit_all.sh --dry-run   # preview"
echo "       bash hpcc/submit_all.sh             # submit all 30 jobs"
