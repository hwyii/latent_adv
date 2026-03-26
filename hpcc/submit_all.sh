#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit one SLURM job per (model, dataset) pair.
#
# Usage:
#   bash hpcc/submit_all.sh                        # all 30 jobs
#   bash hpcc/submit_all.sh --models gemma2-2b     # subset of models
#   bash hpcc/submit_all.sh --datasets imdb,enron  # subset of datasets
#   bash hpcc/submit_all.sh --dry-run              # print sbatch commands only
#
# Run from the project root:
#   cd /mnt/home/heweiyi/Documents/latent_adv
#   bash hpcc/submit_all.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ALL_MODELS="llama3-8b qwen25-7b pythia-6.9b mistral-7b gemma2-2b"
ALL_DATASETS="imdb enron harmless helpful passwordmatch wordlength"

MODELS_FILTER=""
DATASETS_FILTER=""
DRY_RUN=0

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)    MODELS_FILTER="$2";   shift 2 ;;
        --datasets)  DATASETS_FILTER="$2"; shift 2 ;;
        --dry-run)   DRY_RUN=1;            shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Apply filters (comma-separated → space-separated)
if [[ -n "$MODELS_FILTER" ]]; then
    MODELS=$(echo "$MODELS_FILTER" | tr ',' ' ')
else
    MODELS="$ALL_MODELS"
fi

if [[ -n "$DATASETS_FILTER" ]]; then
    DATASETS=$(echo "$DATASETS_FILTER" | tr ',' ' ')
else
    DATASETS="$ALL_DATASETS"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_stage0.sh"

echo "========================================"
echo "Models:   ${MODELS}"
echo "Datasets: ${DATASETS}"
echo "Dry run:  ${DRY_RUN}"
echo "========================================"

N=0
for MODEL in $MODELS; do
    for DATASET in $DATASETS; do
        TAG="${MODEL}/${DATASET}"
        OUT_DIR="outputs/stage0/${MODEL}/${DATASET}"

        CMD="sbatch \
            --job-name=s0_${MODEL}_${DATASET} \
            --output=${OUT_DIR}/slurm_%j.log \
            --error=${OUT_DIR}/slurm_%j.err \
            --export=ALL,MODEL_SHORT=${MODEL},DATASET_TAG=${DATASET} \
            ${SLURM_SCRIPT}"

        echo "  [${TAG}]"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "    (dry-run) $CMD"
        else
            mkdir -p "${OUT_DIR}"
            eval "$CMD"
        fi
        N=$((N + 1))
    done
done

echo "========================================"
echo "Total jobs submitted: ${N}"
echo "========================================"
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R'"
