#!/bin/bash
#SBATCH --job-name=SymmTri_diagnose
#SBATCH --output=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/diagnose_%j.out
#SBATCH --error=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/diagnose_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00

set -eo pipefail

cd /home/msai/harihara011/Symm3dTri/SymmCompletion
mkdir -p logs

echo "Job started: $(date)"

# ── conda ──────────────────────────────────────────────────────────────────
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  CONDA_BIN=$(command -v conda || true)
  if [ -n "$CONDA_BIN" ]; then
    CONDA_PREFIX_DIR=$(dirname "$(dirname "$CONDA_BIN")")
    [ -f "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh" ] && \
      source "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh"
  fi
fi

conda activate Sym3dT
export CUDA_HOME="$CONDA_PREFIX"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DISABLE_TFBOARD=1

# ── config ────────────────────────────────────────────────────────────────
CONFIG=${CONFIG:-cfgs/PCN_models/SymmCompletion.yaml}
CKPT=${CKPT:-experiments/SymmCompletion/PCN_models/train_pcn/ckpt-best.pth}
MODE=${MODE:-median}   # easy | median | hard (only used for ShapeNet)

echo "Config  : $CONFIG"
echo "Ckpt    : $CKPT"
echo "Mode    : $MODE"
echo ""

python diagnose.py \
  --config  "$CONFIG" \
  --ckpts   "$CKPT"   \
  --mode    "$MODE"

echo "Job finished: $(date)"
