#!/bin/bash
#SBATCH --job-name=SymmCompletion_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -eo pipefail

# create logs dir (safe when run locally or via sbatch)
mkdir -p logs

echo "Job started: $(date)"

# Initialize conda (works for common install locations)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  # fallback if `conda` is on PATH
  CONDA_BIN=$(command -v conda || true)
  if [ -n "$CONDA_BIN" ]; then
    CONDA_PREFIX_DIR=$(dirname "$(dirname "$CONDA_BIN")")
    if [ -f "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh" ]; then
      source "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh"
    fi
  fi
fi

# Activate environment
conda activate Sym3d
export CUDA_HOME="$CONDA_PREFIX"

# --- Configurable variables (can be overridden by env vars) ---
CONFIG=${CONFIG:-cfgs/ShapeNet55_models/SymmCompletion.yaml}
EXP_NAME=${EXP_NAME:-train_pcn}
VAL_FREQ=${VAL_FREQ:-10}
VAL_INTERVAL=${VAL_INTERVAL:-50}
PYTHON=${PYTHON:-python}

# Free any lingering GPU memory and reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

# Echo the command to be executed
echo "Running: $PYTHON main.py --config $CONFIG --val_freq $VAL_FREQ --val_interval $VAL_INTERVAL --exp_name $EXP_NAME --deterministic"

# Run training
$PYTHON main.py --config "$CONFIG" --val_freq "$VAL_FREQ" --val_interval "$VAL_INTERVAL" --exp_name "$EXP_NAME" --deterministic

echo "Job finished: $(date)"
