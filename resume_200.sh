#!/bin/bash
#SBATCH --job-name=SymmTri_200ep
#SBATCH --output=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/train_pcn_%j.out
#SBATCH --error=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/train_pcn_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nodelist=TC2N03

set -eo pipefail

cd /home/msai/harihara011/Symm3dTri/SymmCompletion
mkdir -p logs

echo "Job started: $(date)"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  CONDA_BIN=$(command -v conda || true)
  if [ -n "$CONDA_BIN" ]; then
    CONDA_PREFIX_DIR=$(dirname "$(dirname "$CONDA_BIN")")
    if [ -f "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh" ]; then
      source "$CONDA_PREFIX_DIR/etc/profile.d/conda.sh"
    fi
  fi
fi

conda activate Sym3dT
export CUDA_HOME="$CONDA_PREFIX"

# Use the 200-epoch config, resume from existing train_pcn checkpoint
CONFIG=${CONFIG:-cfgs/PCN_models/SymmCompletion_200.yaml}
EXP_NAME=${EXP_NAME:-train_pcn}
VAL_FREQ=${VAL_FREQ:-10}
VAL_INTERVAL=${VAL_INTERVAL:-50}
PYTHON=${PYTHON:-python}

export DISABLE_TFBOARD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PYTHON -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

CKPT_DIR="./experiments/SymmCompletion/PCN_models/${EXP_NAME}"
BEST_CKPT="${CKPT_DIR}/ckpt-best.pth"
LAST_CKPT="${CKPT_DIR}/ckpt-last.pth"

if [ -f "$LAST_CKPT" ]; then
  echo "Found $LAST_CKPT — resuming from latest saved epoch"
elif [ -f "$BEST_CKPT" ]; then
  echo "No ckpt-last.pth found, copying $BEST_CKPT -> $LAST_CKPT"
  cp "$BEST_CKPT" "$LAST_CKPT"
else
  echo "ERROR: No checkpoint found in $CKPT_DIR"
  exit 1
fi

echo "Running: $PYTHON main.py --config $CONFIG --val_freq $VAL_FREQ --val_interval $VAL_INTERVAL --exp_name $EXP_NAME --resume --deterministic"

$PYTHON main.py --config "$CONFIG" --val_freq "$VAL_FREQ" --val_interval "$VAL_INTERVAL" --exp_name "$EXP_NAME" --resume --deterministic

echo "Job finished: $(date)"
