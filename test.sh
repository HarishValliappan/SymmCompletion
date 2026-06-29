#!/bin/bash
#SBATCH --job-name=SymmTriV10_test
#SBATCH --output=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/test_%j.out
#SBATCH --error=/home/msai/harihara011/Symm3dTri/SymmCompletion/logs/test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

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

CONFIG=${CONFIG:-cfgs/PCN_models/SymmCompletion.yaml}
EXP_NAME=${EXP_NAME:-test_pcn}
CKPT=${CKPT:-experiments/SymmCompletion/PCN_models/train_pcn/ckpt-best.pth}
PYTHON=${PYTHON:-python}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Running: $PYTHON main.py --test --config $CONFIG --exp_name $EXP_NAME --ckpts $CKPT"
$PYTHON main.py --test --config "$CONFIG" --exp_name "$EXP_NAME" --ckpts "$CKPT"

echo "Job finished: $(date)"
