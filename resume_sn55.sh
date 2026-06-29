#!/bin/bash
#SBATCH --job-name=ResSN55
#SBATCH --output=/home/msai/srihari003/Symm3dTri/SymmCompletion/logs/resume_sn55_%j.out
#SBATCH --error=/home/msai/srihari003/Symm3dTri/SymmCompletion/logs/resume_sn55_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nodelist=TC2N04
set -eo pipefail

cd /home/msai/srihari003/Symm3dTri/SymmCompletion
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

export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONPATH="/home/msai/srihari003/Symm3dTri/SymmCompletion/extensions/KNN_CUDA:$PYTHONPATH"

export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/g++"

NVIDIA_SITE="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
export CPATH="$NVIDIA_SITE/cuda_runtime/include:$NVIDIA_SITE/cusparse/include:$NVIDIA_SITE/cublas/include:$NVIDIA_SITE/cufft/include:$NVIDIA_SITE/curand/include:$NVIDIA_SITE/cusolver/include:$CPATH"

export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NVIDIA_SITE/cuda_runtime/lib:$LD_LIBRARY_PATH"

CONFIG=${CONFIG:-cfgs/ShapeNet55_models/SymmCompletion_500.yaml}
EXP_NAME=${EXP_NAME:-train_sn55}
VAL_FREQ=${VAL_FREQ:-10}
VAL_INTERVAL=${VAL_INTERVAL:-50}
PYTHON=${PYTHON:-python}

export DISABLE_TFBOARD=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
$PYTHON -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

CKPT_DIR="./experiments/SymmCompletion_500/ShapeNet55_models/${EXP_NAME}"
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
