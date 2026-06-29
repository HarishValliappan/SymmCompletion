#!/bin/bash
#SBATCH --job-name=TriP_SN55
#SBATCH --output=/home/msai/srihari003/Symm3dTri/SymmCompletion/logs/train_sn55_%j.out
#SBATCH --error=/home/msai/srihari003/Symm3dTri/SymmCompletion/logs/train_sn55_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --nodelist=TC2N04
set -eo pipefail

# Change to the project directory where main.py lives
cd /home/msai/srihari003/Symm3dTri/SymmCompletion

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
conda activate Sym3dT
export CUDA_HOME="$CONDA_PREFIX"

# Add nvcc (11.8) from conda env to PATH
export PATH="$CONDA_PREFIX/bin:$PATH"

# KNN_CUDA is JIT-compiled at runtime — needs to be on PYTHONPATH
export PYTHONPATH="/home/msai/srihari003/Symm3dTri/SymmCompletion/extensions/KNN_CUDA:$PYTHONPATH"

# Host compiler: conda gcc 11 (CUDA 11.8 requires gcc <= 11)
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDAHOSTCXX="$CONDA_PREFIX/bin/g++"

# CUDA headers (cuda_runtime.h, cusparse.h, etc.) from PyPI nvidia packages
NVIDIA_SITE="$CONDA_PREFIX/lib/python3.11/site-packages/nvidia"
export CPATH="$NVIDIA_SITE/cuda_runtime/include:$NVIDIA_SITE/cusparse/include:$NVIDIA_SITE/cublas/include:$NVIDIA_SITE/cufft/include:$NVIDIA_SITE/curand/include:$NVIDIA_SITE/cusolver/include:$CPATH"

# libcudart.so for the JIT linker
export LIBRARY_PATH="$CONDA_PREFIX/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$NVIDIA_SITE/cuda_runtime/lib:$LD_LIBRARY_PATH"

# --- ShapeNet55 dataset training config ---
CONFIG=${CONFIG:-cfgs/ShapeNet55_models/SymmCompletion_500.yaml}
EXP_NAME=${EXP_NAME:-train_sn55}
VAL_FREQ=${VAL_FREQ:-10}
VAL_INTERVAL=${VAL_INTERVAL:-50}
PYTHON=${PYTHON:-python}

# Disable TensorBoard logging to save disk quota
export DISABLE_TFBOARD=1

# Free any lingering GPU memory and reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force unbuffered Python output so logs appear immediately
export PYTHONUNBUFFERED=1
$PYTHON -c "import torch; torch.cuda.empty_cache(); print('CUDA cache cleared')"

# Echo the command to be executed
echo "Running: $PYTHON main.py --config $CONFIG --val_freq $VAL_FREQ --val_interval $VAL_INTERVAL --exp_name $EXP_NAME --deterministic"

# Run training
$PYTHON main.py --config "$CONFIG" --val_freq "$VAL_FREQ" --val_interval "$VAL_INTERVAL" --exp_name "$EXP_NAME" --deterministic

echo "Job finished: $(date)"
