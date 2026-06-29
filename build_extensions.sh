#!/bin/bash
# Build all CUDA extensions for SymmCompletion
# Run from the project root: bash build_extensions.sh

set -e
PROJECT_DIR="/home/msai/srihari003/Symm3dTri/SymmCompletion"
CONDA_ENV="/home/msai/srihari003/.conda/envs/Sym3dT"

cd "$PROJECT_DIR"

# Put conda env's nvcc (11.8) and gcc (11) first on PATH
export PATH="$CONDA_ENV/bin:$PATH"

# CUDA_HOME must point to the conda env (has nvcc 11.8)
export CUDA_HOME="$CONDA_ENV"

# CUDA headers live in PyPI nvidia packages — expose them all to the compiler
NVIDIA_SITE="$CONDA_ENV/lib/python3.11/site-packages/nvidia"
export CPATH="$NVIDIA_SITE/cuda_runtime/include:$NVIDIA_SITE/cusparse/include:$NVIDIA_SITE/cublas/include:$NVIDIA_SITE/cufft/include:$NVIDIA_SITE/curand/include:$NVIDIA_SITE/cusolver/include:$CPATH"

# libcudart.so lives in the PyPI nvidia package — expose it to the linker
export LIBRARY_PATH="$CONDA_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LIBRARY_PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# Use conda's gcc 11 as host compiler (CUDA 11.8 doesn't support gcc > 11)
export CC="$CONDA_ENV/bin/gcc"
export CXX="$CONDA_ENV/bin/g++"
export CUDAHOSTCXX="$CONDA_ENV/bin/g++"

echo "Using nvcc: $(which nvcc)"
echo "nvcc version: $(nvcc --version | grep 'release')"
echo "Using gcc:  $($CC --version | head -1)"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

# Clean old build artifacts to avoid stale .so from Python 3.9
echo "=== Cleaning old builds ==="
rm -rf extensions/chamfer_dist/build extensions/chamfer_dist/*.egg-info
rm -rf extensions/expansion_penalty/build extensions/expansion_penalty/*.egg-info
rm -rf extensions/Pointnet2/pointnet2/build extensions/Pointnet2/pointnet2/*.egg-info

echo ""
echo "=== Building chamfer_dist ==="
cd "$PROJECT_DIR/extensions/chamfer_dist"
python setup.py install --user

echo ""
echo "=== Building expansion_penalty ==="
cd "$PROJECT_DIR/extensions/expansion_penalty"
python setup.py install --user

echo ""
echo "=== Building Pointnet2 ==="
cd "$PROJECT_DIR/extensions/Pointnet2/pointnet2"
python setup.py install --user

echo ""
echo "=== All extensions built successfully ==="
echo ""
echo "Verifying imports..."
cd "$PROJECT_DIR"
python -c "
import sys
sys.path.insert(0, '.')
from extensions.chamfer_dist import ChamferDistanceL1
print('  chamfer_dist        OK')
from extensions.expansion_penalty import expansion_penalty_module
print('  expansion_penalty   OK')
"
