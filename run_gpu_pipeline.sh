#!/usr/bin/env bash
set -euo pipefail

# run_gpu_pipeline.sh
# Automated build + run script for GPU host
# Usage: ./run_gpu_pipeline.sh [--size_mb N] [--repeat R] [--cupy PACKAGE]

SIZE_MB=100
REPEAT=3
CUPY_PKG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --size_mb) SIZE_MB="$2"; shift 2;;
    --repeat) REPEAT="$2"; shift 2;;
    --cupy) CUPY_PKG="$2"; shift 2;;
    -h|--help) echo "Usage: $0 [--size_mb N] [--repeat R] [--cupy PACKAGE]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "== COBOL GPU Pipeline: build+run =="
echo "Size: ${SIZE_MB} MB, Repeat: ${REPEAT}"

# 1) Basic environment checks
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi found:"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
else
  echo "Warning: nvidia-smi not found. Ensure CUDA drivers are installed on host." >&2
fi

if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc found:"; nvcc --version | head -n 1 || true
else
  echo "Warning: nvcc not found. Installing cupy may still work with binary wheels." >&2
fi

# 2) Upgrade pip and install base deps
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt || true
python -m pip install numba || true

# 3) Install CuPy: prefer user-provided package, else try detection
if [[ -n "${CUPY_PKG}" ]]; then
  echo "Installing CuPy package ${CUPY_PKG}..."
  python -m pip install "${CUPY_PKG}" || { echo "Failed to install ${CUPY_PKG}"; exit 1; }
else
  echo "Attempting to install CuPy (best-effort):"
  python -m pip install cupy || {
    echo "cupy install failed; trying common CUDA wheel names..."
    python -m pip install cupy-cuda11x || python -m pip install cupy-cuda12x || echo "Install CuPy manually matching your CUDA version." >&2
  }
fi

# 4) Compile CUDA kernels (via CuPy RawModule) - will succeed only if nvcc/CuPy present
echo "Compiling CUDA kernels (compile_kernels.py)..."
python compile_kernels.py || echo "Kernel compilation failed or CuPy not available; continuing with CPU fallback."

# 5) Run end-to-end benchmark
echo "Running end-to-end benchmark..."
python bench_end_to_end.py --size_mb ${SIZE_MB} --repeat ${REPEAT}

echo "Done. Review output above for throughput and errors."
