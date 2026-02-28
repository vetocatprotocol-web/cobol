COBOL Protocol GPU Acceleration

Quick start (host with CUDA & Python):

1) Install dependencies (example for CUDA 11.x):

```bash
python -m pip install cupy-cuda11x numpy
pip install -r requirements.txt
```

2) Compile kernels:

```bash
python compile_kernels.py
```

3) Run end-to-end benchmark:

```bash
python bench_end_to_end.py --size_mb 100 --repeat 3
```

Notes:
- If no GPU is available, pipeline falls back to CPU implementations (Numba JIT).
- For best performance enable page-pinned memory and run on NUMA-aware machine.
