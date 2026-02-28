GPU CI / Host Run: build + run

Quick steps to run the automated pipeline on a GPU host:

1) Make script executable:

```bash
chmod +x run_gpu_pipeline.sh
```

2) Run the pipeline (example):

```bash
./run_gpu_pipeline.sh --size_mb 200 --repeat 3
```

Options:
- `--size_mb N` : size of random input data for benchmark
- `--repeat R`  : number of benchmark runs
- `--cupy PACKAGE` : explicitly specify CuPy wheel (e.g. `cupy-cuda11x`)

Notes:
- Requires Python and appropriate CUDA/CuPy wheel to reach GPU acceleration.
- If kernels fail to compile or CuPy is missing, the pipeline will continue using CPU fallback (Numba / pure-Python).
- For best performance enable page-pinned memory and run on NUMA-aware host.
