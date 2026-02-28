"""GPU helpers for Layer-7 Huffman.

Provides `compute_histograms` which computes 256-bin histograms per fixed-size
sub-chunk (default 64KB). If CuPy/CUDA is available the work is performed on
GPU (via `compute_histograms_kernel`); otherwise a CPU fallback is used.

The histograms can then be used to build canonical Huffman tables per-chunk
and encode each chunk in parallel (see `huffman_parallel.py`).
"""

import os
import numpy as np

_HAS_CUPY = False
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None


def _load_raw_module(use_warp: bool = False):
    # Read main kernel and optional warp-aggregation kernel, concatenate
    base_dir = os.path.dirname(__file__)
    parts = []
    main_path = os.path.join(base_dir, 'huffman_gpu_kernel.cu')
    with open(main_path, 'r') as f:
        parts.append(f.read())
    if use_warp:
        warp_path = os.path.join(base_dir, 'huffman_gpu_kernel_warp.cu')
        if os.path.exists(warp_path):
            with open(warp_path, 'r') as f:
                parts.append(f.read())
    src = '\n'.join(parts)
    mod = cp.RawModule(code=src, backend='nvcc', options=('-std=c++11',))
    return mod


def compute_histograms_cpu(data: bytes, chunk_size: int = 65536) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    n = arr.size
    num_chunks = (n + chunk_size - 1) // chunk_size
    out = np.zeros((num_chunks, 256), dtype=np.uint32)
    for i in range(num_chunks):
        a = arr[i * chunk_size:(i + 1) * chunk_size]
        if a.size:
            h = np.bincount(a, minlength=256).astype(np.uint32)
            out[i, :] = h
    return out


def compute_histograms(data: bytes, chunk_size: int = 65536, use_warp_agg: bool = False) -> np.ndarray:
    """Compute per-chunk 256-bin histograms.

    If `use_warp_agg` is True and a warp-aggregation kernel is available,
    the warp-optimized kernel will be used (fewer global atomics).

    Returns a NumPy array shaped `(num_chunks, 256)` of uint32 counts.
    If CuPy/CUDA is available, runs on GPU; otherwise falls back to CPU.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError('data must be bytes or bytearray')

    if not _HAS_CUPY:
        return compute_histograms_cpu(data, chunk_size=chunk_size)

    # GPU path
    src = np.frombuffer(data, dtype=np.uint8)
    n = src.size
    num_chunks = (n + chunk_size - 1) // chunk_size

    mod = _load_raw_module(use_warp=use_warp_agg)
    kernel_name = 'compute_histograms_warp_kernel' if use_warp_agg else 'compute_histograms_kernel'
    kernel = mod.get_function(kernel_name)

    d_data = cp.asarray(src)
    d_out = cp.zeros(int(num_chunks * 256), dtype=cp.uint32)

    threads = 256
    blocks = int(num_chunks)
    shared_mem_bytes = 256 * np.dtype(np.uint32).itemsize

    # Launch kernel: signature (data, n, out_hist, chunk_size)
    kernel((blocks,), (threads,), (d_data, np.uint64(n), d_out, np.uint64(chunk_size)), shared_mem=shared_mem_bytes)
    cp.cuda.Stream.null.synchronize()

    out = cp.asnumpy(d_out).reshape((num_chunks, 256))
    return out

