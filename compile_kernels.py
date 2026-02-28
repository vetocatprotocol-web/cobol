"""Compile CUDA kernels used by COBOL Protocol GPU acceleration.

Usage:
    python compile_kernels.py

This script uses nvcc (via cupy.RawModule) to compile kernels to a cache
that CuPy can load quickly.
"""
import cupy as cp

kernels = [
    ('trie_search_kernel.cu', 'trie_search_kernel'),
    ('huffman_gpu_kernel.cu', 'compute_histograms_kernel'),
    ('huffman_gpu_kernel_warp.cu', 'compute_histograms_warp_kernel'),
]

for filename, func in kernels:
    try:
        with open(filename) as f:
            code = f.read()
    except FileNotFoundError:
        print(f"Skipping missing {filename}")
        continue
    print(f"Compiling {filename}...")
    module = cp.RawModule(code=code, backend='nvcc', options=('-std=c++11',))
    # force compile by retrieving the function
    try:
        module.get_function(func)
    except Exception as e:
        print(f"Warning: function {func} not found in {filename}: {e}")
    print(f"Compiled {filename}")

print("All kernels compiled (or skipped if missing).")
