"""Compile CUDA kernels used by COBOL Protocol GPU acceleration.

Usage:
    python compile_kernels.py

This script uses nvcc (via cupy.RawModule) to compile kernels to a cache
that CuPy can load quickly.
"""
import cupy as cp

kernels = [
    ('trie_search_kernel.cu', 'trie_search_kernel'),
    ('huffman_gpu_kernel.cu', 'build_huffman_tree'),
]

for filename, func in kernels:
    with open(filename) as f:
        code = f.read()
    print(f"Compiling {filename}...")
    module = cp.RawModule(code=code, backend='nvcc', options=('-std=c++11',))
    # force compile by retrieving the function
    module.get_function(func)
    print(f"Compiled {filename}")

print("All kernels compiled.")
