"""
Python wrapper for GPU Huffman kernels using CuPy.
Assumes `huffman_gpu_kernel.cu` compiled similarly.
"""

import cupy as cp
import numpy as np

_kernel_code = open('huffman_gpu_kernel.cu').read()
_huff_module = cp.RawModule(code=_kernel_code, backend='nvcc', options=('-std=c++11',))
_build_kernel = _huff_module.get_function('build_huffman_tree')
_encode_kernel = _huff_module.get_function('huffman_encode')
_decode_kernel = _huff_module.get_function('huffman_decode')


def build_tree(freqs: np.ndarray):
    """Build Huffman tree on GPU. Returns device array of nodes."""
    freqs_gpu = cp.asarray(freqs.astype(np.int32))
    tree_gpu = cp.zeros((512, 4), dtype=cp.int32)  # HuffmanNode: left,right,symbol,freq
    node_count = cp.zeros(1, dtype=cp.int32)
    threads = 256
    blocks = 1
    _build_kernel((blocks,), (threads,), (freqs_gpu, tree_gpu, node_count))
    return tree_gpu, int(node_count.get()[0])


def encode(data: bytes, tree_gpu):
    """Encode data using Huffman tree on GPU."""
    data_arr = np.frombuffer(data, dtype=np.uint8)
    d_data = cp.asarray(data_arr)
    encoded = cp.zeros(len(data_arr), dtype=cp.uint8)
    encoded_len = cp.zeros(1, dtype=cp.int32)
    threads = 256
    blocks = (len(data_arr) + threads - 1) // threads
    _encode_kernel((blocks,), (threads,),
                   (d_data, data_arr.size, tree_gpu, encoded, encoded_len))
    length = int(encoded_len.get()[0])
    return cp.asnumpy(encoded[:length])


def decode(encoded_bytes: bytes, tree_gpu):
    data_arr = np.frombuffer(encoded_bytes, dtype=np.uint8)
    d_enc = cp.asarray(data_arr)
    decoded = cp.zeros(len(data_arr), dtype=cp.uint8)
    decoded_len = cp.zeros(1, dtype=cp.int32)
    threads = 256
    blocks = (len(data_arr) + threads - 1) // threads
    _decode_kernel((blocks,), (threads,),
                   (d_enc, data_arr.size, tree_gpu, decoded, decoded_len))
    length = int(decoded_len.get()[0])
    return cp.asnumpy(decoded[:length])
