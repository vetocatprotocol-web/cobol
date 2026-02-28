"""
Python wrapper for GPU Trie search kernel using CuPy.
Requires `trie_search_kernel.cu` compiled to a CuPy RawModule.
"""

import cupy as cp
import numpy as np

# compile kernel on import
_kernel_code = open('trie_search_kernel.cu').read()
_trie_module = cp.RawModule(code=_kernel_code, backend='nvcc', options=('-std=c++11',))
_trie_kernel = _trie_module.get_function('trie_search_kernel')


def search_gpu(data: bytes, trie_array: np.ndarray, trie_size: int):
    """Search patterns using GPU kernel.

    Args:
        data: input byte string
        trie_array: flat numpy array of TrieNode structure (dtype=np.int32)
        trie_size: number of nodes in trie

    Returns:
        list of (offset, pattern_id)
    """
    data_arr = np.frombuffer(data, dtype=np.uint8)
    d_data = cp.asarray(data_arr)
    d_trie = cp.asarray(trie_array)
    # prepare output buffers
    max_matches = len(data_arr)
    d_offsets = cp.zeros(max_matches, dtype=cp.int32)
    d_ids = cp.zeros(max_matches, dtype=cp.int32)
    d_count = cp.zeros(1, dtype=cp.int32)
    # launch kernel
    threads = 256
    blocks = (len(data_arr) + threads - 1) // threads
    _trie_kernel((blocks,), (threads,),
                  (d_data, data_arr.size, d_trie, trie_size,
                   d_offsets, d_ids, d_count))
    count = int(d_count.get()[0])
    offsets = cp.asnumpy(d_offsets[:count])
    ids = cp.asnumpy(d_ids[:count])
    return list(zip(offsets.tolist(), ids.tolist()))
