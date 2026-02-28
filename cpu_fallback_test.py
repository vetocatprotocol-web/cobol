"""
CPU fallback validation script.
- Tests Layer 6 CPU search path via `GPUAccelerationEngine.l6_search_cpu`
- Tests pure-Python pattern search in `numba_dictionary.py`
- Tests shared memory DMA helper in `hpc_engine.py`
"""
import sys
import traceback

from gpu_acceleration import GPUAccelerationEngine
import numba_dictionary
from hpc_engine import dma_transfer_example
import numpy as np


def test_l6_cpu_search():
    engine = GPUAccelerationEngine()
    text = b"abcde pattern abc pattern pattern xyz abc"
    patterns = {1: b"abc", 2: b"pattern", 3: b"xyz"}
    matches = engine.l6_search_cpu(text, patterns)
    print('L6 CPU matches:', matches)


def test_python_pattern_search():
    patterns = {1: b"abc", 2: b"pattern"}
    text = b"xxabcxxpatternxxabc"
    matches = numba_dictionary.python_pattern_search(patterns, text)
    print('python_pattern_search matches:', matches)


def test_dma_example():
    arr = np.arange(1024, dtype=np.uint8)
    received = dma_transfer_example(arr, shm_name='cobol_test_shm')
    print('DMA transfer works, first 5 bytes:', received[:5])


if __name__ == '__main__':
    try:
        print('Running CPU fallback tests...')
        test_l6_cpu_search()
        test_python_pattern_search()
        test_dma_example()
        print('CPU fallback tests completed successfully')
    except Exception as e:
        print('CPU fallback tests failed:', e)
        traceback.print_exc()
        sys.exit(1)
