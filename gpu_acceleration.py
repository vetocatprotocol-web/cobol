"""
Minimal GPU Acceleration module with CPU fallback for Layer 6.
This version is intentionally lightweight so it can run in environments
without CUDA/CuPy. GPU wrappers (trie_gpu.py, huffman_gpu.py) are used
when available.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class GPUDetector:
    """Detects whether a GPU/CuPy is available."""

    def __init__(self):
        self.gpu_available = False
        self.cuda_version = None
        self.gpu_count = 0
        self.compute_capability = None
        try:
            import cupy as cp
            # simple probe
            try:
                dev = cp.cuda.Device()
                cc = dev.compute_capability
                self.compute_capability = f"{cc[0]}.{cc[1]}"
                self.gpu_count = cp.cuda.runtime.getDeviceCount()
                self.gpu_available = True
                logger.info(f"CuPy GPU detected: count={self.gpu_count} cc={self.compute_capability}")
            except Exception:
                # cupy installed but device init failed
                self.gpu_available = False
        except Exception:
            self.gpu_available = False

    def get_info(self) -> Dict[str, Any]:
        return {
            'gpu_available': self.gpu_available,
            'cuda_version': self.cuda_version,
            'gpu_count': self.gpu_count,
            'compute_capability': self.compute_capability,
        }


class GPUMemoryManager:
    """Simple wrapper for CPU/GPU transfers (CuPy) with graceful fallback."""

    def __init__(self, detector: GPUDetector):
        self.detector = detector
        self.cp = None
        self.cupy_available = False
        if detector.gpu_available:
            try:
                import cupy as cp
                self.cp = cp
                self.cupy_available = True
            except Exception:
                self.cupy_available = False

    def to_gpu(self, data):
        if not self.cupy_available or self.cp is None:
            return data
        try:
            import numpy as np
            if isinstance(data, bytes):
                arr = np.frombuffer(data, dtype=np.uint8)
                return self.cp.asarray(arr)
            elif hasattr(data, '__array__'):
                return self.cp.asarray(data)
            else:
                return data
        except Exception as e:
            logger.warning(f"to_gpu failed: {e}")
            return data

    def to_cpu(self, gpu_data):
        if not self.cupy_available or self.cp is None:
            return gpu_data
        try:
            if hasattr(gpu_data, 'get'):
                return gpu_data.get()
            return gpu_data
        except Exception as e:
            logger.warning(f"to_cpu failed: {e}")
            return gpu_data


class GPUTrieAccelerator:
    """
    Provides search_patterns_gpu and CPU fallback _search_patterns_cpu.
    """

    def __init__(self, detector: GPUDetector):
        self.detector = detector
        self.gpu_mem_manager = GPUMemoryManager(detector)
        self.use_gpu = detector.gpu_available
        # try to import GPU wrapper
        try:
            from trie_gpu import search_gpu
            self.search_kernel = search_gpu
        except Exception:
            self.search_kernel = None

    def search_patterns_gpu(self, text: bytes, patterns: Dict[int, bytes]) -> list:
        if not self.use_gpu or self.search_kernel is None:
            return self._search_patterns_cpu(text, patterns)
        try:
            # build a simple flat trie array placeholder (wrapper expects it)
            trie_array = self._build_flat_trie(patterns)
            trie_size = 1
            return self.search_kernel(text, trie_array, trie_size)
        except Exception as e:
            logger.warning(f"GPU search failed: {e}")
            return self._search_patterns_cpu(text, patterns)

    def _search_patterns_cpu(self, text: bytes, patterns: Dict[int, bytes]) -> list:
        # pure Python fallback
        matches = []
        for pattern_id, pattern_bytes in patterns.items():
            plen = len(pattern_bytes)
            if plen == 0:
                continue
            for offset in range(len(text) - plen + 1):
                if text[offset:offset+plen] == pattern_bytes:
                    matches.append((offset, pattern_id))
        return matches

    def _build_flat_trie(self, patterns: Dict[int, bytes]):
        # placeholder serialization
        import numpy as np
        return np.zeros(1, dtype=np.int32)

    def get_info(self) -> Dict[str, Any]:
        return {'use_gpu': self.use_gpu, 'search_kernel': self.search_kernel is not None}


class GPUAccelerationEngine:
    def __init__(self):
        self.detector = GPUDetector()
        self.memory_manager = GPUMemoryManager(self.detector)
        self.trie_accelerator = GPUTrieAccelerator(self.detector)

    def gpu_available(self) -> bool:
        return self.detector.gpu_available

    def l6_search(self, text: bytes, patterns: Dict[int, bytes]) -> list:
        return self.trie_accelerator.search_patterns_gpu(text, patterns)

    def l6_search_cpu(self, text: bytes, patterns: Dict[int, bytes]) -> list:
        return self.trie_accelerator._search_patterns_cpu(text, patterns)

    def get_status(self) -> Dict[str, Any]:
        return {'detector': self.detector.get_info(), 'trie': self.trie_accelerator.get_info()}
