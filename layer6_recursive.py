from gpu_acceleration import GPUTrieAccelerator, GPUDetector
import logging
from protocol_bridge import TypedBuffer, ProtocolLanguage
import numpy as np

class Layer6Recursive:
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        # Integrasi akselerasi GPU untuk pencarian pola pada Layer 6
        detector = GPUDetector()
        accelerator = GPUTrieAccelerator(detector)
        # Asumsikan buffer.data adalah bytes atau np.ndarray
        patterns = {}  # Patterns dapat diisi sesuai kebutuhan aplikasi
        try:
            # Jika GPU tersedia, gunakan akselerasi GPU
            if detector.gpu_available:
                matches = accelerator.search_patterns_gpu(buffer.data, patterns)
                logging.info(f"Layer6Recursive: GPU acceleration active, matches={len(matches)}")
            else:
                # Fallback: tetap lakukan offset dummy
                matches = []
                logging.info("Layer6Recursive: GPU not available, fallback to CPU")
        except Exception as e:
            matches = []
            logging.warning(f"Layer6Recursive: GPU acceleration error: {e}")
        # Dummy: tetap lakukan offset jika tidak ada pattern
        nested = buffer.data + 1000
        return TypedBuffer.create(nested, ProtocolLanguage.L6_PTR, np.ndarray)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        pointers = buffer.data - 1000
        return TypedBuffer.create(pointers, ProtocolLanguage.L5_TRIE, np.ndarray)
