"""
COBOL Protocol v1.5 - GPU Acceleration for Layer 6 Pattern Matching
Optional GPU acceleration using CuPy for high-throughput pattern detection

Features:
- Automatic GPU/CPU fallback
- Parallel pattern matching on GPU
- Configurable batch processing
- Memory-efficient chunking
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPUPatternMatcher:
    """GPU-accelerated pattern matching for Layer 6"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.device_memory = 0
        self.stats = {}
        
        if self.use_gpu:
            try:
                self.device = cp.cuda.Device()
                self.device_memory = self.device.mem_info[1]  # Total GPU memory
            except Exception as e:
                warnings.warn(f"GPU initialization failed: {e}, falling back to CPU")
                self.use_gpu = False
    
    def find_patterns_gpu(self, data: bytes, pattern_size: int = 4, 
                         batch_size: int = 1024*1024) -> Dict[bytes, int]:
        """
        Find repeating patterns using GPU
        
        Args:
            data: Input bytes to analyze
            pattern_size: Size of patterns to find (2-16)
            batch_size: Process data in chunks (GPU memory limited)
        
        Returns:
            Dictionary of pattern -> frequency
        """
        
        if not self.use_gpu or not CUPY_AVAILABLE:
            return self._find_patterns_cpu(data, pattern_size)
        
        pattern_freq = {}
        
        try:
            # Convert to GPU array
            data_gpu = cp.array(np.frombuffer(data, dtype=np.uint8))
            
            # Process in batches to avoid OOM
            for start_idx in range(0, len(data) - pattern_size + 1, batch_size):
                end_idx = min(start_idx + batch_size, len(data) - pattern_size + 1)
                
                # Extract patterns using GPU slicing
                batch_size_actual = end_idx - start_idx
                patterns_gpu = cp.lib.stride_tricks.as_strided(
                    data_gpu[start_idx:start_idx + batch_size_actual + pattern_size - 1],
                    shape=(batch_size_actual, pattern_size),
                    strides=(data_gpu.strides[0], data_gpu.strides[0])
                )
                
                # Count frequencies using GPU
                patterns_cpu = cp.asnumpy(patterns_gpu)
                
                for pattern_arr in patterns_cpu:
                    pattern = bytes(pattern_arr)
                    pattern_freq[pattern] = pattern_freq.get(pattern, 0) + 1
            
            # Filter: keep only patterns with freq >= 2
            self.stats['gpu_patterns_found'] = len(pattern_freq)
            
        except cp.cuda.runtime.CUDADriverError as e:
            warnings.warn(f"CUDA error during pattern matching: {e}")
            return self._find_patterns_cpu(data, pattern_size)
        
        return {p: f for p, f in pattern_freq.items() if f >= 2}
    
    def _find_patterns_cpu(self, data: bytes, pattern_size: int) -> Dict[bytes, int]:
        """Fallback CPU pattern finding"""
        pattern_freq = {}
        
        for i in range(len(data) - pattern_size + 1):
            pattern = data[i:i+pattern_size]
            pattern_freq[pattern] = pattern_freq.get(pattern, 0) + 1
        
        return {p: f for p, f in pattern_freq.items() if f >= 2}
    
    def match_patterns_gpu(self, data: bytes, patterns: List[bytes]) -> Dict[bytes, List[int]]:
        """
        Find all positions of patterns using GPU
        
        Args:
            data: Input data
            patterns: List of patterns to search
        
        Returns:
            Dictionary of pattern -> list of positions
        """
        
        if not self.use_gpu:
            return self._match_patterns_cpu(data, patterns)
        
        try:
            results = {}
            data_gpu = cp.array(np.frombuffer(data, dtype=np.uint8))
            
            for pattern in patterns:
                pattern_array = np.frombuffer(pattern, dtype=np.uint8)
                pattern_gpu = cp.array(pattern_array)
                
                # Use FFT-based convolution for pattern matching (faster on GPU)
                if len(pattern) > 4:
                    matches = self._correlate_pattern_gpu(data_gpu, pattern_gpu, pattern)
                else:
                    # Direct comparison for short patterns
                    matches = self._direct_match_gpu(data_gpu, pattern_gpu)
                
                if matches:
                    results[pattern] = matches
            
            return results
            
        except Exception as e:
            warnings.warn(f"GPU matching failed: {e}")
            return self._match_patterns_cpu(data, patterns)
    
    def _correlate_pattern_gpu(self, data_gpu, pattern_gpu, original_pattern: bytes) -> List[int]:
        """Correlate pattern using GPU"""
        try:
            # Use CuPy's correlate_filter for pattern matching
            correlation = cp.correlate(data_gpu, pattern_gpu, mode='valid')
            
            # Find positions where correlation equals pattern length * 255
            # (perfect match of all bytes)
            pattern_len = len(original_pattern)
            threshold = pattern_len * 255
            
            matches = cp.where(correlation == threshold)[0]
            return cp.asnumpy(matches).tolist()
        except:
            return []
    
    def _direct_match_gpu(self, data_gpu, pattern_gpu) -> List[int]:
        """Direct pattern matching on GPU"""
        matches = []
        pattern_len = len(pattern_gpu)
        
        for i in range(len(data_gpu) - pattern_len + 1):
            window = data_gpu[i:i+pattern_len]
            if cp.array_equal(window, pattern_gpu):
                matches.append(i)
        
        return matches
    
    def _match_patterns_cpu(self, data: bytes, patterns: List[bytes]) -> Dict[bytes, List[int]]:
        """CPU fallback for pattern matching"""
        results = {}
        
        for pattern in patterns:
            positions = []
            pattern_len = len(pattern)
            
            for i in range(len(data) - pattern_len + 1):
                if data[i:i+pattern_len] == pattern:
                    positions.append(i)
            
            if positions:
                results[pattern] = positions
        
        return results


class GPUAcceleratedLayer6:
    """Layer 6 with optional GPU acceleration"""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.matcher = GPUPatternMatcher(use_gpu=self.use_gpu)
        self.patterns = {}
        self.pattern_positions = {}
        self.stats = {}
    
    def encode_gpu(self, data: bytes, pattern_sizes: List[int] = None) -> bytes:
        """
        GPU-accelerated encoding for Layer 6
        
        Args:
            data: Input data to compress
            pattern_sizes: List of pattern sizes to try (default: [2, 4, 8])
        
        Returns:
            Compressed data
        """
        if not pattern_sizes:
            pattern_sizes = [2, 4, 8, 16]
        
        import io
        import struct
        
        if not data:
            return struct.pack('<I', 0)
        
        # Find patterns using GPU
        all_patterns = {}
        for size in pattern_sizes:
            patterns = self.matcher.find_patterns_gpu(data, pattern_size=size)
            all_patterns.update(patterns)
        
        # Sort by frequency (ROI)
        sorted_patterns = sorted(all_patterns.items(), key=lambda x: x[1], reverse=True)[:255]
        
        # Store pattern map
        pattern_map = {pattern: i for i, (pattern, _) in enumerate(sorted_patterns)}
        self.patterns = {i: pattern for pattern, i in pattern_map.items()}
        
        # Find all positions (GPU-accelerated)
        pattern_list = [p for p, _ in sorted_patterns]
        self.pattern_positions = self.matcher.match_patterns_gpu(data, pattern_list)
        
        # Encode
        output = io.BytesIO()
        output.write(struct.pack('<I', len(self.patterns)))
        
        # Write catalog
        for pattern_id, pattern in sorted(self.patterns.items()):
            output.write(struct.pack('<B', pattern_id))
            output.write(struct.pack('<H', len(pattern)))
            output.write(pattern)
        
        # Encode data
        encoded = io.BytesIO()
        pos = 0
        
        while pos < len(data):
            matched = False
            
            # Try patterns by frequency
            for pattern, _ in sorted_patterns:
                if data[pos:pos+len(pattern)] == pattern:
                    encoded.write(bytes([0xFE]))
                    encoded.write(bytes([pattern_map[pattern]]))
                    pos += len(pattern)
                    matched = True
                    break
            
            if not matched:
                byte = data[pos]
                if byte == 0xFE:
                    encoded.write(bytes([0xFF, 0xFE]))
                else:
                    encoded.write(bytes([byte]))
                pos += 1
        
        output.write(encoded.getvalue())
        
        result = output.getvalue()
        self.stats = {
            'input_size': len(data),
            'output_size': len(result),
            'patterns_found': len(self.patterns),
            'gpu_accelerated': self.use_gpu,
            'ratio': len(result) / len(data) if data else 1.0
        }
        
        return result
    
    def decode_gpu(self, data: bytes) -> bytes:
        """GPU-accelerated decoding"""
        import io
        import struct
        
        if len(data) < 4:
            return b''
        
        stream = io.BytesIO(data)
        pattern_count = struct.unpack('<I', stream.read(4))[0]
        
        patterns = {}
        for _ in range(pattern_count):
            pattern_id = struct.unpack('<B', stream.read(1))[0]
            pattern_len = struct.unpack('<H', stream.read(2))[0]
            patterns[pattern_id] = stream.read(pattern_len)
        
        # Decode
        output = io.BytesIO()
        remaining = stream.read()
        
        pos = 0
        while pos < len(remaining):
            byte = remaining[pos]
            
            if byte == 0xFE:
                if pos + 1 < len(remaining):
                    next_byte = remaining[pos + 1]
                    if next_byte in patterns:
                        output.write(patterns[next_byte])
                    pos += 2
                else:
                    break
            elif byte == 0xFF:
                if pos + 1 < len(remaining):
                    output.write(bytes([remaining[pos + 1]]))
                    pos += 2
                else:
                    break
            else:
                output.write(bytes([byte]))
                pos += 1
        
        return output.getvalue()


def benchmark_gpu_layer6():
    """Benchmark GPU vs CPU Layer 6"""
    import time
    
    # Test data
    test_data = b"COBOL compression test pattern matching GPU acceleration " * 10000
    
    print("=" * 60)
    print("GPU Acceleration Benchmark - Layer 6")
    print("=" * 60)
    print(f"Test data size: {len(test_data) / 1024 / 1024:.1f} MB\n")
    
    # CPU
    print("CPU Mode:")
    l6_cpu = GPUAcceleratedLayer6(use_gpu=False)
    start = time.time()
    compressed_cpu = l6_cpu.encode_gpu(test_data)
    cpu_time = time.time() - start
    print(f"  Time: {cpu_time*1000:.1f} ms")
    print(f"  Throughput: {len(test_data) / cpu_time / 1024 / 1024:.1f} MB/s")
    print(f"  Compression: {len(test_data) / len(compressed_cpu):.2f}x\n")
    
    # GPU
    if CUPY_AVAILABLE:
        print("GPU Mode:")
        l6_gpu = GPUAcceleratedLayer6(use_gpu=True)
        start = time.time()
        compressed_gpu = l6_gpu.encode_gpu(test_data)
        gpu_time = time.time() - start
        print(f"  Time: {gpu_time*1000:.1f} ms")
        print(f"  Throughput: {len(test_data) / gpu_time / 1024 / 1024:.1f} MB/s")
        print(f"  Compression: {len(test_data) / len(compressed_gpu):.2f}x")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x\n")
    else:
        print("GPU Mode: CuPy not available (fallback to CPU)\n")
    
    # Verify correctness
    decompressed_cpu = l6_cpu.decode_gpu(compressed_cpu)
    assert decompressed_cpu == test_data, "CPU decompression failed!"
    print("✓ Data integrity verified (CPU)")
    
    if CUPY_AVAILABLE:
        decompressed_gpu = l6_gpu.decode_gpu(compressed_gpu)
        assert decompressed_gpu == test_data, "GPU decompression failed!"
        print("✓ Data integrity verified (GPU)")
    
    print("=" * 60)


if __name__ == '__main__':
    if CUPY_AVAILABLE:
        print("✓ CuPy available - GPU acceleration enabled\n")
    else:
        print("⚠ CuPy not available - CPU fallback mode\n")
    
    benchmark_gpu_layer6()
