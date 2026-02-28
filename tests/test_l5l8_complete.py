"""
COBOL Protocol v1.5 - L5-L8 Complete Test Suite
Comprehensive testing for optimized pipeline, GPU acceleration, and federated learning

Test Coverage:
- Layer 5-8 roundtrip integrity
- Performance benchmarks
- GPU acceleration validation
- Federated learning aggregation
- Edge cases and error handling
"""

import pytest
import time
import os
from typing import Tuple
import numpy as np

from l5l8_optimized_pipeline import OptimizedL5L8Pipeline
from layer6_gpu_acceleration import GPUAcceleratedLayer6, GPUPatternMatcher, CUPY_AVAILABLE
from federated_dictionary_learning import (
    LocalDictionary, FederatedPatternAggregator, DistributedDictionaryManager,
    FederationStrategy, DifferentialPrivacy
)


# ============================================================================
# TESTS: Optimized L5-L8 Pipeline
# ============================================================================

class TestOptimizedL5L8Pipeline:
    """Tests for optimized L5-L8 pipeline"""
    
    def test_basic_compression_roundtrip(self):
        """Test basic compression and decompression"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b"The quick brown fox jumps over the lazy dog " * 100
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data, "Roundtrip failed"
        assert len(compressed) < len(test_data), "No compression achieved"
    
    def test_empty_data(self):
        """Test empty input handling"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b""
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_single_byte(self):
        """Test single byte"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b"X"
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_highly_compressible_data(self):
        """Test highly repetitive data"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b"ABC" * 10000  # 30 KB repetitive
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
        ratio = len(test_data) / len(compressed)
        print(f"\nHighly compressible: {ratio:.2f}x")
        assert ratio >= 5.0, f"Compression ratio {ratio:.2f}x too low"
    
    def test_random_data(self):
        """Test incompressible random data"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = os.urandom(10000)
        
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data
    
    def test_cobol_like_data(self):
        """Test COBOL program-like data"""
        cobol_data = b"""
        IDENTIFICATION DIVISION.
        PROGRAM-ID. TEST-PROGRAM.
        DATA DIVISION.
        WORKING-STORAGE SECTION.
        01 WS-COUNTER PIC 9(5) VALUE 0.
        PROCEDURE DIVISION.
        MAIN-PROCEDURE.
            PERFORM UNTIL DONE.
               ADD 1 TO WS-COUNTER.
            END-PERFORM.
            STOP RUN.
        """ * 100
        
        pipeline = OptimizedL5L8Pipeline()
        compressed = pipeline.compress(cobol_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == cobol_data
        ratio = len(cobol_data) / len(compressed)
        print(f"COBOL-like data: {ratio:.2f}x")
        assert ratio >= 2.0
    
    def test_compression_statistics(self):
        """Test statistics collection"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b"Test data " * 1000
        
        pipeline.compress(test_data)
        stats = pipeline.get_stats()
        
        assert stats['input_size'] == len(test_data)
        assert stats['output_size'] > 0
        assert 'compression_ratio' in stats
        assert 'throughput_mbps' in stats
    
    def test_large_data_5mb(self):
        """Test 5 MB data"""
        pipeline = OptimizedL5L8Pipeline()
        test_data = b"COBOL compression " * (5 * 1024 * 1024 // 18)
        
        start = time.time()
        compressed = pipeline.compress(test_data)
        compress_time = time.time() - start
        
        decompressed = pipeline.decompress(compressed)
        assert decompressed == test_data
        
        throughput = len(test_data) / compress_time / 1024 / 1024
        print(f"5 MB compression: {throughput:.1f} MB/s")
        assert throughput > 10, f"Throughput {throughput:.1f} MB/s too low"
    
    def test_mixed_data_types(self):
        """Test mixed text and binary data"""
        text_part = b"COBOL Program Data " * 100
        binary_part = os.urandom(1000)
        test_data = text_part + binary_part
        
        pipeline = OptimizedL5L8Pipeline()
        compressed = pipeline.compress(test_data)
        decompressed = pipeline.decompress(compressed)
        
        assert decompressed == test_data


# ============================================================================
# TESTS: GPU Acceleration (Layer 6)
# ============================================================================

class TestGPUAcceleration:
    """Tests for GPU acceleration in Layer 6"""
    
    def test_gpu_pattern_matcher_cpu(self):
        """Test CPU fallback pattern matching"""
        matcher = GPUPatternMatcher(use_gpu=False)
        test_data = b"ABCABC" * 100
        
        patterns = matcher.find_patterns_gpu(test_data, pattern_size=3)
        
        assert b"ABC" in patterns
        assert patterns[b"ABC"] >= 100
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_pattern_matcher_gpu(self):
        """Test GPU pattern matching"""
        matcher = GPUPatternMatcher(use_gpu=True)
        test_data = b"PATTERN PATTERN " * 500
        
        patterns = matcher.find_patterns_gpu(test_data, pattern_size=8)
        
        assert len(patterns) > 0
    
    def test_gpu_layer6_compression_cpu(self):
        """Test Layer 6 GPU mode with CPU fallback"""
        layer6 = GPUAcceleratedLayer6(use_gpu=False)
        test_data = b"Test pattern " * 1000
        
        compressed = layer6.encode_gpu(test_data)
        decompressed = layer6.decode_gpu(compressed)
        
        assert decompressed == test_data
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_layer6_compression_gpu(self):
        """Test Layer 6 with GPU acceleration"""
        layer6 = GPUAcceleratedLayer6(use_gpu=True)
        test_data = b"GPU acceleration test pattern " * 500
        
        compressed = layer6.encode_gpu(test_data)
        decompressed = layer6.decode_gpu(compressed)
        
        assert decompressed == test_data
        assert layer6.stats['gpu_accelerated']
    
    def test_gpu_multiple_pattern_sizes(self):
        """Test multiple pattern sizes"""
        layer6 = GPUAcceleratedLayer6(use_gpu=False)
        test_data = b"AB" * 100 + b"ABCD" * 100 + b"ABCDEFGH" * 100
        
        compressed = layer6.encode_gpu(
            test_data,
            pattern_sizes=[2, 4, 8]
        )
        decompressed = layer6.decode_gpu(compressed)
        
        assert decompressed == test_data


# ============================================================================
# TESTS: Federated Learning
# ============================================================================

class TestFederatedLearning:
    """Tests for federated dictionary learning"""
    
    def test_local_dictionary_creation(self):
        """Test local dictionary creation"""
        local_dict = LocalDictionary(node_id="test_node")
        
        local_dict.add_pattern(b"ABC", frequency=5)
        local_dict.add_pattern(b"DEF", frequency=3)
        
        assert len(local_dict.patterns) == 2
        assert local_dict.patterns[b"ABC"].frequency == 5
    
    def test_local_dictionary_metrics(self):
        """Test local dictionary entropy and ROI calculation"""
        local_dict = LocalDictionary(node_id="test_node")
        
        local_dict.add_pattern(b"ABC", frequency=100)
        local_dict.add_pattern(b"DEF", frequency=50)
        local_dict.add_pattern(b"GHI", frequency=25)
        
        local_dict.calculate_entropy()
        local_dict.calculate_roi()
        
        # Check ROI calculation
        abc_info = local_dict.patterns[b"ABC"]
        assert abc_info.roi > 0
    
    def test_local_dictionary_serialization(self):
        """Test JSON serialization"""
        local_dict = LocalDictionary(node_id="test_node")
        local_dict.add_pattern(b"TEST", frequency=42)
        
        json_str = local_dict.to_json()
        restored = LocalDictionary.from_json(json_str)
        
        assert restored.node_id == "test_node"
        assert restored.patterns[b"TEST"].frequency == 42
    
    def test_frequency_weighted_aggregation(self):
        """Test frequency-weighted aggregation"""
        # Create local dictionaries
        node1 = LocalDictionary(node_id="node1")
        node1.add_pattern(b"ABC", frequency=100)
        node1.add_pattern(b"DEF", frequency=50)
        
        node2 = LocalDictionary(node_id="node2")
        node2.add_pattern(b"ABC", frequency=80)
        node2.add_pattern(b"GHI", frequency=60)
        
        # Aggregate
        aggregator = FederatedPatternAggregator(
            FederationStrategy.FREQUENCY_WEIGHTED
        )
        result = aggregator.aggregate([node1, node2])
        
        assert b"ABC" in result
        assert result[b"ABC"].frequency == 180
    
    def test_entropy_based_aggregation(self):
        """Test entropy-based aggregation"""
        node1 = LocalDictionary(node_id="node1")
        node1.add_pattern(b"A", frequency=1000)
        node1.calculate_entropy()
        
        node2 = LocalDictionary(node_id="node2")
        node2.add_pattern(b"B", frequency=500)
        node2.calculate_entropy()
        
        aggregator = FederatedPatternAggregator(
            FederationStrategy.ENTROPY_BASED
        )
        result = aggregator.aggregate([node1, node2])
        
        assert len(result) > 0
    
    def test_consensus_aggregation(self):
        """Test consensus-based aggregation"""
        nodes = [
            LocalDictionary(node_id=f"node{i}") for i in range(5)
        ]
        
        # All nodes have ABC
        for node in nodes:
            node.add_pattern(b"ABC", frequency=50)
        
        # Only 2 nodes have DEF
        nodes[0].add_pattern(b"DEF", frequency=30)
        nodes[1].add_pattern(b"DEF", frequency=30)
        
        aggregator = FederatedPatternAggregator(
            FederationStrategy.CONSENSUS
        )
        aggregator.consensus_threshold = 0.8  # 80% of nodes
        result = aggregator.aggregate(nodes)
        
        # ABC should be in result (100% consensus)
        assert b"ABC" in result
        # DEF should not be (only 40% consensus)
        assert b"DEF" not in result
    
    def test_differential_privacy(self):
        """Test differential privacy noise addition"""
        privacy = DifferentialPrivacy(epsilon=0.1)
        
        from federated_dictionary_learning import PatternInfo
        info = PatternInfo(pattern=b"TEST", frequency=1000)
        
        noisy = privacy.add_laplace_noise(info)
        
        assert noisy.frequency != info.frequency
        assert noisy.frequency >= 0
    
    def test_anonymous_dictionary(self):
        """Test dictionary anonymization"""
        local_dict = LocalDictionary(node_id="original_node")
        local_dict.add_pattern(b"ABC", frequency=100)
        local_dict.add_pattern(b"DEF", frequency=50)
        
        privacy = DifferentialPrivacy(epsilon=0.5)
        anon = privacy.anonymize_dictionary(local_dict)
        
        assert anon.node_id == "anonymized"
        assert len(anon.patterns) == len(local_dict.patterns)
    
    def test_distributed_manager_basic(self):
        """Test distributed dictionary manager"""
        manager = DistributedDictionaryManager()
        
        # Register nodes
        manager.register_node("node1")
        manager.register_node("node2")
        
        # Update local dictionaries
        manager.update_local_dictionary("node1", b"COBOL PROGRAM " * 100)
        manager.update_local_dictionary("node2", b"DATA DIVISION " * 100)
        
        # Federated aggregation
        global_dict = manager.federated_aggregation()
        
        assert len(global_dict) > 0
    
    def test_distributed_manager_statistics(self):
        """Test statistics from distributed manager"""
        manager = DistributedDictionaryManager()
        
        manager.register_node("node1")
        manager.update_local_dictionary("node1", b"test data " * 1000)
        
        stats = manager.get_node_statistics("node1")
        assert stats['node_id'] == "node1"
        assert stats['data_processed'] > 0
        assert 'patterns' in stats
    
    def test_aggregation_strategies_comparison(self):
        """Compare different aggregation strategies"""
        nodes = []
        for i in range(3):
            node = LocalDictionary(node_id=f"node{i}")
            # Add different patterns
            node.add_pattern(b"COMMON", frequency=100)
            node.add_pattern(b"PATTERN" * (i+1), frequency=20 * (i+1))
            nodes.append(node)
        
        strategies = [
            FederationStrategy.FREQUENCY_WEIGHTED,
            FederationStrategy.ENTROPY_BASED,
            FederationStrategy.CONSENSUS,
            FederationStrategy.ADAPTIVE
        ]
        
        results = {}
        for strategy in strategies:
            aggregator = FederatedPatternAggregator(strategy)
            result = aggregator.aggregate(nodes, max_patterns=10)
            results[strategy.name] = len(result)
        
        print(f"\nAggregation results:")
        for strategy_name, count in results.items():
            print(f"  {strategy_name}: {count} patterns")


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks"""
    
    def test_pipeline_throughput_benchmark(self):
        """Benchmark full L5-L8 pipeline"""
        pipeline = OptimizedL5L8Pipeline()
        
        test_sizes = [1, 5, 10]  # MB
        
        print("\n" + "=" * 60)
        print("L5-L8 Pipeline Throughput Benchmark")
        print("=" * 60)
        
        for size_mb in test_sizes:
            test_data = b"COBOL compression benchmark " * (1024 * 1024 * size_mb // 28)
            
            start = time.time()
            compressed = pipeline.compress(test_data)
            elapsed = time.time() - start
            
            throughput = size_mb / elapsed
            ratio = len(test_data) / len(compressed)
            
            print(f"\n{size_mb} MB:")
            print(f"  Time: {elapsed*1000:.1f} ms")
            print(f"  Throughput: {throughput:.1f} MB/s")
            print(f"  Ratio: {ratio:.2f}x")
    
    @pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
    def test_gpu_cpu_comparison(self):
        """Compare GPU vs CPU performance"""
        test_data = b"GPU vs CPU pattern matching test data " * 100000
        
        print("\n" + "=" * 60)
        print("GPU vs CPU Performance")
        print("=" * 60)
        
        # CPU
        l6_cpu = GPUAcceleratedLayer6(use_gpu=False)
        start = time.time()
        compressed_cpu = l6_cpu.encode_gpu(test_data)
        cpu_time = time.time() - start
        
        print(f"\nCPU Mode: {cpu_time*1000:.1f} ms")
        print(f"  Throughput: {len(test_data) / cpu_time / 1024 / 1024:.1f} MB/s")
        
        # GPU
        l6_gpu = GPUAcceleratedLayer6(use_gpu=True)
        start = time.time()
        compressed_gpu = l6_gpu.encode_gpu(test_data)
        gpu_time = time.time() - start
        
        print(f"\nGPU Mode: {gpu_time*1000:.1f} ms")
        print(f"  Throughput: {len(test_data) / gpu_time / 1024 / 1024:.1f} MB/s")
        print(f"  Speedup: {cpu_time / gpu_time:.2f}x")


# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
