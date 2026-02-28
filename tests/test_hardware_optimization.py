"""
COBOL Protocol v1.5: Hardware Optimization Test Suite
=====================================================

Comprehensive tests for:
1. Hardware detection and capability mapping
2. Per-layer hardware-optimized implementations
3. Adaptive pipeline and strategy selection
4. Health monitoring and circuit breaker
5. Fallback mechanisms
6. End-to-end compression/decompression
"""

import pytest
import numpy as np
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)

from hardware_abstraction_layer import (
    HardwareDetector, HardwareContext, HardwareType, ComputeCapability,
    OptimizationStrategy, HardwareProfile, CPUCapabilities, get_hardware_context
)
from hardware_optimized_layers import (
    HardwareOptimizedLayer1, HardwareOptimizedLayer2, HardwareOptimizedLayer3,
    HardwareOptimizedLayer4, HardwareOptimizedLayer5, HardwareOptimizedLayer6,
    HardwareOptimizedLayer7, HardwareOptimizedLayer8, HardwareOptimizedPipeline
)
from adaptive_pipeline import (
    AdaptivePipeline, LayerHealthMonitor, CircuitBreaker, 
    HealthStatus, StabilityManager
)


# ============================================================================
# HARDWARE DETECTION TESTS
# ============================================================================


class TestHardwareDetection:
    """Test hardware detection capabilities."""
    
    def test_detector_initialization(self):
        """Test HardwareDetector initialization."""
        detector = HardwareDetector()
        assert detector is not None
        assert isinstance(detector.profiles, list)
    
    def test_cpu_detection(self):
        """Test CPU detection."""
        detector = HardwareDetector()
        cpu_profile = detector._detect_cpu()
        
        assert cpu_profile is not None
        assert cpu_profile.hardware_type == HardwareType.CPU
        assert cpu_profile.cpu_caps is not None
        assert cpu_profile.cpu_caps.cores > 0
        assert cpu_profile.cpu_caps.frequency_ghz > 0
    
    def test_detect_all(self):
        """Test detecting all hardware."""
        detector = HardwareDetector()
        profiles = detector.detect_all()
        
        assert len(profiles) > 0
        assert any(p.hardware_type == HardwareType.CPU for p in profiles)
    
    def test_primary_device_selection(self):
        """Test primary device selection."""
        detector = HardwareDetector()
        detector.detect_all()
        
        assert detector.primary_device is not None
        # Primary should be best-scoring device
        scores = [p.score() for p in detector.profiles]
        assert detector.primary_device.score() == max(scores)
    
    def test_hardware_context_singleton(self):
        """Test HardwareContext singleton pattern."""
        context1 = get_hardware_context()
        context2 = get_hardware_context()
        
        assert context1 is context2
    
    def test_hardware_context_summary(self):
        """Test hardware context summary."""
        context = get_hardware_context()
        summary = context.summary()
        
        assert "HARDWARE CONTEXT SUMMARY" in summary
        assert "Primary Device" in summary
        assert len(context.profiles) > 0


# ============================================================================
# LAYER TESTS (1-8)
# ============================================================================


class TestLayer1:
    """Test Layer 1: Semantic Tokenization."""
    
    def test_layer1_encode(self):
        """Test Layer 1 encoding."""
        layer = HardwareOptimizedLayer1()
        data = b"test data for layer 1"
        
        encoded = layer.encode(data)
        
        assert encoded is not None
        assert isinstance(encoded, np.ndarray)
        assert len(encoded) > 0
    
    def test_layer1_decode(self):
        """Test Layer 1 decoding."""
        layer = HardwareOptimizedLayer1()
        original = b"test data"
        
        encoded = layer.encode(original)
        decoded = layer.decode(encoded)
        
        assert decoded is not None
        assert len(decoded) == len(original)
    
    def test_layer1_roundtrip(self):
        """Test Layer 1 encode-decode roundtrip."""
        layer = HardwareOptimizedLayer1()
        test_data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        
        encoded = layer.encode(test_data)
        decoded = layer.decode(encoded)
        
        assert bytes(decoded) == test_data
    
    def test_layer1_performance(self):
        """Test Layer 1 performance metrics."""
        layer = HardwareOptimizedLayer1()
        large_data = b"x" * (1024 * 1024)  # 1 MB
        
        layer.encode(large_data)
        
        stats = layer.get_stats()
        assert stats["calls"] == 1
        assert stats["bytes"] == len(large_data)


class TestLayer3:
    """Test Layer 3: Delta Compression."""
    
    def test_layer3_encode(self):
        """Test Layer 3 encoding."""
        layer = HardwareOptimizedLayer3()
        data = b"\x01\x02\x03\x04\x05"
        
        encoded = layer.encode(data)
        
        assert encoded is not None
        assert isinstance(encoded, np.ndarray)
    
    def test_layer3_decode(self):
        """Test Layer 3 decoding."""
        layer = HardwareOptimizedLayer3()
        original = b"\x10\x20\x30\x40"
        
        encoded = layer.encode(original)
        decoded = layer.decode(encoded)
        
        assert len(decoded) == len(original)
    
    def test_layer3_delta_property(self):
        """Test Layer 3 delta encoding property."""
        layer = HardwareOptimizedLayer3()
        data = b"\x00\x10\x10\x20\x20"  # Sequential deltas
        
        encoded = layer.encode(data)
        
        # First byte should equal first byte of original
        assert encoded[0] == data[0]


class TestLayer5:
    """Test Layer 5: Adaptive Framework."""
    
    def test_layer5_entropy_computation(self):
        """Test entropy computation."""
        layer = HardwareOptimizedLayer5()
        
        # Low entropy (repeated pattern)
        low_entropy_data = b"AAAA" * 250
        entropy_low = layer._compute_entropy(np.frombuffer(low_entropy_data, dtype=np.uint8))
        
        # High entropy (random data)
        high_entropy_data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        entropy_high = layer._compute_entropy(np.frombuffer(high_entropy_data, dtype=np.uint8))
        
        assert entropy_low < entropy_high
    
    def test_layer5_skip_expensive_layers(self):
        """Test expensive layer skipping."""
        layer = HardwareOptimizedLayer5()
        
        # Already compressed (high entropy)
        random_data = np.random.randint(0, 256, 1000, dtype=np.uint8)
        should_skip = layer.should_skip_expensive_layers(random_data)
        
        assert isinstance(should_skip, bool)
    
    def test_layer5_encode_includes_metadata(self):
        """Test Layer 5 includes entropy metadata."""
        layer = HardwareOptimizedLayer5()
        data = b"test"
        
        encoded = layer.encode(data)
        
        # Should be original data + 1 byte metadata
        assert len(encoded) >= len(data)


class TestLayer8:
    """Test Layer 8: Final Hardening."""
    
    def test_layer8_sha256_verification(self):
        """Test SHA-256 hash verification."""
        layer = HardwareOptimizedLayer8()
        data = b"test data for layer 8"
        
        encoded = layer.encode(data)
        
        # Should contain 32-byte hash + data
        assert len(encoded) >= len(data) + 32
    
    def test_layer8_hash_integrity(self):
        """Test hash integrity checking."""
        layer = HardwareOptimizedLayer8()
        original = b"integrity test"
        
        encoded = layer.encode(original)
        decoded = layer.decode(encoded)
        
        assert bytes(decoded) == original
    
    def test_layer8_decode_corrupted_hash(self):
        """Test decoder with corrupted hash."""
        layer = HardwareOptimizedLayer8()
        data = b"test"
        
        encoded = layer.encode(data)
        
        # Corrupt the hash
        corrupted = b"CORRUPTED" + encoded[9:]
        decoded = layer.decode(corrupted)
        
        # Should still decode but log warning
        assert len(decoded) > 0


# ============================================================================
# PIPELINE TESTS
# ============================================================================


class TestHardwareOptimizedPipeline:
    """Test unified pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = HardwareOptimizedPipeline()
        
        assert pipeline is not None
        assert len(pipeline.layers) == 8
        assert pipeline.hw_context is not None
    
    def test_pipeline_compress_decompress(self):
        """Test full compression/decompression cycle."""
        pipeline = HardwareOptimizedPipeline()
        original = b"COBOL" * 100
        
        compressed = pipeline.compress(original)
        
        assert len(compressed) > 0
        # Compression should reduce size (for repetitive data)
        assert len(compressed) <= len(original) * 1.1  # Allow 10% expansion
    
    def test_pipeline_with_various_data(self):
        """Test pipeline with different data types."""
        pipeline = HardwareOptimizedPipeline()
        
        test_cases = [
            b"A" * 1000,              # Repetitive
            bytes(range(256)) * 4,    # Sequential
            np.random.bytes(1000),    # Random
            b"",                       # Empty
        ]
        
        for data in test_cases:
            if len(data) > 0:
                compressed = pipeline.compress(data)
                assert len(compressed) > 0
    
    def test_pipeline_statistics(self):
        """Test pipeline statistics collection."""
        pipeline = HardwareOptimizedPipeline()
        original = b"test" * 100
        
        pipeline.compress(original)
        
        stats = pipeline.get_compression_stats()
        
        assert len(stats) == 8
        for layer_num, layer_stats in stats.items():
            assert "calls" in layer_stats
            assert "bytes" in layer_stats


# ============================================================================
# ADAPTIVE PIPELINE & MONITORING TESTS
# ============================================================================


class TestLayerHealthMonitor:
    """Test layer health monitoring."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        layer = HardwareOptimizedLayer1()
        monitor = LayerHealthMonitor(layer)
        
        assert monitor is not None
        assert monitor.metrics.success_rate == 1.0
    
    def test_monitor_record_success(self):
        """Test recording successful operations."""
        layer = HardwareOptimizedLayer1()
        monitor = LayerHealthMonitor(layer)
        
        monitor.record_encode(10.0, 1000, success=True)
        
        assert monitor.metrics.request_count == 1
        assert monitor.metrics.error_count == 0
    
    def test_monitor_record_failure(self):
        """Test recording failed operations."""
        layer = HardwareOptimizedLayer1()
        monitor = LayerHealthMonitor(layer)
        
        monitor.record_encode(5.0, 0, success=False)
        
        assert monitor.metrics.request_count == 0
        assert monitor.metrics.error_count == 1
    
    def test_health_score_calculation(self):
        """Test health score calculation."""
        layer = HardwareOptimizedLayer1()
        monitor = LayerHealthMonitor(layer)
        
        # Healthy scenario
        monitor.record_encode(10.0, 1000, success=True)
        score = monitor.get_health_score()
        assert score > 80
        
        # Add failures
        for _ in range(5):
            monitor.record_encode(5.0, 0, success=False)
        
        score = monitor.get_health_score()
        assert score < 80


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker("test", failure_threshold=3)
        
        assert breaker.state.value == "closed"
        assert breaker.is_available()
    
    def test_circuit_breaker_state_transitions(self):
        """Test state transitions."""
        breaker = CircuitBreaker("test", failure_threshold=2)
        
        # Record successes - stay closed
        breaker.record_success()
        assert breaker.is_available()
        
        # Record failures - transition to open
        breaker.record_failure()
        breaker.record_failure()
        
        assert not breaker.is_available()
    
    def test_circuit_breaker_recovery(self):
        """Test recovery mechanism."""
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.1)
        
        breaker.record_failure()
        assert not breaker.is_available()
        
        # Wait for recovery timeout
        import time
        time.sleep(0.2)
        
        # Should enter half-open
        is_avail = breaker.is_available()
        assert is_avail  # Should be available for testing
        
        # Success in half-open returns to closed
        breaker.record_success()
        assert breaker.is_available()


class TestAdaptivePipeline:
    """Test adaptive pipeline with monitoring."""
    
    def test_adaptive_pipeline_initialization(self):
        """Test adaptive pipeline initialization."""
        pipeline = AdaptivePipeline()
        
        assert pipeline is not None
        assert len(pipeline.monitors) == 8
    
    def test_compress_with_monitoring(self):
        """Test compression with health monitoring."""
        pipeline = AdaptivePipeline()
        data = b"test" * 100
        
        compressed, metadata = pipeline.compress_with_monitoring(data)
        
        assert len(compressed) > 0
        assert "start_time" in metadata
        assert "per_layer_stats" in metadata
        assert "total_time_ms" in metadata
    
    def test_system_health_report(self):
        """Test system health reporting."""
        pipeline = AdaptivePipeline()
        
        # Do some work
        data = b"x" * 10000
        pipeline.compress_with_monitoring(data)
        
        health = pipeline.get_system_health()
        
        assert "overall_score" in health
        assert "layer_scores" in health
        assert "layer_statuses" in health
        assert len(health["layer_scores"]) == 8
    
    def test_optimization_hints_generation(self):
        """Test optimization hint generation."""
        pipeline = AdaptivePipeline()
        
        # Trigger some operations
        for i in range(10):
            data = b"data" * 100
            pipeline.compress_with_monitoring(data)
        
        health = pipeline.get_system_health()
        
        # Should have hints or empty list
        assert isinstance(health["optimization_hints"], list)
    
    def test_adaptive_skip_expensive_layers(self):
        """Test adaptive layer skipping."""
        pipeline = AdaptivePipeline()
        
        # Random data (high entropy - no skipping)
        random_data = bytes(np.random.randint(0, 256, 1000, dtype=np.uint8))
        compressed, metadata = pipeline.compress_with_monitoring(random_data, adaptive=True)
        
        assert len(compressed) > 0


class TestStabilityManager:
    """Test stability management."""
    
    def test_stability_manager_initialization(self):
        """Test stability manager initialization."""
        pipeline = AdaptivePipeline()
        manager = StabilityManager(pipeline)
        
        assert manager is not None
        assert manager.consecutive_failures == 0
    
    def test_health_check(self):
        """Test health checking."""
        pipeline = AdaptivePipeline()
        manager = StabilityManager(pipeline)
        
        # First check should pass (interval not elapsed)
        result = manager.check_health()
        assert isinstance(result, bool)
    
    def test_recovery_recommendation(self):
        """Test recovery action recommendation."""
        pipeline = AdaptivePipeline()
        manager = StabilityManager(pipeline)
        
        action = manager.get_recovery_action()
        
        # Action can be None or callable
        assert action is None or callable(action)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """End-to-end integration tests."""
    
    def test_full_pipeline_with_monitoring_and_stability(self):
        """Test full pipeline with all features enabled."""
        pipeline = AdaptivePipeline()
        stability = StabilityManager(pipeline)
        
        # Compress with monitoring
        test_data = b"COBOL" * 1000
        compressed, comp_metadata = pipeline.compress_with_monitoring(test_data)
        
        assert len(compressed) > 0
        
        # Check system health
        health = pipeline.get_system_health()
        assert health["overall_score"] >= 0
        
        # Check stability
        is_stable = stability.check_health()
        assert isinstance(is_stable, bool)
    
    def test_multiple_compression_cycles(self):
        """Test multiple compression cycles with monitoring."""
        pipeline = AdaptivePipeline()
        
        test_cases = [
            b"A" * 1000,
            b"COBOL" * 200,
            bytes(range(256)),
        ]
        
        for data in test_cases:
            compressed, _ = pipeline.compress_with_monitoring(data)
            assert len(compressed) > 0
        
        # Metrics should accumulate
        detailed = pipeline.get_detailed_metrics()
        assert len(detailed) == 8
        
        # Check that work was recorded
        for layer_num, info in detailed.items():
            assert "metrics" in info
            assert "health_status" in info
    
    def test_hardware_adaptation(self):
        """Test hardware-aware adaptation."""
        context = get_hardware_context()
        
        # Get strategies for each layer
        strategies = context.get_all_layer_strategies()
        
        assert len(strategies) == 8
        for layer, strategy in strategies.items():
            assert isinstance(strategy, OptimizationStrategy)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
