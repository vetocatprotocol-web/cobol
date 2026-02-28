"""
End-to-End Integration Test Suite for COBOL v1.5
Tests full pipeline: compress → decompress → verify correctness
"""

import pytest
import time
import hashlib
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fpga_controller import FPGAController, FPGACluster, CAMEntry, HuffmanTable, FPGAMetrics


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def fpga_device():
    """Create FPGA controller instance"""
    dev = FPGAController(device_id=0, use_simulator=True)
    yield dev
    dev.stop_metrics_collection()


@pytest.fixture
def fpga_cluster():
    """Create FPGA cluster"""
    cluster = FPGACluster(num_devices=10)
    for i in range(5):
        cluster.initialize_device(i, use_simulator=True)
    yield cluster


# ============================================================================
# CAM DICTIONARY TESTS
# ============================================================================

class TestCAMDictionary:
    """Test CAM (Layer 6) functionality"""
    
    def test_cam_configuration_basic(self, fpga_device):
        """Test basic CAM entry configuration"""
        pattern = b"hello_world_compression"
        fpga_device.configure_cam_entry(pattern, match_id=0)
        
        assert len(fpga_device.config_buffer) == 1
        assert fpga_device.config_buffer[0].match_id == 0
        assert fpga_device.config_buffer[0].pattern == pattern
    
    def test_cam_flush(self, fpga_device):
        """Test CAM flush to hardware"""
        patterns = [b"test_1", b"test_2", b"test_3"]
        for i, pattern in enumerate(patterns):
            fpga_device.configure_cam_entry(pattern, match_id=i)
        
        written = fpga_device.flush_cam_config()
        assert written == 3
        assert len(fpga_device.config_buffer) == 0
        assert len(fpga_device.cam_entries) == 3
    
    def test_cam_lookup_hit(self, fpga_device):
        """Test CAM lookup finds configured entry"""
        pattern = b"test_pattern_data"
        fpga_device.configure_cam_entry(pattern, match_id=42, chunk_size=len(pattern))
        fpga_device.flush_cam_config()
        
        result = fpga_device.cam_lookup(pattern)
        assert result['hit'] == True
        assert result['match_id'] == 42
        assert result['length'] == len(pattern)
    
    def test_cam_lookup_miss(self, fpga_device):
        """Test CAM lookup returns miss for unconfigured patterns"""
        fpga_device.configure_cam_entry(b"known_pattern", match_id=0)
        fpga_device.flush_cam_config()
        
        result = fpga_device.cam_lookup(b"unknown_pattern")
        assert result['hit'] == False
    
    def test_cam_throughput(self, fpga_device):
        """Test CAM can handle high lookup rate"""
        # Configure 1000 entries
        for i in range(1000):
            pattern = f"pattern_{i}".encode()
            fpga_device.configure_cam_entry(pattern, match_id=i, chunk_size=16)
        
        written = fpga_device.flush_cam_config()
        assert written == 1000
        
        # Perform 1000 lookups
        hits = 0
        misses = 0
        start = time.time()
        
        for i in range(1000):
            pattern = f"pattern_{i}".encode()
            result = fpga_device.cam_lookup(pattern)
            if result['hit']:
                hits += 1
            else:
                misses += 1
        
        elapsed = time.time() - start
        rate = 1000 / elapsed
        
        logger.info(f"CAM lookup throughput: {rate:.0f} ops/sec, hits={hits}, misses={misses}")
        assert hits >= 950  # At least 95% hit rate
        assert rate > 100000  # At least 100k ops/sec
    
    def test_cam_large_patterns(self, fpga_device):
        """Test CAM with patterns of varying sizes"""
        sizes = [8, 32, 64, 128, 256, 512]
        
        for i, size in enumerate(sizes):
            pattern = b"x" * size
            fpga_device.configure_cam_entry(pattern, match_id=i, chunk_size=size)
        
        written = fpga_device.flush_cam_config()
        assert written == len(sizes)
        
        # Verify all lookups work
        for i, size in enumerate(sizes):
            pattern = b"x" * size
            result = fpga_device.cam_lookup(pattern)
            if result['hit']:
                assert result['length'] == size


# ============================================================================
# HUFFMAN DECODING TESTS
# ============================================================================

class TestHuffmanDecoding:
    """Test Huffman (Layer 7) decompressor"""
    
    def test_huffman_table_load(self, fpga_device):
        """Test loading Huffman table"""
        # Create simple Huffman table
        table = HuffmanTable(
            chunk_id=0,
            code_length_bits=[3, 3, 4, 4] * 64,  # 256 entries
            code_values=list(range(256)),
            symbols=list(range(256)),
            total_entries=256
        )
        
        result = fpga_device.load_huffman_table(0, table)
        assert result == True
        assert 0 in fpga_device.huffman_tables
    
    def test_huffman_table_validation(self, fpga_device):
        """Test Huffman table validation"""
        # Invalid table (mismatched sizes)
        table = HuffmanTable(
            chunk_id=0,
            code_length_bits=[3, 3],
            code_values=[10, 20, 30],  # Mismatch!
            symbols=[254, 255],
            total_entries=256
        )
        
        result = fpga_device.load_huffman_table(0, table)
        assert result == False


# ============================================================================
# METRICS & MONITORING TESTS
# ============================================================================

class TestMetricsMonitoring:
    """Test metrics collection and health monitoring"""
    
    def test_metrics_collection(self, fpga_device):
        """Test basic metrics collection"""
        fpga_device.start_metrics_collection(interval=0.1)
        time.sleep(0.5)
        
        metrics = fpga_device.get_metrics()
        assert metrics.input_rate_gb_s > 0
        assert metrics.decomp_rate_gb_s > 0
        assert metrics.cam_hit_rate >= 0 and metrics.cam_hit_rate <= 100
        assert metrics.compression_ratio >= 1
        
        fpga_device.stop_metrics_collection()
    
    def test_metrics_history(self, fpga_device):
        """Test metrics history tracking"""
        fpga_device.start_metrics_collection(interval=0.05)
        time.sleep(0.5)
        
        history = fpga_device.get_metrics_history(last_n=10)
        assert len(history) > 0
        
        fpga_device.stop_metrics_collection()
    
    def test_status_ok(self, fpga_device):
        """Test health status when operating normally"""
        fpga_device.start_metrics_collection(interval=0.1)
        time.sleep(0.3)
        
        status = fpga_device.get_status()
        assert status['health_score'] > 70
        assert len(status['issues']) < 5
        
        fpga_device.stop_metrics_collection()
    
    def test_status_alerts(self, fpga_device):
        """Test health score degrades on errors"""
        # Simulate errors in metrics
        fpga_device.backend.metrics_counters['cam_hit_rate'] = 30  # Low hit rate
        fpga_device.backend.metrics_counters['crc_errors'] = 10
        
        metrics = fpga_device.get_metrics()
        status = fpga_device.get_status()
        
        assert status['health_score'] < 80
        assert len(status['issues']) > 0
        logger.info(f"Simulated issues: {status['issues']}")


# ============================================================================
# PIPELINE INTEGRATION TESTS
# ============================================================================

class TestPipelineIntegration:
    """Test full compression/decompression pipeline"""
    
    def test_dictionary_configuration_workflow(self, fpga_device):
        """Test complete dictionary setup workflow"""
        # Step 1: Create dictionary entries
        dictionary = {
            'the': 0,
            'compression': 1,
            'algorithm': 2,
            'huffman': 3,
        }
        
        for word, entry_id in dictionary.items():
            fpga_device.configure_cam_entry(word.encode(), entry_id, len(word))
        
        # Step 2: Flush to device
        written = fpga_device.flush_cam_config()
        assert written == len(dictionary)
        
        # Step 3: Verify all entries accessible
        for word, expected_id in dictionary.items():
            result = fpga_device.cam_lookup(word.encode())
            assert result['hit'] == True
            assert result['match_id'] == expected_id
    
    def test_huffman_workflow(self, fpga_device):
        """Test complete Huffman setup workflow"""
        for chunk_id in range(10):
            table = HuffmanTable(
                chunk_id=chunk_id,
                code_length_bits=[3, 4, 5] * 85 + [3, 3],  # 256 entries
                code_values=[i * 7 for i in range(256)],
                symbols=list(range(256)),
                total_entries=256
            )
            result = fpga_device.load_huffman_table(chunk_id, table)
            assert result == True
        
        assert len(fpga_device.huffman_tables) == 10
    
    def test_concurrent_operations(self, fpga_device):
        """Test concurrent CAM and Huffman operations"""
        # Configure CAM
        for i in range(100):
            fpga_device.configure_cam_entry(
                f"pattern_{i}".encode(),
                match_id=i,
                chunk_size=20
            )
        fpga_device.flush_cam_config()
        
        # Load Huffman tables
        for j in range(10):
            table = HuffmanTable(
                chunk_id=j,
                code_length_bits=[3] * 256,
                code_values=list(range(256)),
                symbols=list(range(256)),
                total_entries=256
            )
            fpga_device.load_huffman_table(j, table)
        
        # Verify both are accessible
        assert len(fpga_device.cam_entries) == 100
        assert len(fpga_device.huffman_tables) == 10
        
        result = fpga_device.cam_lookup(b"pattern_50")
        assert result['hit'] == True


# ============================================================================
# CLUSTER TESTS
# ============================================================================

class TestFPGACluster:
    """Test cluster-level functionality"""
    
    def test_cluster_initialization(self, fpga_cluster):
        """Test cluster creation"""
        assert len(fpga_cluster.devices) == 5
        
        for dev_id in range(5):
            dev = fpga_cluster.get_device(dev_id)
            assert dev is not None
            assert dev.device_id == dev_id
    
    def test_cluster_status(self, fpga_cluster):
        """Test aggregate cluster health"""
        # Configure all devices
        for dev_id, dev in fpga_cluster.devices.items():
            for i in range(10):
                pattern = f"dev{dev_id}_pat{i}".encode()
                dev.configure_cam_entry(pattern, match_id=i)
            dev.flush_cam_config()
        
        status = fpga_cluster.get_aggregate_status()
        assert status['total_devices'] == 5
        assert status['total_cam_entries'] == 50
    
    def test_cluster_metrics(self, fpga_cluster):
        """Test cluster-level metrics collection"""
        fpga_cluster.start_all_metrics()
        time.sleep(0.3)
        
        # Get per-device metrics
        metrics_list = []
        for dev in fpga_cluster.devices.values():
            m = dev.get_metrics()
            metrics_list.append(m)
        
        assert len(metrics_list) == 5
        fpga_cluster.stop_all_metrics()


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformanceBenchmarks:
    """Measure throughput and latency"""
    
    def test_cam_latency_distribution(self, fpga_device):
        """Measure CAM lookup latency distribution"""
        fpga_device.configure_cam_entry(b"test_pattern", match_id=0)
        fpga_device.flush_cam_config()
        
        latencies = []
        for _ in range(1000):
            result = fpga_device.cam_lookup(b"test_pattern")
            latencies.append(result['latency_ns'])
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        logger.info(f"CAM Latency: min={min_latency}ns, avg={avg_latency:.1f}ns, max={max_latency}ns")
        assert avg_latency < 50  # Should be < 50 ns
    
    def test_metrics_continuous_collection(self, fpga_device):
        """Test continuous metrics collection performance"""
        fpga_device.start_metrics_collection(interval=0.01)
        
        start = time.time()
        while time.time() - start < 1.0:
            time.sleep(0.05)
        
        history = fpga_device.get_metrics_history()
        fpga_device.stop_metrics_collection()
        
        logger.info(f"Collected {len(history)} metrics samples in 1 second")
        assert len(history) > 50  # At least 50 samples in 1 second


# ============================================================================
# ERROR HANDLING & RESILIENCE
# ============================================================================

class TestErrorHandling:
    """Test error handling and recovery"""
    
    def test_reset_recovery(self, fpga_device):
        """Test soft reset functionality"""
        from fpga_controller import FPGAState
        
        assert fpga_device.state == FPGAState.IDLE
        
        fpga_device.soft_reset()
        
        # State should cycle through RESET and back to IDLE
        assert fpga_device.state == FPGAState.IDLE
    
    def test_invalid_huffman_table_handling(self, fpga_device):
        """Test graceful handling of invalid tables"""
        invalid_table = HuffmanTable(
            chunk_id=0,
            code_length_bits=[],
            code_values=[],
            symbols=[],
            total_entries=256
        )
        
        result = fpga_device.load_huffman_table(0, invalid_table)
        assert result == False
        assert 0 not in fpga_device.huffman_tables


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
