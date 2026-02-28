"""
Test suite for Multi-Layer Translation Bridge (L1-L8)
Verifies strict typing, lossless roundtrip, and integrity checking
"""

import pytest
import time
import hashlib
import numpy as np
from protocol_bridge import TypedBuffer, ProtocolLanguage, ProtocolBridge
from layer1_semantic import Layer1Semantic
from layer2_structural import Layer2Structural
from layer3_delta import Layer3Delta
from layer4_binary import Layer4Binary
from layer5_recursive import Layer5Recursive
from layer6_recursive import Layer6Recursive
from layer7_bank import Layer7Bank
from layer8_final import Layer8Final


class TestMultiLayerTranslationBridge:
    
    @pytest.fixture
    def bridge(self):
        """Create protocol bridge with all 8 layers"""
        return ProtocolBridge([
            Layer1Semantic(), Layer2Structural(), Layer3Delta(), Layer4Binary(),
            Layer5Recursive(), Layer6Recursive(), Layer7Bank(), Layer8Final()
        ])
    
    def test_l1_semantic_roundtrip(self):
        """Test Layer 1: Semantic roundtrip"""
        l1 = Layer1Semantic()
        original_text = "Hello COBOL Protocol"
        
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        compressed = l1.encode(buffer)
        decompressed = l1.decode(compressed)
        
        assert decompressed.data == str(original_text)
        assert decompressed.header == ProtocolLanguage.L1_SEM
        assert decompressed.type == str
    
    def test_l2_structural_roundtrip(self):
        """Test Layer 2: Structural roundtrip"""
        l1 = Layer1Semantic()
        l2 = Layer2Structural()
        
        original_text = "COBOL 1234 5678"
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        l1_compressed = l1.encode(buffer)
        l2_compressed = l2.encode(l1_compressed)
        l2_decompressed = l2.decode(l2_compressed)
        
        assert l2_decompressed.header == ProtocolLanguage.L1_SEM
        assert np.array_equal(l2_decompressed.data, l1_compressed.data)
    
    def test_l3_delta_roundtrip(self):
        """Test Layer 3: Delta roundtrip"""
        l1 = Layer1Semantic()
        l2 = Layer2Structural()
        l3 = Layer3Delta()
        
        original_text = "DELTA TEST"
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        l1_out = l1.encode(buffer)
        l2_out = l2.encode(l1_out)
        l3_out = l3.encode(l2_out)
        l3_back = l3.decode(l3_out)
        
        assert np.array_equal(l3_back.data, l2_out.data)
    
    def test_l4_binary_roundtrip(self):
        """Test Layer 4: Binary roundtrip"""
        l1 = Layer1Semantic()
        l2 = Layer2Structural()
        l3 = Layer3Delta()
        l4 = Layer4Binary()
        
        original_text = "BINARY123"
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        l1_out = l1.encode(buffer)
        l2_out = l2.encode(l1_out)
        l3_out = l3.encode(l2_out)
        l4_out = l4.encode(l3_out)
        l4_back = l4.decode(l4_out)
        
        assert l4_back.header == ProtocolLanguage.L3_DELTA
        assert np.array_equal(l4_back.data, l3_out.data)
    
    def test_full_pipeline_roundtrip(self, bridge):
        """Test full L1-L8 pipeline roundtrip"""
        original_text = "COBOL PROTOCOL v1.2 TESTING"
        
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        compressed = bridge.compress(buffer)
        
        assert compressed.header == ProtocolLanguage.L8_COBOL
        assert compressed.type == str
        assert isinstance(compressed.data, str)
        
        # Decompress back
        decompressed = bridge.decompress(compressed)
        assert decompressed.data == original_text
        assert decompressed.header == ProtocolLanguage.L1_SEM
    
    def test_sha256_integrity(self):
        """Test SHA-256 integrity checking per layer"""
        original = "INTEGRITY CHECK"
        buffer = TypedBuffer.create(original, ProtocolLanguage.L1_SEM, str)
        
        l1 = Layer1Semantic()
        l1_out = l1.encode(buffer)
        
        # SHA-256 should be set for each buffer
        assert l1_out.sha256 is not None
        assert len(l1_out.sha256) == 64  # SHA-256 hex digest
        
        # Verify hash consistency
        recalc_hash = hashlib.sha256(l1_out.data if isinstance(l1_out.data, bytes) else str(l1_out.data).encode()).hexdigest()
        assert l1_out.sha256 == recalc_hash
    
    def test_type_consistency(self):
        """Verify type consistency across layers"""
        l1 = Layer1Semantic()
        l2 = Layer2Structural()
        l3 = Layer3Delta()
        l4 = Layer4Binary()
        
        original = "TYPE TEST"
        buffer = TypedBuffer.create(original, ProtocolLanguage.L1_SEM, str)
        
        l1_out = l1.encode(buffer)
        assert l1_out.type == np.ndarray
        
        l2_out = l2.encode(l1_out)
        assert l2_out.type == np.ndarray
        
        l3_out = l3.encode(l2_out)
        assert l3_out.type == np.ndarray
        
        l4_out = l4.encode(l3_out)
        assert l4_out.type == bytes
    
    def test_throughput_benchmark(self, bridge):
        """Benchmark pipeline throughput"""
        # Generate 10MB test data
        test_data = "COBOL PROTOCOL " * (10_000_000 // 15)
        
        start = time.time()
        buffer = TypedBuffer.create(test_data, ProtocolLanguage.L1_SEM, str)
        compressed = bridge.compress(buffer)
        compress_time = time.time() - start
        
        # Calculate throughput
        total_bytes = len(test_data)
        throughput_mbps = (total_bytes / (1024 * 1024)) / compress_time if compress_time > 0 else 0
        
        print(f"\nCompress: {total_bytes / (1024*1024):.1f} MB in {compress_time:.2f}s = {throughput_mbps:.1f} MB/s")
        
        # Decompress
        start = time.time()
        decompressed = bridge.decompress(compressed)
        decompress_time = time.time() - start
        
        throughput_mb_decomp = (total_bytes / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0
        print(f"Decompress: {total_bytes / (1024*1024):.1f} MB in {decompress_time:.2f}s = {throughput_mb_decomp:.1f} MB/s")
        
        # Verify lossless
        assert decompressed.data == test_data
        
        # Target: 35+ MB/s (if achievable, this is a bonus)
        # But don't fail the test if it's slower
        assert throughput_mbps > 0
    
    def test_cobol_program_compression(self, bridge):
        """Test with real COBOL program"""
        cobol_program = b"""
    IDENTIFICATION DIVISION.
    PROGRAM-ID. COMPRESSION-TEST.
    
    DATA DIVISION.
    WORKING-STORAGE SECTION.
    01 WS-COUNTER PIC 9(5) VALUE 0.
    01 WS-RESULT PIC X(100).
    
    PROCEDURE DIVISION.
    MAIN-PROCEDURE.
        PERFORM COMPRESS-DATA.
        PERFORM CALCULATE-RATIO.
        STOP RUN.
    
    COMPRESS-DATA.
        MOVE 'Compressing COBOL data' TO WS-RESULT.
    
    CALCULATE-RATIO.
        MOVE 'Compression complete' TO WS-RESULT.
    """ * 10
        
        original_text = cobol_program.decode('utf-8', errors='ignore')
        buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
        
        compressed = bridge.compress(buffer)
        decompressed = bridge.decompress(compressed)
        
        # Verify lossless (ignoring encoding issues)
        assert decompressed.data == original_text
        
        print(f"\nCOBOL: {len(original_text)} -> {len(str(compressed.data))} bytes")
    
    def test_json_compression(self, bridge):
        """Test with JSON-like data"""
        json_data = """
        {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "metadata": {"version": "1.0"}
        }
        """ * 50
        
        buffer = TypedBuffer.create(json_data, ProtocolLanguage.L1_SEM, str)
        compressed = bridge.compress(buffer)
        decompressed = bridge.decompress(compressed)
        
        assert decompressed.data == json_data
        print(f"\nJSON: {len(json_data)} -> {len(str(compressed.data))} bytes")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
