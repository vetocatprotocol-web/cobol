"""
Simple test runner for Multi-Layer Translation Bridge (without pytest)
"""

import sys
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

def test_l1_semantic_roundtrip():
    """Test Layer 1: Semantic roundtrip"""
    print("\n[TEST] L1 Semantic Roundtrip...", end=" ")
    l1 = Layer1Semantic()
    original_text = "Hello COBOL Protocol"
    
    buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
    compressed = l1.encode(buffer)
    decompressed = l1.decode(compressed)
    
    assert decompressed.data == str(original_text), "Data mismatch"
    assert decompressed.header == ProtocolLanguage.L1_SEM, "Header mismatch"
    assert decompressed.type == str, "Type mismatch"
    print("✓ PASS")

def test_l2_structural_roundtrip():
    """Test Layer 2: Structural roundtrip"""
    print("[TEST] L2 Structural Roundtrip...", end=" ")
    l1 = Layer1Semantic()
    l2 = Layer2Structural()
    
    original_text = "COBOL 1234 5678"
    buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
    l1_compressed = l1.encode(buffer)
    l2_compressed = l2.encode(l1_compressed)
    l2_decompressed = l2.decode(l2_compressed)
    
    assert l2_decompressed.header == ProtocolLanguage.L1_SEM, "Header mismatch"
    assert np.array_equal(l2_decompressed.data, l1_compressed.data), "Data mismatch"
    print("✓ PASS")

def test_l3_delta_roundtrip():
    """Test Layer 3: Delta roundtrip"""
    print("[TEST] L3 Delta Roundtrip...", end=" ")
    l1 = Layer1Semantic()
    l2 = Layer2Structural()
    l3 = Layer3Delta()
    
    original_text = "DELTA TEST"
    buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
    l1_out = l1.encode(buffer)
    l2_out = l2.encode(l1_out)
    l3_out = l3.encode(l2_out)
    l3_back = l3.decode(l3_out)
    
    assert np.array_equal(l3_back.data, l2_out.data), "Data mismatch"
    print("✓ PASS")

def test_l4_binary_roundtrip():
    """Test Layer 4: Binary roundtrip"""
    print("[TEST] L4 Binary Roundtrip...", end=" ")
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
    
    assert l4_back.header == ProtocolLanguage.L3_DELTA, "Header mismatch"
    assert np.array_equal(l4_back.data, l3_out.data), "Data mismatch"
    print("✓ PASS")

def test_full_pipeline_roundtrip():
    """Test full L1-L8 pipeline roundtrip"""
    print("[TEST] Full L1-L8 Pipeline Roundtrip...", end=" ")
    bridge = ProtocolBridge([
        Layer1Semantic(), Layer2Structural(), Layer3Delta(), Layer4Binary(),
        Layer5Recursive(), Layer6Recursive(), Layer7Bank(), Layer8Final()
    ])
    
    original_text = "COBOL PROTOCOL v1.2 TESTING"
    
    buffer = TypedBuffer.create(original_text, ProtocolLanguage.L1_SEM, str)
    compressed = bridge.compress(buffer)
    
    assert compressed.header == ProtocolLanguage.L8_COBOL, "Final header mismatch"
    assert compressed.type == str, "Final type mismatch"
    assert isinstance(compressed.data, str), "Data not string"
    
    # Decompress back
    decompressed = bridge.decompress(compressed)
    assert decompressed.data == original_text, "Roundtrip failed"
    assert decompressed.header == ProtocolLanguage.L1_SEM, "Decompressed header mismatch"
    print("✓ PASS")

def test_sha256_integrity():
    """Test SHA-256 integrity checking per layer"""
    print("[TEST] SHA-256 Integrity...", end=" ")
    original = "INTEGRITY CHECK"
    buffer = TypedBuffer.create(original, ProtocolLanguage.L1_SEM, str)
    
    l1 = Layer1Semantic()
    l1_out = l1.encode(buffer)
    
    # SHA-256 should be set for each buffer
    assert l1_out.sha256 is not None, "SHA-256 not set"
    assert len(l1_out.sha256) == 64, "SHA-256 length wrong"
    
    # Verify hash consistency
    recalc_hash = hashlib.sha256(l1_out.data if isinstance(l1_out.data, bytes) else str(l1_out.data).encode()).hexdigest()
    assert l1_out.sha256 == recalc_hash, "SHA-256 mismatch"
    print("✓ PASS")

def test_type_consistency():
    """Verify type consistency across layers"""
    print("[TEST] Type Consistency...", end=" ")
    l1 = Layer1Semantic()
    l2 = Layer2Structural()
    l3 = Layer3Delta()
    l4 = Layer4Binary()
    
    original = "TYPE TEST"
    buffer = TypedBuffer.create(original, ProtocolLanguage.L1_SEM, str)
    
    l1_out = l1.encode(buffer)
    assert l1_out.type == np.ndarray, "L1 type wrong"
    
    l2_out = l2.encode(l1_out)
    assert l2_out.type == np.ndarray, "L2 type wrong"
    
    l3_out = l3.encode(l2_out)
    assert l3_out.type == np.ndarray, "L3 type wrong"
    
    l4_out = l4.encode(l3_out)
    assert l4_out.type == bytes, "L4 type wrong"
    print("✓ PASS")

def test_throughput_benchmark():
    """Benchmark pipeline throughput"""
    print("[TEST] Throughput Benchmark...", end=" ")
    bridge = ProtocolBridge([
        Layer1Semantic(), Layer2Structural(), Layer3Delta(), Layer4Binary(),
        Layer5Recursive(), Layer6Recursive(), Layer7Bank(), Layer8Final()
    ])
    
    # Generate 1MB test data
    test_data = "COBOL PROTOCOL " * (1_000_000 // 15)
    
    start = time.time()
    buffer = TypedBuffer.create(test_data, ProtocolLanguage.L1_SEM, str)
    compressed = bridge.compress(buffer)
    compress_time = time.time() - start
    
    # Calculate throughput
    total_bytes = len(test_data)
    throughput_mbps = (total_bytes / (1024 * 1024)) / compress_time if compress_time > 0 else 0
    
    # Decompress
    start = time.time()
    decompressed = bridge.decompress(compressed)
    decompress_time = time.time() - start
    
    throughput_mb_decomp = (total_bytes / (1024 * 1024)) / decompress_time if decompress_time > 0 else 0
    
    # Verify lossless
    assert decompressed.data == test_data, "Lossless check failed"
    
    print(f"\n  Compress: {throughput_mbps:.2f} MB/s, Decompress: {throughput_mb_decomp:.2f} MB/s ✓ PASS")

def test_cobol_program_compression():
    """Test with real COBOL program"""
    print("[TEST] COBOL Program Compression...", end=" ")
    bridge = ProtocolBridge([
        Layer1Semantic(), Layer2Structural(), Layer3Delta(), Layer4Binary(),
        Layer5Recursive(), Layer6Recursive(), Layer7Bank(), Layer8Final()
    ])
    
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
    
    # Verify lossless
    assert decompressed.data == original_text, "Lossless check failed"
    print(f"\n  {len(original_text)} -> {len(str(compressed.data))} bytes ✓ PASS")

def main():
    print("\n" + "="*70)
    print("MULTI-LAYER TRANSLATION BRIDGE (L1-L8) TEST SUITE")
    print("="*70)
    
    tests = [
        ("L1 Semantic Roundtrip", test_l1_semantic_roundtrip),
        ("L2 Structural Roundtrip", test_l2_structural_roundtrip),
        ("L3 Delta Roundtrip", test_l3_delta_roundtrip),
        ("L4 Binary Roundtrip", test_l4_binary_roundtrip),
        ("Full L1-L8 Roundtrip", test_full_pipeline_roundtrip),
        ("SHA-256 Integrity", test_sha256_integrity),
        ("Type Consistency", test_type_consistency),
        ("Throughput Benchmark", test_throughput_benchmark),
        ("COBOL Compression", test_cobol_program_compression),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ FAIL: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} PASSED, {failed} FAILED")
    print("="*70 + "\n")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
