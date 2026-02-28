#!/usr/bin/env python3
"""Layer-by-layer optimization analyzer and benchmarker for COBOL Protocol.

This script:
1. Analyzes each layer's current implementation
2. Runs micro-benchmarks for each layer
3. Identifies bottlenecks and optimization opportunities
4. Produces a detailed optimization report
"""

import time
import sys
import numpy as np
import importlib
from typing import Dict, List, Tuple, Type

# Import protocol bridge
try:
    from protocol_bridge import TypedBuffer, ProtocolLanguage
except Exception:
    # Fallback if protocol_bridge not fully initialized
    class TypedBuffer:
        @staticmethod
        def create(data, lang=None, dtype=None):
            class Buf:
                def __init__(self, d):
                    self.data = d
            return Buf(data)
    
    class ProtocolLanguage:
        L1_SEM = "L1"
        L2_STRUCT = "L2"

# Layer metadata: (name, module_name, [possible class names], description, notes)
LAYER_SPECS = [
    ("Layer 1", "layer1_semantic", ["Layer1Semantic", "OptimizedSemanticTokenizer", "OptimizedLayer1Encoder"], 
     "Semantic Tokenization", "Token mapping"),
    ("Layer 2", "layer2_structural", ["Layer2Structural", "Layer2Encoder", "OptimizedLayer2Encoder"],
     "Structural Encoding", "Schema templates"),
    ("Layer 3", "layer3_delta", ["Layer3Delta", "OptimizedDeltaEncoder"],
     "Delta Compression", "Differential coding"),
    ("Layer 4", "layer4_binary", ["Layer4Binary", "BinaryEncoder"],
     "Binary Encoding", "Bit packing"),
    ("Layer 5", "layer5_framework", ["Layer5Framework", "Layer5"],
     "Framework Aggregation", "Multi-layer"),
    ("Layer 6", "layer6_recursive", ["Layer6Recursive", "TrieAccelerator"],
     "GPU Trie Search", "Pattern matching"),
    ("Layer 7", "layer7_optimized", ["Layer7Optimized", "HuffmanEncoder"],
     "Parallel Huffman", "Compression"),
    ("Layer 8", "layer8_final", ["Layer8Final", "FinalEncoder"],
     "Final Hardening", "COBOL copybook"),
]


def load_layer_class(module_name: str, possible_names: List[str]) -> Type:
    """Try to load layer class from module with multiple possible names."""
    try:
        mod = importlib.import_module(module_name)
        for class_name in possible_names:
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
    except Exception as e:
        pass
    return None


def gen_test_data(size_bytes: int = 10240) -> bytes:
    """Generate compressible test data."""
    pattern = b"The quick brown fox jumps over the lazy dog. " * 50
    data = pattern * (size_bytes // len(pattern) + 1)
    return data[:size_bytes]


def benchmark_layer(layer_name: str, layer_class, test_data: bytes, iterations: int = 3) -> Dict:
    """Benchmark a single layer's encode/decode performance."""
    results = {
        "layer": layer_name,
        "encode_times": [],
        "decode_times": [],
        "throughput_encode": [],
        "throughput_decode": [],
    }

    try:
        layer = layer_class()
        
        for _ in range(iterations):
            try:
                # Encode benchmark
                buf = TypedBuffer.create(test_data, "L1", bytes)
                start = time.perf_counter()
                try:
                    encoded_buf = layer.encode(buf)
                except AttributeError:
                    # Try different method name
                    if hasattr(layer, 'compress'):
                        encoded_buf = layer.compress(buf)
                    else:
                        raise
                elapsed = time.perf_counter() - start
                results["encode_times"].append(elapsed)
                
                size_mb = len(test_data) / (1024 ** 2)
                throughput = size_mb / elapsed if elapsed > 0 else 0
                results["throughput_encode"].append(throughput)
                
                # Decode benchmark
                start = time.perf_counter()
                try:
                    decoded_buf = layer.decode(encoded_buf)
                except AttributeError:
                    if hasattr(layer, 'decompress'):
                        decoded_buf = layer.decompress(encoded_buf)
                    else:
                        decoded_buf = encoded_buf
                elapsed = time.perf_counter() - start
                results["decode_times"].append(elapsed)
                
                throughput = size_mb / elapsed if elapsed > 0 else 0
                results["throughput_decode"].append(throughput)
            except Exception as e:
                results["error"] = str(e)
                break
    except Exception as e:
        results["error"] = f"Init error: {e}"

    return results


def analyze_implementation(layer_name: str, layer_class) -> Dict:
    """Analyze implementation structure."""
    analysis = {
        "layer": layer_name,
        "has_encode": False,
        "has_decode": False,
        "status": "OK",
    }

    try:
        layer = layer_class()
        analysis["has_encode"] = hasattr(layer, "encode") or hasattr(layer, "compress")
        analysis["has_decode"] = hasattr(layer, "decode") or hasattr(layer, "decompress")
    except Exception as e:
        analysis["status"] = f"FAILED: {e}"

    return analysis


def main():
    print("\n" + "=" * 80)
    print("COBOL PROTOCOL: LAYER-BY-LAYER OPTIMIZATION ANALYZER")
    print("=" * 80)

    test_data = gen_test_data(10240)
    print(f"\nTest data size: {len(test_data)} bytes")

    # Load all layers
    print("\n" + "-" * 80)
    print("LOADING LAYERS")
    print("-" * 80)

    layers = []
    for layer_name, module_name, class_names, desc, notes in LAYER_SPECS:
        layer_class = load_layer_class(module_name, class_names)
        if layer_class:
            print(f"✓ {layer_name:10s} ({layer_class.__name__:30s}) - {desc}")
            layers.append((layer_name, layer_class, desc, notes))
        else:
            print(f"✗ {layer_name:10s} - NOT FOUND")

    # Phase 1: Implementation Analysis
    print("\n" + "-" * 80)
    print("PHASE 1: IMPLEMENTATION ANALYSIS")
    print("-" * 80)

    analyses = []
    for layer_name, layer_class, desc, notes in layers:
        analysis = analyze_implementation(layer_name, layer_class)
        analyses.append(analysis)
        enc = "✓" if analysis.get("has_encode") else "✗"
        dec = "✓" if analysis.get("has_decode") else "✗"
        status = analysis.get("status", "OK")
        print(f"{layer_name:10s} | Encode {enc} | Decode {dec} | {status:20s} | {desc}")

    # Phase 2: Performance Benchmarking
    print("\n" + "-" * 80)
    print("PHASE 2: PERFORMANCE BENCHMARKING (10 KB data, 3 iterations)")
    print("-" * 80)

    benchmarks = []
    for layer_name, layer_class, desc, notes in layers:
        print(f"Benchmarking {layer_name}...", end=" ", flush=True)
        bench = benchmark_layer(layer_name, layer_class, test_data, iterations=3)
        benchmarks.append(bench)

        if "error" in bench:
            print(f"FAILED: {bench['error']}")
        else:
            enc_avg = np.mean(bench["throughput_encode"]) if bench["throughput_encode"] else 0
            dec_avg = np.mean(bench["throughput_decode"]) if bench["throughput_decode"] else 0
            print(f"Encode: {enc_avg:.2f} MB/s | Decode: {dec_avg:.2f} MB/s")

    # Phase 3: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Layer':<12} {'Encode (MB/s)':>18} {'Decode (MB/s)':>18} {'Status':>15}")
    print("-" * 65)

    total_encode = 0
    total_decode = 0
    count = 0

    for bench in benchmarks:
        layer = bench["layer"]
        enc = np.mean(bench.get("throughput_encode", [0])) if bench.get("throughput_encode") else 0
        dec = np.mean(bench.get("throughput_decode", [0])) if bench.get("throughput_decode") else 0
        status = "OK" if (enc > 0 and dec > 0) else "NEEDS HELP"

        print(f"{layer:<12} {enc:>18.2f} {dec:>18.2f} {status:>15}")
        
        if enc > 0:
            total_encode += enc
            count += 1
        if dec > 0:
            total_decode += dec

    if count > 0:
        print("-" * 65)
        print(f"{'Average':<12} {total_encode/count:>18.2f} {total_decode/count:>18.2f}")

    print("\n" + "=" * 80)
    print("NEXT STEPS: Run individual layer benchmarks and optimization scripts")
    print("=" * 80)


if __name__ == "__main__":
    main()
