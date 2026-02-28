#!/usr/bin/env python3
"""Benchmark script for Layer 7 Huffman histogram kernels.

Measures throughput (GB/s) of:
  1. CPU histogram computation
  2. GPU histogram (standard atomics in shared memory)
  3. GPU histogram (warp-aggregation via __shfl_xor_sync)

Outputs timing and throughput for different input sizes and chunk sizes.
"""

import time
import sys
import numpy as np

try:
    import huffman_gpu
    HAS_CUPY = huffman_gpu._HAS_CUPY
except Exception as e:
    print(f"Warning: could not import huffman_gpu: {e}")
    HAS_CUPY = False


def gen_random_data(size_mb: int) -> bytes:
    """Generate random compressible data (weighted distribution)."""
    # Create a distribution where some symbols are more frequent
    probs = np.ones(256, dtype=np.float64) / 256.0
    # Boost first 36 symbols (1/7 of 256) to be 10x more likely
    frequent_count = 256 // 7
    probs[:frequent_count] = 10.0 / 256.0
    # Renormalize to sum to 1
    probs /= probs.sum()
    
    arr = np.random.choice(
        np.arange(256),
        size=size_mb * 1024 * 1024,
        p=probs
    ).astype(np.uint8)
    return arr.tobytes()


def benchmark_histogram(data: bytes, chunk_size: int, use_warp: bool = False, label: str = "CPU") -> dict:
    """Run histogram computation and measure throughput.

    Returns dict with keys: 'label', 'time_s', 'throughput_gbs', 'hist_shape'.
    """
    print(f"\n  Benchmarking {label}...", end=" ", flush=True)

    start = time.perf_counter()
    hist = huffman_gpu.compute_histograms(data, chunk_size=chunk_size, use_warp_agg=use_warp)
    elapsed = time.perf_counter() - start

    size_gb = len(data) / (1024 ** 3)
    throughput = size_gb / elapsed if elapsed > 0 else 0

    print(f"done ({elapsed:.3f}s, {throughput:.2f} GB/s)")

    return {
        "label": label,
        "time_s": elapsed,
        "throughput_gbs": throughput,
        "hist_shape": hist.shape,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark Huffman histogram kernels")
    parser.add_argument("--size_mb", type=int, default=100, help="Input size in MB")
    parser.add_argument("--chunk_mb", type=int, default=64, help="Chunk size in KB (default 64 = 64KB)")
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetitions")
    args = parser.parse_args()

    chunk_size = args.chunk_mb * 1024
    size_bytes = args.size_mb * 1024 * 1024

    print(f"Generating {args.size_mb} MB random data (weighted distribution)...")
    data = gen_random_data(args.size_mb)
    assert len(data) == size_bytes

    print(f"Chunk size: {chunk_size} bytes ({args.chunk_mb} KB)")
    print(f"Data size: {args.size_mb} MB")
    print(f"Repetitions: {args.repeat}")

    results = []
    for rep in range(args.repeat):
        print(f"\nRun {rep + 1}/{args.repeat}:")

        # CPU benchmark
        r_cpu = benchmark_histogram(data, chunk_size, use_warp=False, label="CPU")
        results.append(("CPU", r_cpu))

        # GPU benchmarks (if available)
        if HAS_CUPY:
            r_gpu_std = benchmark_histogram(data, chunk_size, use_warp=False, label="GPU (standard atomics)")
            results.append(("GPU_STD", r_gpu_std))

            r_gpu_warp = benchmark_histogram(data, chunk_size, use_warp=True, label="GPU (warp-aggregation)")
            results.append(("GPU_WARP", r_gpu_warp))
        else:
            print("\n  GPU benchmarks skipped (CuPy not available)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    by_label = {}
    for label, res in results:
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(res["throughput_gbs"])

    for label in by_label:
        times = by_label[label]
        avg_throughput = np.mean(times)
        std_throughput = np.std(times)
        print(f"{label:25s}: {avg_throughput:6.2f} ± {std_throughput:5.2f} GB/s")

    # Compute speedup
    if "GPU_WARP" in by_label and "GPU_STD" in by_label:
        avg_warp = np.mean(by_label["GPU_WARP"])
        avg_std = np.mean(by_label["GPU_STD"])
        if avg_std > 0:
            speedup = avg_warp / avg_std
            print(f"\nWarp-aggregation speedup vs standard: {speedup:.2f}x")

    if "GPU_STD" in by_label:
        avg_gpu = np.mean(by_label["GPU_STD"])
        avg_cpu = np.mean(by_label["CPU"])
        if avg_cpu > 0:
            gpu_speedup = avg_gpu / avg_cpu
            print(f"GPU speedup vs CPU: {gpu_speedup:.2f}x")

    print()


if __name__ == "__main__":
    main()
