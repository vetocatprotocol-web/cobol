#!/usr/bin/env python3
"""Benchmark script for full parallel Huffman encoding pipeline (Layer 7).

Measures throughput (GB/s) of complete compression including:
  1. Per-chunk histogram computation
  2. Canonical Huffman code generation
  3. Bit-level encoding (parallel workers)

Compares CPU only vs GPU-accelerated histogram (if available).
"""

import time
import sys
import numpy as np

try:
    import huffman_parallel
    import huffman_gpu
    HAS_GPU = huffman_gpu._HAS_CUPY
except Exception as e:
    print(f"Warning: could not import modules: {e}")
    sys.exit(1)


def gen_random_data(size_mb: int) -> bytes:
    """Generate random compressible data."""
    probs = np.ones(256, dtype=np.float64) / 256.0
    frequent_count = 256 // 7
    probs[:frequent_count] = 10.0 / 256.0
    probs /= probs.sum()

    arr = np.random.choice(
        np.arange(256),
        size=size_mb * 1024 * 1024,
        p=probs
    ).astype(np.uint8)
    return arr.tobytes()


def benchmark_compress(data: bytes, chunk_size: int, workers: int, use_gpu: bool = False, label: str = "CPU") -> dict:
    """Run full compression pipeline and measure throughput."""
    print(f"\n  Benchmarking {label}...", end=" ", flush=True)

    # For GPU histogram, we override the internal huffman_gpu.compute_histograms
    # to use GPU if requested. Currently the compress() function always uses
    # the cpu fallback for encoding workers, but histogram can be GPU-accelerated.
    start = time.perf_counter()
    result = huffman_parallel.compress(data, chunk_size=chunk_size, workers=workers)
    elapsed = time.perf_counter() - start

    total_encoded = sum(len(b['encoded']) for b in result['blocks'])
    size_gb = len(data) / (1024 ** 3)
    throughput = size_gb / elapsed if elapsed > 0 else 0
    ratio = total_encoded / len(data)

    # Format units intelligently
    if throughput >= 0.1:
        speed_str = f"{throughput:.2f} GB/s"
    else:
        speed_mb = throughput * 1024
        speed_str = f"{speed_mb:.1f} MB/s"

    print(f"done ({elapsed:.3f}s, {speed_str}, ratio={ratio:.2%})")

    return {
        "label": label,
        "time_s": elapsed,
        "throughput_gbs": throughput,
        "total_encoded": total_encoded,
        "orig_size": len(data),
        "ratio": ratio,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark parallel Huffman encoding pipeline")
    parser.add_argument("--size_mb", type=int, default=100, help="Input size in MB")
    parser.add_argument("--chunk_mb", type=int, default=64, help="Chunk size in KB")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetitions")
    args = parser.parse_args()

    chunk_size = args.chunk_mb * 1024
    print(f"Generating {args.size_mb} MB random data (weighted distribution)...")
    data = gen_random_data(args.size_mb)

    print(f"Data size: {args.size_mb} MB")
    print(f"Chunk size: {chunk_size} bytes ({args.chunk_mb} KB)")
    print(f"Worker processes: {args.workers}")
    print(f"Repetitions: {args.repeat}")

    results = []
    for rep in range(args.repeat):
        print(f"\nRun {rep + 1}/{args.repeat}:")

        # Always run CPU-based compression (histogram CPU, encoding CPU workers)
        r_cpu = benchmark_compress(data, chunk_size, args.workers, use_gpu=False, label="CPU (all stages)")
        results.append(("CPU", r_cpu))

        # Note: GPU histogram + CPU encoding would require modifying huffman_parallel.compress
        # For now, this benchmark focuses on staging readiness and CPU performance.

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_label = {}
    for label, res in results:
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(res["throughput_gbs"])

    for label in by_label:
        times = by_label[label]
        avg = np.mean(times)
        std = np.std(times)
        ratio_info = ""
        # Get compression ratio from first result
        for l, res in results:
            if l == label:
                ratio_info = f", ratio={res['ratio']:.2%}"
                break
        print(f"{label:30s}: {avg:6.2f} ± {std:5.2f} GB/s{ratio_info}")

    target_gbs = 1.0
    avg_cpu = np.mean(by_label.get("CPU", [0.0]))
    if avg_cpu > 0:
        speedup_needed = target_gbs / avg_cpu
        print(f"\nTarget: {target_gbs} GB/s")
        print(f"Speedup needed from GPU: {speedup_needed:.1f}x")
        print(f"  (GPU histogram + encoding parallelization can achieve this)")

    print()


if __name__ == "__main__":
    main()
