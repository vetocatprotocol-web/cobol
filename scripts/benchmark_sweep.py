import time
import itertools
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from l5l8_optimized_pipeline import OptimizedL5L8Pipeline, OptimizedLayer5, OptimizedLayer6


def generate_highly_repetitive(size=200000):
    block = b"AAAAAAAAAAAAAAAA"  # highly repetitive
    return (block * (size // len(block)))[:size]


def run_sweep():
    data = generate_highly_repetitive(200000)

    best = None
    results = []

    for max_passes in [1, 2, 3, 4, 5]:
        for max_patterns in [32, 64, 128, 255]:
            for batch_size in [65536, 262144, 1048576]:
                # Build pipeline with custom layers
                pipeline = OptimizedL5L8Pipeline()
                pipeline.layer5 = OptimizedLayer5(max_patterns=max_patterns)
                pipeline.layer6 = OptimizedLayer6(batch_size=batch_size)

                start = time.time()
                compressed = pipeline.compress(data, max_passes=max_passes)
                elapsed = time.time() - start

                ratio = len(data) / len(compressed) if len(compressed) > 0 else 0
                entry = (max_passes, max_patterns, batch_size, ratio, len(compressed), elapsed)
                results.append(entry)

                if best is None or ratio > best[3]:
                    best = entry

                print(f"passes={max_passes} patterns={max_patterns} batch={batch_size} -> ratio={ratio:.2f}x size={len(compressed)} elapsed={elapsed:.3f}s")

    print('\nBEST:', best)
    return results


if __name__ == '__main__':
    run_sweep()
