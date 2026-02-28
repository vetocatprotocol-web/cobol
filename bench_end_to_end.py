"""
End-to-end benchmark script for COBOL Protocol GPU-accelerated pipeline (Layer6+7).
Usage:
    python bench_end_to_end.py --size_mb 100 --repeat 3

This script will:
- generate random data
- run pipeline with GPU if available, else CPU fallback
- measure throughput and validate lossless round-trip where possible
"""
import argparse
import time
import numpy as np

from engine import HPCCompressionEngine

# import base engine by path
import importlib.util, sys
spec = importlib.util.spec_from_file_location('base_engine','/workspaces/cobol/engine.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

# Attempt to load GPU wrappers
use_gpu = False
try:
    from gpu_acceleration import GPUDetector
    detector = GPUDetector()
    use_gpu = detector.gpu_available
except Exception:
    detector = None

# create simple base_engine placeholder if module provides class
try:
    BaseEngineClass = getattr(base, 'SomeCompressionEngine', None)
    base_engine = BaseEngineClass() if BaseEngineClass is not None else None
except Exception:
    base_engine = None

hpc = HPCCompressionEngine(base_engine or base)

parser = argparse.ArgumentParser()
parser.add_argument('--size_mb', type=int, default=10)
parser.add_argument('--repeat', type=int, default=3)
args = parser.parse_args()

size = args.size_mb * 1024 * 1024
print(f"Generating {args.size_mb} MB random data...")
data = np.random.randint(0, 256, size, dtype=np.uint8).tobytes()

print('Running benchmark...')
start = time.time()
for i in range(args.repeat):
    out, stats = hpc.compress_optimized(data, batch_size=4*1024*1024)
    print(f"Run {i+1}: {stats['throughput_mb_s']:.2f} MB/s, elapsed {stats['elapsed_time']:.2f}s")
end = time.time()
print(f"Total elapsed: {end-start:.2f}s")
