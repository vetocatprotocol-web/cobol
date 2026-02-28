import time
from l5l8_optimized_pipeline import OptimizedL5L8Pipeline


def generate_highly_repetitive(size=200000):
    # create synthetic data with extreme repetition to approach high compression ratios
    block = b"ABCDEFGHIJKLMNOPQRST" * 4
    return (block * (size // len(block)))[:size]


if __name__ == '__main__':
    pipeline = OptimizedL5L8Pipeline()
    data = generate_highly_repetitive(200000)  # 200 KB

    start = time.time()
    compressed = pipeline.compress(data)
    elapsed = time.time() - start

    ratio = len(data) / len(compressed) if len(compressed) > 0 else 0
    print(f"Input: {len(data)} bytes")
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Ratio: {ratio:.2f}x")
    print(f"Elapsed: {elapsed:.3f}s | Throughput: {len(data)/elapsed/1024/1024:.2f} MB/s")
    print('Layer stats:', pipeline.get_stats()['layer_stats'])
