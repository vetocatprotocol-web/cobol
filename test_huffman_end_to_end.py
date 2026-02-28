#!/usr/bin/env python3
"""End-to-end test for parallel Huffman encoding pipeline (Layer 7).

Tests:
  1. Histogram computation (CPU and optionally GPU)
  2. Canonical Huffman code generation per-block
  3. Bit-level encoding with CPU workers
  4. Decompression validation
"""

import sys
import numpy as np

# Import modules
try:
    import huffman_gpu
    import huffman_parallel
    HAS_GPU = huffman_gpu._HAS_CUPY
except Exception as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def decode_block(encoded_bytes: bytes, orig_len: int, lengths: list) -> bytes:
    """Decode a single block given its metadata.

    `lengths` is a list of 256 code lengths (one per symbol).
    """
    # Sanity check: need at least one non-zero length
    if all(l == 0 for l in lengths):
        return b''

    # Build canonical codes from lengths (same logic as encoder)
    items = sorted(((lengths[i], i) for i in range(256) if lengths[i] > 0))
    if not items:
        return b''

    codes_to_sym = {}
    code = 0
    prev_len = items[0][0]
    for length, symbol in items:
        if length > prev_len:
            code <<= (length - prev_len)
            prev_len = length
        codes_to_sym[(code, length)] = symbol
        code += 1

    # Decode bit stream
    out = bytearray()
    byte_idx = 0
    bit_pos = 7
    acc_code = 0
    acc_len = 0

    while len(out) < orig_len and byte_idx < len(encoded_bytes):
        b = encoded_bytes[byte_idx]
        bit = (b >> bit_pos) & 1
        acc_code = (acc_code << 1) | bit
        acc_len += 1

        if (acc_code, acc_len) in codes_to_sym:
            sym = codes_to_sym[(acc_code, acc_len)]
            out.append(sym)
            acc_code = 0
            acc_len = 0

        bit_pos -= 1
        if bit_pos < 0:
            byte_idx += 1
            bit_pos = 7

    return bytes(out[:orig_len])


def test_small_data():
    """Test with small compressible data."""
    print("=" * 60)
    print("Test 1: Small data (1000 bytes, intentionally compressible)")
    print("=" * 60)

    data = b"abcdefghij" * 100
    assert len(data) == 1000

    # Compress
    print(f"Original size: {len(data)} bytes")
    print("Running compression pipeline...")
    res = huffman_parallel.compress(data, chunk_size=500, workers=1)
    print(f"Num chunks: {len(res['blocks'])}")

    total_encoded = sum(len(b['encoded']) for b in res['blocks'])
    print(f"Total encoded size: {total_encoded} bytes")
    print(f"Compression ratio: {total_encoded / len(data):.2%}")

    # Decompress and verify
    print("Decompressing and verifying...")
    reconstructed = bytearray()
    for block in res['blocks']:
        decoded = decode_block(block['encoded'], block['orig_len'], block['lengths'])
        reconstructed.extend(decoded)

    if bytes(reconstructed) == data:
        print("✓ Decompression successful, data matches!")
        return True
    else:
        print("✗ Decompression failed, data mismatch!")
        print(f"  Expected {len(data)} bytes, got {len(reconstructed)}")
        if len(reconstructed) > 0:
            mismatch_idx = next((i for i in range(min(len(data), len(reconstructed)))
                                 if data[i] != reconstructed[i]), -1)
            if mismatch_idx >= 0:
                print(f"  First mismatch at byte {mismatch_idx}")
        return False


def test_random_data():
    """Test with random uniform data."""
    print("\n" + "=" * 60)
    print("Test 2: Random uniform data (10 KB)")
    print("=" * 60)

    data = np.random.bytes(10240)
    assert len(data) == 10240

    print(f"Original size: {len(data)} bytes")
    print("Running compression pipeline...")
    res = huffman_parallel.compress(data, chunk_size=2048, workers=2)
    print(f"Num chunks: {len(res['blocks'])}")

    total_encoded = sum(len(b['encoded']) for b in res['blocks'])
    print(f"Total encoded size: {total_encoded} bytes")
    print(f"Compression ratio: {total_encoded / len(data):.2%}")

    # Decompress and verify
    print("Decompressing and verifying...")
    reconstructed = bytearray()
    for block in res['blocks']:
        decoded = decode_block(block['encoded'], block['orig_len'], block['lengths'])
        reconstructed.extend(decoded)

    if bytes(reconstructed) == data:
        print("✓ Decompression successful, data matches!")
        return True
    else:
        print("✗ Decompression failed, data mismatch!")
        return False


def test_histogram_accuracy():
    """Test histogram computation accuracy."""
    print("\n" + "=" * 60)
    print("Test 3: Histogram computation accuracy")
    print("=" * 60)

    data = b"aaabbbccddeef" * 1000
    print(f"Data size: {len(data)} bytes")

    # Compute histogram
    hist = huffman_gpu.compute_histograms(data, chunk_size=4096)
    print(f"Histogram shape: {hist.shape}")

    # Verify total count
    total = hist.sum()
    expected = len(data)
    print(f"Total count: {total}, expected: {expected}")

    if total == expected:
        print("✓ Histogram total count matches!")
    else:
        print("✗ Histogram count mismatch!")
        return False

    # Verify individual symbol counts for first chunk
    chunk0_data = data[:4096]
    chunk0_hist_py = np.bincount(np.frombuffer(chunk0_data, dtype=np.uint8), minlength=256)
    chunk0_hist_gpu = hist[0, :]

    if np.array_equal(chunk0_hist_py, chunk0_hist_gpu):
        print("✓ Chunk 0 histogram matches!")
        return True
    else:
        print("✗ Chunk 0 histogram mismatch!")
        mismatch = np.where(chunk0_hist_py != chunk0_hist_gpu)[0]
        if len(mismatch) > 0:
            print(f"  Differences in symbols: {mismatch[:5]}...")
        return False


def main():
    print("\n" + "=" * 60)
    print("LAYER 7: PARALLEL HUFFMAN ENCODING END-TO-END TESTS")
    print("=" * 60)
    print(f"GPU available: {HAS_GPU}")
    print()

    results = {}

    # Test 1: Small compressible data
    results["small_data"] = test_small_data()

    # Test 2: Random data
    results["random_data"] = test_random_data()

    # Test 3: Histogram accuracy
    results["histogram"] = test_histogram_accuracy()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
