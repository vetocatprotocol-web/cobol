"""Parallel canonical Huffman encoder using per-chunk histograms.

This module performs the remaining steps of a parallel Huffman pipeline:
- Build canonical Huffman codes per chunk (based on histogram)
- Encode each chunk in parallel (multiprocessing)

It expects `huffman_gpu.compute_histograms` to provide per-chunk histograms.
If a GPU is available, histogram calculation is fast on the device; encoding is
performed on CPU in parallel (per-chunk) to keep implementation portable.
"""

import heapq
import multiprocessing as mp
from typing import Dict, Tuple, List
import numpy as np

try:
    # when used as package
    from . import huffman_gpu
except Exception:
    # when executed as script / top-level import
    import huffman_gpu


def _build_huffman_tree_lengths(freqs: np.ndarray) -> Dict[int, int]:
    """Return code lengths (symbol -> length) using standard Huffman algorithm.

    `freqs` is a 1-D array of length 256 of integer counts.
    """
    items = [(int(f), i) for i, f in enumerate(freqs) if int(f) > 0]
    if not items:
        return {}

    # Special-case: single symbol
    if len(items) == 1:
        sym = items[0][1]
        return {sym: 1}

    # Build heap of (freq, node_id). We'll keep a simple tree as tuples.
    heap = []
    node_id = 0
    nodes = {}
    for freq, sym in items:
        nodes[node_id] = (None, None, sym)  # left, right, symbol
        heapq.heappush(heap, (freq, node_id))
        node_id += 1

    while len(heap) > 1:
        f1, n1 = heapq.heappop(heap)
        f2, n2 = heapq.heappop(heap)
        nodes[node_id] = (n1, n2, -1)
        heapq.heappush(heap, (f1 + f2, node_id))
        node_id += 1

    # root
    _, root = heap[0]

    # traverse to get lengths
    lengths = {}

    def dfs(n, depth):
        left, right, sym = nodes[n]
        if sym != -1:
            lengths[sym] = depth
            return
        if left is not None:
            dfs(left, depth + 1)
        if right is not None:
            dfs(right, depth + 1)

    dfs(root, 0)
    return lengths


def _canonical_codes_from_lengths(lengths: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """Given symbol->length, return canonical codes as symbol->(code, length).

    Codes are MSB-first integers.
    """
    if not lengths:
        return {}

    items = sorted(((l, s) for s, l in lengths.items()))
    codes: Dict[int, Tuple[int, int]] = {}
    code = 0
    prev_len = items[0][0]
    for length, symbol in items:
        if length > prev_len:
            code <<= (length - prev_len)
            prev_len = length
        codes[symbol] = (code, length)
        code += 1

    return codes


def _encode_block(args) -> Tuple[int, bytes, dict]:
    """Worker: encode one block.

    Returns tuple (index, encoded_bytes, metadata) where metadata contains
    'orig_len' and 'lengths' (list of 256 code lengths) to allow decoding.
    """
    idx, block_bytes, codes = args
    out = bytearray()
    acc = 0
    acc_len = 0
    # codes: dict symbol -> (code, length)
    for b in block_bytes:
        c, l = codes[b]
        # append bits (MSB-first)
        acc = (acc << l) | c
        acc_len += l
        while acc_len >= 8:
            shift = acc_len - 8
            out.append((acc >> shift) & 0xFF)
            acc_len -= 8
            acc &= (1 << acc_len) - 1 if acc_len > 0 else 0
    if acc_len > 0:
        out.append((acc << (8 - acc_len)) & 0xFF)

    # Build lengths metadata: 256-length list
    lengths = [0] * 256
    for s, (c, l) in codes.items():
        lengths[s] = l

    meta = {"orig_len": len(block_bytes), "lengths": lengths}
    return idx, bytes(out), meta


def compress(data: bytes, chunk_size: int = 65536, workers: int = None):
    """Compress `data` by splitting into `chunk_size` blocks, building a
    canonical Huffman code per-block, and encoding blocks in parallel.

    Returns a dict with keys:
    - `chunk_size`
    - `blocks`: list of dicts with `index`, `orig_len`, `lengths`, `encoded`
    """
    if workers is None:
        workers = max(1, mp.cpu_count() - 1)

    hist = huffman_gpu.compute_histograms(data, chunk_size=chunk_size)
    num_chunks = hist.shape[0]

    # Prepare codes per chunk
    codes_per_chunk: List[Dict[int, Tuple[int, int]]] = []
    for i in range(num_chunks):
        lengths = _build_huffman_tree_lengths(hist[i])
        codes = _canonical_codes_from_lengths(lengths)
        # For decoding convenience, ensure every symbol present in codes has tuple
        codes_per_chunk.append(codes)

    # Prepare block bytes
    arr = memoryview(data)
    blocks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        blocks.append((i, bytes(arr[start:end]), codes_per_chunk[i]))

    # Encode in parallel
    with mp.Pool(processes=workers) as pool:
        results = pool.map(_encode_block, blocks)

    # Collect results in index order
    results.sort(key=lambda x: x[0])
    output_blocks = []
    for idx, encoded_bytes, meta in results:
        output_blocks.append({"index": idx, "orig_len": meta["orig_len"], "lengths": meta["lengths"], "encoded": encoded_bytes})

    return {"chunk_size": chunk_size, "blocks": output_blocks}
