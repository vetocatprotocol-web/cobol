# FPGA-Python Integration Guide
## Bridging COBOL v1.5 Hardware & Software Layers

---

## Overview

This document explains how the **FPGA RTL modules** (`cam_bank.v`, `hash_core.v`, `decompressor.v`) integrate with the existing **Python-based COBOL pipeline** (Layers 1-8).

**Key Principle:** Hardware accelerates bottleneck layers (6 & 7); Python remains for orchestration & metadata management.

---

## Software Layers → Hardware Mapping

### Original v1.4 (Pure Python)

```
Layer 1: Chunking (4 MiB blocks)
Layer 2: Entropy analysis
Layer 3: Preprocessing (delta, BPE)
Layer 4: Pattern frequency analysis
Layer 5: Recursive dictionary building
Layer 6: Trie-based dictionary encoding      ← SLOW (CPU-bound)
Layer 7: Huffman + RLE encoding              ← SLOW (bit-serial operations)
Layer 8: Hyper-Index metadata                ← Moderate (disk I/O)
```

### v1.5 (Hybrid FPGA + Python)

```
Layer 1: Chunking (4 MiB blocks)                           [CPU: Python]
Layer 2: Entropy analysis                                   [CPU: Python]
Layer 3: Preprocessing (delta, BPE)                         [CPU: Python]
Layer 4: Pattern frequency analysis                         [CPU: Python]
Layer 5: Recursive dictionary building                      [CPU: Python]
Layer 6: Trie dictionary lookup        →  [FPGA: CAM_BANK]  [HW: Accelerated]
Layer 7: Huffman encoding (encode)                          [CPU: Python]
         Huffman decoding (decode)     →  [FPGA: DECOMPRESSOR] [HW: Accelerated]
         RLE encoding/decoding         →  [FPGA: DECOMPRESSOR] [HW: Accelerated]
Layer 8: Hyper-Index metadata          →  [Mixed: HBM + NVMe] [HW-assisted]
```

---

## Data Flow Architecture

### Encoding Path (CPU → FPGA → Storage)

```
┌─────────────────────────────────────────────────────────────┐
│ Python Layer 1-5: Preprocessing (CPU)                       │
│  Input: 15 EB raw data                                      │
│  Output: 4 MiB chunks + symbol frequency distribution      │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Python Layer 6: Dictionary Building & Encoding (CPU)        │
│  Input: chunks + frequencies                                │
│  Processing:                                                │
│    1. Build Trie structure (recursive)                      │
│    2. Enumerate unique patterns → match_ids                │
│    3. Generate CAM entries: sha256(pattern) → match_id     │
│  Output: CAM configuration, compressed chunks with IDs     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
        ┌────────────────────────┐
        │ CONFIGURE FPGA CAM     │
        │  (one-time per chunk)  │
        └────────────────────────┘
         (send_to_fpga_step_1)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Python Layer 7: Huffman Encoding (CPU)                      │
│  Input: chunk (post-Layer 6 encoding)                       │
│  Processing:                                                │
│    1. Compute Huffman codes per chunk                       │
│    2. Apply RLE preprocessing                               │
│    3. Bitpack using Huffman code tables                     │
│  Output: compressed payload (30 PB), Huffman tables         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
        ┌────────────────────────┐
        │ CONFIGURE FPGA         │
        │ HUFFMAN TABLES         │
        │  (per 4 MiB chunk)     │
        └────────────────────────┘
         (send_to_fpga_step_2)
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Data → NVMe-oF Storage (3 geographic centers A, B, C)      │
│  Monitoring: SHA-256 of chunk, size, match_ids            │
└─────────────────────────────────────────────────────────────┘
```

### Decoding Path (Storage → FPGA → Network → Client)

```
┌─────────────────────────────────────────────────────────────┐
│ Compressed data in 3 geographic centers A, B, C             │
│ Accessed via NVMe-oF, triggered by client request          │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ FPGA DECOMPRESSOR (hardware pipeline)                       │
│                                                             │
│  Stage 1: Bit-level extraction                            │
│           Convert byte stream → bit stream                │
│           Input: 512 bits/cycle (64 B compressed)         │
│                                                             │
│  Stage 2: Huffman decode                                  │
│           Lookup canonical Huffman table (per-chunk)      │
│           Extract variable-length codes                   │
│           Output: symbols (bytes)                         │
│                                                             │
│  Stage 3: RLE expansion                                   │
│           Detect repeat markers                           │
│           Emit run_count copies of symbol                 │
│                                                             │
│  Stage 4: Output assembly                                 │
│           Pack symbols into 512-bit words                 │
│           Output: 512 bits/cycle (64 B decompressed)      │
│                                                             │
│  Output throughput: 25 GB/s input (compressed)            │
│                     → 12.5 TB/s logical (restored)         │
└────────────────────┬────────────────────────────────────────┘
                     ↓
         ┌───────────────────────┐
         │ Optional: CAM Lookup  │
         │ (if dictionary IDs    │
         │  were stored in       │
         │  decompressed data)   │
         └───────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Python Layer 8: Hyper-Index Lookup (CPU + FPGA cache)      │
│  Input: decompressed metadata                               │
│  Processing:                                                │
│    1. Query GRT (RAM on aggregator)                        │
│    2. Fetch per-pod HBM index (RDMA)                       │
│    3. Search FST/B-tree on NVMe                            │
│  Output: file locations, sizes, metadata                   │
└────────────────────┬────────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────────┐
│ Network transmission to client                              │
│  At 500:1 compression:                                     │
│    - Physical rate: 200 Gbps (FPGA output link)           │
│    - Logical rate perceived by client: 200 Gbps × 500    │
│      (but transmitted as 80 Mbps physical if client is 2G)│
└─────────────────────────────────────────────────────────────┘
```

---

## Hardware-Software Interface Specification

### 1. Control Path (Python → FPGA Configuration)

#### CAM Dictionary Configuration

```python
# fpga_config.py: Python module for FPGA control

class FPGAController:
    def __init__(self, device_id=0):
        self.fpga = open_fpga_device(device_id)
        self.config_buffer = []
    
    def configure_cam_entry(self, pattern_bytes, match_id, chunk_size):
        """
        Configure a single CAM entry.
        
        Args:
            pattern_bytes: original uncompressed pattern (up to 512 bytes)
            match_id: integer ID (e.g., dictionary entry #)
            chunk_size: size hint for decompression (in bytes)
        """
        # Step 1: Compute SHA-256 truncated to 96 bits
        sha256_full = hashlib.sha256(pattern_bytes).digest()
        cam_key = int.from_bytes(sha256_full[:12], 'big')  # 96 bits
        
        # Step 2: Compose CAM write command
        cam_cfg = {
            'addr': match_id,           # Write address
            'key': cam_key,             # 96-bit lookup key
            'match_id': match_id,       # Entry ID (echo)
            'len': len(pattern_bytes),  # Original length
            'is_hbm': 1 if len(pattern_bytes) > 64 else 0  # Tier selection
        }
        
        # Step 3: Queue for batch write
        self.config_buffer.append(cam_cfg)
    
    def flush_cam_config(self):
        """Flush all queued CAM entries to FPGA."""
        for cfg in self.config_buffer:
            self.fpga.write_csr(
                addr=0x1000 + cfg['addr'],
                data={
                    'key': cfg['key'],
                    'match_id': cfg['match_id'],
                    'len': cfg['len'],
                    'is_hbm': cfg['is_hbm'],
                    'valid': 1
                }
            )
        self.config_buffer.clear()
    
    def load_huffman_tables(self, chunk_id, huffman_code_table):
        """
        Load Huffman decoding tables for a chunk.
        
        Args:
            chunk_id: unique identifier (4 MiB chunk)
            huffman_code_table: list of (code_bits, code_length, symbol)
        """
        for i, (code, code_len, symbol) in enumerate(huffman_code_table):
            self.fpga.write_csr(
                addr=0x2000 + (chunk_id << 12) + i,
                data={
                    'code_len': code_len,
                    'code_val': code,
                    'symbol': symbol,
                    'valid': 1
                }
            )

# Usage example
fpga = FPGAController(device_id=0)

# Configure CAM with sample dictionary
for entry_id, pattern in enumerate(dictionary_patterns):
    fpga.configure_cam_entry(pattern, entry_id, len(pattern))
fpga.flush_cam_config()

# Load Huffman tables
for chunk_id, huffman_data in huffman_tables.items():
    fpga.load_huffman_tables(chunk_id, huffman_data)
```

#### Data Path Interface

```python
class FPGADataPath:
    """Manage data ingress/egress with FPGA."""
    
    def __init__(self, fpga_dev):
        self.fpga = fpga_dev
        self.nvme_engine = NVMeOFClient()  # Access backend storage
    
    def ingest_chunk(self, chunk_id, compressed_data, huffman_table):
        """
        Ingest a compressed chunk through FPGA pipeline.
        
        Flow:
          1. Configure Huffman table
          2. Stream compressed data to NVMe-oF
          3. FPGA reads, decompresses, optionally matches against CAM
          4. Monitor latency & compression ratio
        """
        
        # Step 1: Load Huffman
        self.fpga.load_huffman_tables(chunk_id, huffman_table)
        
        # Step 2: Write to NVMe (triggers FPGA ingress)
        self.nvme_engine.write_chunk(chunk_id, compressed_data)
        
        # Step 3: Poll metrics
        metrics = self.fpga.get_metrics()
        print(f"Chunk {chunk_id}: {metrics['decomp_throughput_gb_s']} GB/s")
    
    def read_from_client(self, client_id, metadata_query):
        """
        Process client request: metadata search across 15 EB.
        
        Returns <5 ms latency via:
          1. GRT lookup (aggregator RAM)
          2. Pod HBM index search
          3. NVMe range query
          4. FPGA decompression (metadata only)
        """
        
        # Query GRT (distributed globally)
        pod_list = grt.lookup(metadata_query['fingerprint'])
        
        # Parallel queries to pod indices
        chunk_list = []
        for pod_id in pod_list:
            chunks = self.fpga.query_hyper_index(pod_id, metadata_query)
            chunk_list.extend(chunks)
        
        # Fetch compressed metadata from one replicaon (B or C if local unavailable)
        for chunk_id in chunk_list:
            compressed_meta = self.nvme_engine.read_chunk(chunk_id)
            
            # FPGA decompresses on-the-fly (< 100 µs per 4 KiB)
            decompressed_meta = self.fpga.decompress_stream(compressed_meta)
            
            # Final matching on restored metadata
            yield from self.match_metadata(decompressed_meta, metadata_query)
```

### 2. Monitoring Path (FPGA → Python Observability)

```python
class FPGAMetrics:
    """Query runtime metrics from FPGA."""
    
    def __init__(self, fpga_dev):
        self.fpga = fpga_dev
    
    def get_metrics(self):
        """Poll FPGA performance counters."""
        return {
            # Throughput
            'input_gb_s': self.fpga.read_counter('INPUT_RATE'),
            'decomp_gb_s': self.fpga.read_counter('DECOMP_RATE'),
            'output_gb_s': self.fpga.read_counter('OUTPUT_RATE'),
            
            # CAM performance
            'cam_hit_rate': self.fpga.read_counter('CAM_HITS') / 
                            max(1, self.fpga.read_counter('CAM_PROBES')),
            'cam_probe_depth': self.fpga.read_counter('AVG_PROBE_LATENCY'),
            
            # Memory utilization
            'hbm_cache_util': self.fpga.read_counter('HBM_USED_MB'),
            'bram_util': self.fpga.read_counter('BRAM_USED_KB'),
            
            # Data integrity
            'crc32_errors': self.fpga.read_counter('CRC32_MISMATCH'),
            'compression_ratio': self.fpga.read_counter('DECOMP_SIZE') / 
                                max(1, self.fpga.read_counter('COMP_SIZE')),
            
            # Errors
            'hbm_timeout_errors': self.fpga.read_counter('HBM_TIMEOUT'),
            'nvme_read_errors': self.fpga.read_counter('NVME_ERROR_COUNT'),
        }
    
    def export_dashboard(self):
        """Export metrics for monitoring dashboard."""
        m = self.get_metrics()
        return {
            'performance': {
                'throughput_gb_s': [m['input_gb_s'], m['decomp_gb_s'], m['output_gb_s']],
                'cam_hit_rate_pct': m['cam_hit_rate'] * 100,
                'compression_ratio': m['compression_ratio'],
            },
            'health': {
                'crc_errors': m['crc32_errors'],
                'hbm_timeout': m['hbm_timeout_errors'],
                'nvme_errors': m['nvme_read_errors'],
            },
            'resources': {
                'hbm_util_mb': m['hbm_cache_util'],
                'bram_util_kb': m['bram_util'],
            }
        }

# Usage
metrics_collector = FPGAMetrics(fpga)
while True:
    stats = metrics_collector.get_metrics()
    print(f"Throughput: {stats['decomp_gb_s']:.1f} GB/s")
    print(f"CAM hit: {stats['cam_hit_rate']*100:.1f}%")
    time.sleep(1)
```

### 3. Error Handling & Recovery

```python
class FPGAErrorHandler:
    """Handle failures and recovery."""
    
    def __init__(self, fpga_dev, replication_manager):
        self.fpga = fpga_dev
        self.repl_mgr = replication_manager
    
    def on_crc_mismatch(self, chunk_id):
        """Handle checksumnotify mismatch (data corruption detected)."""
        print(f"[ERROR] Chunk {chunk_id} CRC32 mismatch")
        
        # Step 1: Identify which replica has good copy
        replicas = self.repl_mgr.get_replicas(chunk_id)  # [A, B, C]
        good_replica = None
        
        for replica_id in replicas:
            if self.repl_mgr.verify_checksum(chunk_id, replica_id):
                good_replica = replica_id
                break
        
        if good_replica is None:
            # Last resort: erasure code reconstruction
            reconstructed = self.repl_mgr.reconstruct_from_parity(chunk_id)
            good_replica = reconstructed
        
        # Step 2: Fetch from good replica
        good_data = self.repl_mgr.read_chunk(chunk_id, from_replica=good_replica)
        
        # Step 3: Overwrite local copy
        local_center = self.repl_mgr.get_local_center()
        self.repl_mgr.write_chunk(chunk_id, good_data, to_center=local_center)
        
        # Step 4: Audit log
        self.log_repair_event({
            'chunk_id': chunk_id,
            'from_replica': good_replica,
            'timestamp': time.time(),
            'reason': 'crc32_mismatch'
        })
    
    def on_hbm_timeout(self, access_addr):
        """Handle HBM access timeout (likely stalled thread)."""
        print(f"[WARN] HBM access timeout at addr {access_addr:hex}")
        
        # Attempt restart of FPGA pipeline
        self.fpga.soft_reset()
        time.sleep(0.1)
        
        # Re-configure critical state
        self.fpga.flush_cam_config()
        
        # Mark affected clients for reconnect
        affected_clients = self.get_affected_clients()
        for client in affected_clients:
            self.send_to_client(client, "RETRY_LATER")
    
    def on_replication_lag(self, center_id, lag_seconds):
        """Replication backlog detected."""
        if lag_seconds > 60:
            print(f"[WARN] Center {center_id} replication lag: {lag_seconds}s")
            # Pause new writes to this center temporarily
            self.repl_mgr.throttle_writes(center_id)
        else:
            # Resume normal
            self.repl_mgr.resume_writes(center_id)
```

---

## Compilation & Deployment Example

### Building CAM Dictionary from Python Trie

```python
# layer6_to_fpga.py: Convert Python Trie to FPGA CAM config

from collections import defaultdict
import hashlib

class TrieToCAM:
    """Convert software Trie dictionary to FPGA CAM entries."""
    
    def __init__(self, trie_root):
        self.trie = trie_root
        self.cam_entries = []
    
    def extract_patterns(self):
        """DFS traverse trie, emit all unique patterns."""
        def dfs(node, current_pattern):
            if node.is_terminal:
                match_id = node.dict_id
                pattern_bytes = current_pattern.encode('utf-8')
                self.cam_entries.append({
                    'pattern': pattern_bytes,
                    'match_id': match_id,
                    'length': len(pattern_bytes)
                })
            
            for char, child in node.children.items():
                dfs(child, current_pattern + char)
        
        dfs(self.trie, "")
        return self.cam_entries
    
    def generate_fpga_config(self):
        """Generate FPGA CAM configuration."""
        config = []
        for entry in self.cam_entries:
            # Compute 96-bit CAM key
            sha256_full = hashlib.sha256(entry['pattern']).digest()
            cam_key = int.from_bytes(sha256_full[:12], 'big')
            
            config.append({
                'addr': entry['match_id'],
                'key': cam_key,
                'len': entry['length'],
                'is_hbm': 1 if entry['length'] > 64 else 0
            })
        
        return config
    
    def write_to_fpga(self, fpga_controller):
        """Program FPGA with CAM entries."""
        config = self.generate_fpga_config()
        for cfg in config:
            fpga_controller.configure_cam_entry(
                pattern_bytes=self.cam_entries[cfg['addr']]['pattern'],
                match_id=cfg['match_id'],
                chunk_size=cfg['len']
            )
        fpga_controller.flush_cam_config()

# Usage from main pipeline
from layer6_semantic import trie_root
from layer6_to_fpga import TrieToCAM

converter = TrieToCAM(trie_root)
converter.extract_patterns()
converter.write_to_fpga(fpga_controller)
```

---

## Testing & Validation

```python
# test_fpga_integration.py: End-to-end validation

def test_cam_correctness():
    """Verify CAM lookup finds expected patterns."""
    fpga = FPGAController()
    
    # Load dictionary
    patterns = ["hello", "world", "compression"]
    for i, pat in enumerate(patterns):
        fpga.configure_cam_entry(pat.encode(), i, len(pat))
    fpga.flush_cam_config()
    
    # Test lookups
    for i, pat in enumerate(patterns):
        sha256_full = hashlib.sha256(pat.encode()).digest()
        cam_key = int.from_bytes(sha256_full[:12], 'big')
        
        result = fpga.cam_lookup(cam_key)
        assert result['match_id'] == i, f"CAM lookup failed for {pat}"
    
    print("✓ CAM correctness test passed")

def test_decompressor_throughput():
    """Verify decompressor sustains 25 GB/s."""
    fpga = FPGAController()
    
    # Generate test data
    test_size = 1024 * 1024 * 1024  # 1 GB
    test_data = os.urandom(test_size)
    
    # Compress with Huffman
    compressed = huffman_encode(test_data)
    
    # Measure decompression
    start = time.time()
    decompressed = fpga.decompress_stream(compressed)
    elapsed = time.time() - start
    
    throughput_gb_s = test_size / 1e9 / elapsed
    print(f"Decompressor throughput: {throughput_gb_s:.1f} GB/s")
    assert throughput_gb_s >= 25, "Throughput below 25 GB/s target"
    
    # Verify correctness
    assert hashlib.sha256(decompressed).digest() == \
           hashlib.sha256(test_data).digest(), "Decompression mismatch"
    
    print("✓ Decompressor throughput test passed")

def test_end_to_end_latency():
    """Verify metadata search <5 ms SLA."""
    fpga = FPGAController()
    aggregator_grt = GlobalRoutingTable()
    
    # Simulate query across 15 EB
    query = "find all files with pattern X"
    
    start = time.perf_counter()
    
    # 1. GRT lookup
    pods = aggregator_grt.lookup(fingerprint(query))
    
    # 2. Pod index queries (parallel)
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fpga.query_pod_index, pod, query)
            for pod in pods
        ]
        results = [f.result() for f in as_completed(futures)]
    
    # 3. Decompress metadata
    for chunk_id in results:
        fpga.decompress_stream(read_compressed_metadata(chunk_id))
    
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"End-to-end latency: {latency_ms:.1f} ms")
    assert latency_ms < 5, f"Latency {latency_ms} ms exceeds 5 ms SLA"
    
    print("✓ E2E latency test passed")

if __name__ == "__main__":
    test_cam_correctness()
    test_decompressor_throughput()
    test_end_to_end_latency()
    print("\n✓✓✓ All integration tests passed ✓✓✓")
```

---

## Summary

The FPGA-Python integration follows this pattern:

1. **Python Layers 1-5:** Data preprocessing (CPU-efficient)
2. **Python Layer 6:** Dictionary building + CAM entry generation (CPU)
3. **Python Layer 7:** Huffman encoding (CPU) + table generation
4. **FPGA Ingestion:** CAM configuration + Huffman table loading
5. **FPGA Data Path:** Streaming decompression (HW-accelerated)
6. **Python Layer 8:** Hyper-Index queries (CPU + FPGA cache)
7. **Output:** Decompressed data to client or storage

By separating concerns, we gain:
- **Performance:** Hardware accelerates critical paths (6 & 7)
- **Flexibility:** Python handles logic, FPGA handles throughput
- **Testability:** Each layer can be validated independently
- **Scalability:** Clone FPGA pipeline across 5,000 units without code change

---

**Status:** Integration guide complete. Ready for software development & testing.
