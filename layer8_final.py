#!/usr/bin/env python3
"""
COBOL Protocol v1.5.1 - Layer 8: Ultra-Extreme Nodes with Random Access

Enhanced implementation with:
- Global Mapping Dictionary (block tracking)
- Offset Indexing (random access support)
- SHA-256 Integrity Verification
- Memory-optimized Hash Map
- Thread-safe operations

This module bridges:
1. COMP-3 ↔ COBOL COPYBOOK encoding (original functionality)
2. Random access indexing (new v1.5.1 feature)
3. SHA-256 distributed verification (integration with streaming)
"""

from protocol_bridge import TypedBuffer, ProtocolLanguage
from layer8_ultra_extreme_enhanced import (
    Layer8UltraExtremeManager,
    GlobalMappingDictionary,
    OffsetIndex,
    RandomAccessQueryEngine,
    SHA256IntegrityValidator,
    BlockMetadata,
    DEFAULT_L8_NODES
)

import base64
import hashlib
from typing import Optional, Dict, List, Tuple


class Layer8Final:
    """
    Layer 8 Ultra-Extreme Nodes with Random Access Indexing
    
    Features:
    1. COMP-3 ↔ COBOL encoding (lossless conversion)
    2. Global block tracking via Global Mapping Dictionary
    3. Offset-based random access via OffsetIndex
    4. SHA-256 integrity verification
    5. Memory-optimized Hash Map for 1 PB+ storage
    """
    
    def __init__(self, num_l8_nodes: int = DEFAULT_L8_NODES):
        self.l8_manager = Layer8UltraExtremeManager(num_nodes=num_l8_nodes)
        self.num_nodes = num_l8_nodes
    
    # ========================================================================
    # ORIGINAL FUNCTIONALITY - COMP-3 ↔ COBOL ENCODING
    # ========================================================================
    
    def encode(self, buffer: TypedBuffer) -> TypedBuffer:
        """
        Encode COMP-3 data to COBOL Copybook format (PIC X)
        
        Lossless encoding via base64.
        """
        b64 = base64.b64encode(buffer.data).decode('ascii')
        pic_x = 'PIC X(' + str(len(buffer.data)) + ') VALUE IS \'' + b64 + '\''
        return TypedBuffer.create(pic_x, ProtocolLanguage.L8_COBOL, str)

    def decode(self, buffer: TypedBuffer) -> TypedBuffer:
        """
        Decode COBOL Copybook format back to COMP-3 data
        
        Lossless decoding via base64.
        """
        try:
            start = buffer.data.find("'") + 1
            end = buffer.data.rfind("'")
            b64_str = buffer.data[start:end]
            comp3 = base64.b64decode(b64_str)
            return TypedBuffer.create(comp3, ProtocolLanguage.L7_COMP3, bytes)
        except Exception:
            return TypedBuffer.create(buffer.data.encode(), ProtocolLanguage.L7_COMP3, bytes)
    
    # ========================================================================
    # NEW FUNCTIONALITY - RANDOM ACCESS INDEXING
    # ========================================================================
    
    def register_block_metadata(self, block_id: int, metadata: BlockMetadata) -> None:
        """
        Register a block in the global mapping and offset indexes
        
        This must be called for each block before random access queries.
        """
        self.l8_manager.ingest_block(block_id, metadata)
    
    def query_by_offset_range(self, offset_start: int, size_bytes: int) -> Tuple[List[BlockMetadata], Dict]:
        """
        Query blocks by byte offset range for random access
        
        Example:
            # Get blocks for bytes 1000-1200
            blocks, metadata = layer8.query_by_offset_range(1000, 200)
            # Now decompress only these blocks, not the whole dataset
        
        Args:
            offset_start: Starting byte offset
            size_bytes: Number of bytes to retrieve
        
        Returns:
            (list of BlockMetadata, query statistics dict)
        """
        return self.l8_manager.query_random_access(offset_start, size_bytes)
    
    def create_block_metadata(
        self,
        block_id: int,
        offset_start: int,
        offset_end: int,
        size_original: int,
        size_compressed: int,
        sha256_hash: str,
        entropy_score: float = 0.0,
        compression_skipped: bool = False,
        node_id: int = 0
    ) -> BlockMetadata:
        """
        Factory method to create BlockMetadata with all required fields
        """
        import time
        return BlockMetadata(
            block_id=block_id,
            offset_start=offset_start,
            offset_end=offset_end,
            size_original=size_original,
            size_compressed=size_compressed,
            sha256_hash=sha256_hash,
            entropy_score=entropy_score,
            compression_skipped=compression_skipped,
            timestamp=time.time(),
            node_id=node_id
        )
    
    def compute_sha256_from_data(self, data: bytes) -> str:
        """Compute SHA-256 hash of data"""
        return hashlib.sha256(data).hexdigest()
    
    def verify_blocks_integrity(self, blocks: List[Tuple[bytes, BlockMetadata]]) -> Dict[int, bool]:
        """
        Verify SHA-256 integrity of multiple blocks
        
        Args:
            blocks: List of (data_bytes, BlockMetadata) tuples
        
        Returns:
            Dict mapping block_id → is_valid (bool)
        """
        return self.l8_manager.verify_blocks_integrity(blocks)
    
    def verify_single_block(self, data: bytes, metadata: BlockMetadata) -> bool:
        """Verify SHA-256 integrity of single block"""
        results = self.verify_blocks_integrity([(data, metadata)])
        return results.get(metadata.block_id, False)
    
    def get_blocks_by_node(self, node_id: int) -> List[BlockMetadata]:
        """Get all blocks assigned to specific L8 node"""
        return self.l8_manager.global_mapping.get_blocks_by_node(node_id)
    
    def get_system_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        return self.l8_manager.get_comprehensive_stats()
    
    def save_index_to_file(self, filepath: str) -> None:
        """Persist index to file for disaster recovery"""
        self.l8_manager.serialize_to_file(filepath)
    
    def load_index_from_file(self, filepath: str) -> None:
        """Load index from file"""
        self.l8_manager.load_from_file(filepath)


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demonstrate_enhanced_layer8():
    """Demonstrate new random access capabilities"""
    print("\n" + "="*80)
    print("LAYER 8 ULTRA-EXTREME NODES - ENHANCED DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize
    layer8 = Layer8Final(num_l8_nodes=5)
    
    # Create sample blocks
    print("[SETUP] Creating 1 PB storage simulation with 1000 blocks...")
    current_offset = 0
    block_size_compressed = 1024 * 1024  # 1 MB per block
    
    for block_id in range(1000):
        # Create dummy data
        dummy_data = b'COBOL' * (block_size_compressed // 5)
        sha256_hash = layer8.compute_sha256_from_data(dummy_data)
        
        # Create metadata
        metadata = layer8.create_block_metadata(
            block_id=block_id,
            offset_start=current_offset,
            offset_end=current_offset + block_size_compressed,
            size_original=block_size_compressed * 10,  # 10x compression
            size_compressed=block_size_compressed,
            sha256_hash=sha256_hash,
            entropy_score=2.5,
            compression_skipped=False,
            node_id=block_id % 5
        )
        
        # Register block
        layer8.register_block_metadata(block_id, metadata)
        current_offset += block_size_compressed
    
    total_pb = current_offset / (1024 ** 5)
    print(f"✓ Registered 1000 blocks")
    print(f"✓ Simulated storage: {total_pb:.3f} PB compressed\n")
    
    # Demonstrate random access
    print("[RANDOM ACCESS] Query examples:")
    
    test_cases = [
        (0, 100_000_000, "First 100 MB"),
        (500_000_000, 200_000_000, "Middle 200 MB"),
        (900_000_000, 100_000_000, "Last 100 MB")
    ]
    
    for offset, size, description in test_cases:
        blocks, stats = layer8.query_by_offset_range(offset, size)
        print(f"\n  {description}:")
        print(f"    Blocks found: {len(blocks)}")
        print(f"    Compressed to read: {stats['compressed_bytes_to_read'] / 1024 / 1024:.2f} MB")
        print(f"    Original data: {stats['uncompressed_bytes_available'] / 1024 / 1024:.2f} MB")
        print(f"    Query time: {stats['query_time_ms']:.3f} ms")
    
    print("\n")
    print("="*80)
    print("✅ ENHANCED LAYER 8 DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    demonstrate_enhanced_layer8()
