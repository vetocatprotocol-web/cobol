"""
COBOL Protocol - Dual Mode Engine (Legacy + Bridge)
Supports both OptimizedLayer5/6/7 (legacy) and new L1-L8 Protocol Bridge

This wrapper enables backward compatibility while introducing new architecture.
"""

from typing import Tuple, Optional, Union
import enum


class CompressionMode(enum.Enum):
    LEGACY = "legacy"  # Uses layer5_optimized, layer6_optimized, layer7_optimized
    BRIDGE = "bridge"  # Uses new L1-L8 protocol bridge


class DualModeEngine:
    """
    Engine that supports both legacy and new protocol bridge implementations.
    Automatically selects mode based on configuration or explicit choice.
    """
    
    def __init__(self, mode: CompressionMode = CompressionMode.LEGACY):
        self.mode = mode
        self._init_legacy_layers()
        self._init_bridge_layers()
    
    def _init_legacy_layers(self):
        """Initialize legacy layer5/6/7_optimized implementations"""
        try:
            from layer5_optimized import OptimizedLayer5Pipeline
            from layer6_optimized import OptimizedLayer6Pipeline
            from layer7_optimized import OptimizedLayer7Pipeline
            
            self.l5_legacy = OptimizedLayer5Pipeline()
            self.l6_legacy = OptimizedLayer6Pipeline()
            self.l7_legacy = OptimizedLayer7Pipeline()
            self.legacy_available = True
        except ImportError as e:
            print(f"Warning: Legacy layers not available: {e}")
            self.legacy_available = False
    
    def _init_bridge_layers(self):
        """Initialize new L1-L8 protocol bridge implementation"""
        try:
            from protocol_bridge import ProtocolBridge, TypedBuffer, ProtocolLanguage
            from layer1_semantic import Layer1Semantic
            from layer2_structural import Layer2Structural
            from layer3_delta import Layer3Delta
            from layer4_binary import Layer4Binary
            from layer5_recursive import Layer5Recursive
            from layer6_recursive import Layer6Recursive
            from layer7_bank import Layer7Bank
            from layer8_final import Layer8Final
            
            self.bridge = ProtocolBridge([
                Layer1Semantic(), Layer2Structural(), Layer3Delta(), Layer4Binary(),
                Layer5Recursive(), Layer6Recursive(), Layer7Bank(), Layer8Final()
            ])
            self.bridge_available = True
        except ImportError as e:
            print(f"Warning: Protocol bridge not available: {e}")
            self.bridge_available = False
    
    def compress(self, data: bytes) -> bytes:
        """
        Compress data using selected mode.
        
        Args:
            data: Raw bytes to compress
        
        Returns:
            Compressed bytes
        """
        if self.mode == CompressionMode.LEGACY:
            return self._compress_legacy(data)
        elif self.mode == CompressionMode.BRIDGE:
            return self._compress_bridge(data)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def decompress(self, data: bytes) -> bytes:
        """
        Decompress data using selected mode.
        
        Args:
            data: Compressed bytes
        
        Returns:
            Original uncompressed bytes
        """
        if self.mode == CompressionMode.LEGACY:
            return self._decompress_legacy(data)
        elif self.mode == CompressionMode.BRIDGE:
            return self._decompress_bridge(data)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _compress_legacy(self, data: bytes) -> bytes:
        """Compress using legacy layer5/6/7_optimized pipeline"""
        if not self.legacy_available:
            raise RuntimeError("Legacy layers not available")
        
        # L5 -> L6 -> L7 pipeline
        l5_comp = self.l5_legacy.compress(data)
        l6_comp = self.l6_legacy.compress(l5_comp)
        l7_comp = self.l7_legacy.compress(l6_comp)
        
        return l7_comp
    
    def _decompress_legacy(self, data: bytes) -> bytes:
        """Decompress using legacy layer5/6/7_optimized pipeline"""
        if not self.legacy_available:
            raise RuntimeError("Legacy layers not available")
        
        # L7 -> L6 -> L5 reverse pipeline
        l7_decomp = self.l7_legacy.decompress(data)
        l6_decomp = self.l6_legacy.decompress(l7_decomp)
        l5_decomp = self.l5_legacy.decompress(l6_decomp)
        
        return l5_decomp
    
    def _compress_bridge(self, data: bytes) -> bytes:
        """Compress using new L1-L8 protocol bridge"""
        if not self.bridge_available:
            raise RuntimeError("Protocol bridge not available")
        
        from protocol_bridge import TypedBuffer, ProtocolLanguage
        
        # Convert bytes to text for L1 input
        text = data.decode('utf-8', errors='ignore')
        buffer = TypedBuffer.create(text, ProtocolLanguage.L1_SEM, str)
        
        # Compress through bridge
        compressed = self.bridge.compress(buffer)
        
        # Return as bytes (L8 output is PIC X string)
        if isinstance(compressed.data, str):
            return compressed.data.encode('utf-8')
        else:
            return compressed.data
    
    def _decompress_bridge(self, data: bytes) -> bytes:
        """Decompress using new L1-L8 protocol bridge"""
        if not self.bridge_available:
            raise RuntimeError("Protocol bridge not available")
        
        from protocol_bridge import TypedBuffer, ProtocolLanguage
        
        # Decode L8 input (PIC X string)
        if isinstance(data, bytes):
            pic_x_str = data.decode('utf-8', errors='ignore')
        else:
            pic_x_str = data
        
        buffer = TypedBuffer.create(pic_x_str, ProtocolLanguage.L8_COBOL, str)
        
        # Decompress through bridge
        decompressed = self.bridge.decompress(buffer)
        
        # Return as bytes
        if isinstance(decompressed.data, str):
            return decompressed.data.encode('utf-8')
        else:
            return decompressed.data
    
    def switch_mode(self, mode: CompressionMode):
        """Switch between legacy and bridge mode"""
        if mode == CompressionMode.LEGACY and not self.legacy_available:
            raise RuntimeError("Legacy layers not available")
        if mode == CompressionMode.BRIDGE and not self.bridge_available:
            raise RuntimeError("Protocol bridge not available")
        
        self.mode = mode
        print(f"Switched to {mode.value.upper()} compression mode")
    
    def get_mode(self) -> str:
        """Get current compression mode"""
        return self.mode.value
    
    def get_statistics(self) -> dict:
        """Get compression statistics for current mode"""
        stats = {
            "mode": self.mode.value,
            "legacy_available": self.legacy_available,
            "bridge_available": self.bridge_available,
        }
        
        if self.mode == CompressionMode.LEGACY and self.legacy_available:
            stats["l5_stats"] = self.l5_legacy.get_statistics()
        
        return stats


# Convenience functions for backward compatibility
_default_engine = None

def get_engine(mode: CompressionMode = CompressionMode.LEGACY) -> DualModeEngine:
    """Get or create default dual-mode engine"""
    global _default_engine
    if _default_engine is None:
        _default_engine = DualModeEngine(mode)
    return _default_engine


def compress_pipeline(data: bytes, mode: CompressionMode = CompressionMode.LEGACY) -> bytes:
    """Compress data using specified mode. Legacy: L5->L6->L7. Bridge: L1->L8."""
    engine = get_engine(mode)
    return engine.compress(data)


def decompress_pipeline(data: bytes, mode: CompressionMode = CompressionMode.LEGACY) -> bytes:
    """Decompress data using specified mode."""
    engine = get_engine(mode)
    return engine.decompress(data)
