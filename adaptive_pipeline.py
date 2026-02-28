"""
COBOL Protocol v1.5: Adaptive Pipeline & Stability Monitor
=========================================================

Features:
1. Dynamic layer selection based on hardware and data characteristics
2. Health monitoring and adaptive strategy switching
3. Fallback mechanisms with automatic recovery
4. Load balancing across available hardware
5. Performance profiling and optimization hints
6. Circuit breaker pattern for hardware failures
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import deque

from hardware_abstraction_layer import (
    HardwareContext, OptimizationStrategy, HardwareType
)
from hardware_optimized_layers import (
    HardwareOptimizedPipeline, HardwareOptimizedLayer
)

logger = logging.getLogger(__name__)


# ============================================================================
# HEALTH & STABILITY ENUMS
# ============================================================================


class HealthStatus(Enum):
    """Hardware/layer health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing recovery


# ============================================================================
# PERFORMANCE METRICS
# ============================================================================


@dataclass
class PerformanceMetrics:
    """Performance metrics for a layer or hardware."""
    name: str
    success_rate: float = 1.0          # 0-1
    avg_latency_ms: float = 0.0
    throughput_mbps: float = 0.0
    error_count: int = 0
    request_count: int = 0
    fallback_count: int = 0
    
    def get_health_score(self) -> float:
        """Compute health score (0-100)."""
        score = 100.0
        
        # Success rate impact (0-30 points)
        score -= (1.0 - self.success_rate) * 30
        
        # Error frequency impact (0-20 points)
        error_rate = self.error_count / max(self.request_count, 1)
        score -= error_rate * 20
        
        # Fallback frequency impact (0-20 points)
        fallback_rate = self.fallback_count / max(self.request_count, 1)
        score -= fallback_rate * 20
        
        return max(0, min(100, score))
    
    def get_health_status(self) -> HealthStatus:
        """Get health status based on score."""
        score = self.get_health_score()
        
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 50:
            return HealthStatus.DEGRADED
        elif score > 0:
            return HealthStatus.FAILING
        else:
            return HealthStatus.OFFLINE


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitBreaker:
    """Circuit breaker for hardware failures."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0):
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()
    
    def record_success(self):
        """Record successful operation."""
        with self.lock:
            self.failures = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' closed (recovered)")
    
    def record_failure(self):
        """Record failed operation."""
        with self.lock:
            self.failures += 1
            self.last_failure_time = time.time()
            
            if self.failures >= self.failure_threshold:
                if self.state != CircuitBreakerState.OPEN:
                    self.state = CircuitBreakerState.OPEN
                    logger.warning(f"Circuit breaker '{self.name}' opened (too many failures)")
    
    def is_available(self) -> bool:
        """Check if circuit is available."""
        with self.lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has passed
                elapsed = time.time() - self.last_failure_time
                if elapsed > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.failures = 0
                    logger.info(f"Circuit breaker '{self.name}' half-open (testing recovery)")
                    return True
                return False
            else:  # HALF_OPEN
                return True
    
    def get_state(self) -> CircuitBreakerState:
        """Get current state."""
        with self.lock:
            return self.state


# ============================================================================
# LAYER HEALTH MONITOR
# ============================================================================


class LayerHealthMonitor:
    """Monitor health of individual layers."""
    
    def __init__(self, layer: HardwareOptimizedLayer, 
                 window_size: int = 100):
        self.layer = layer
        self.metrics = PerformanceMetrics(name=f"Layer {layer.layer_num}")
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.lock = threading.Lock()
        self.circuit_breaker = CircuitBreaker(f"Layer {layer.layer_num}")
    
    def record_encode(self, duration_ms: float, bytes_processed: int, 
                     success: bool):
        """Record encode operation."""
        with self.lock:
            if success:
                self.circuit_breaker.record_success()
                self.metrics.request_count += 1
            else:
                self.circuit_breaker.record_failure()
                self.metrics.error_count += 1
            
            self.latencies.append(duration_ms)
            self._update_metrics(bytes_processed, duration_ms)
    
    def record_fallback(self):
        """Record fallback operation."""
        with self.lock:
            self.metrics.fallback_count += 1
    
    def _update_metrics(self, bytes_processed: int, duration_ms: float):
        """Update aggregated metrics."""
        if self.metrics.request_count > 0:
            self.metrics.avg_latency_ms = sum(self.latencies) / len(self.latencies)
            self.metrics.throughput_mbps = (bytes_processed / (duration_ms / 1000)) / 1_000_000 if duration_ms > 0 else 0
            self.metrics.success_rate = 1.0 - (self.metrics.error_count / self.metrics.request_count)
    
    def get_health_score(self) -> float:
        """Get health score."""
        with self.lock:
            return self.metrics.get_health_score()
    
    def get_health_status(self) -> HealthStatus:
        """Get health status."""
        with self.lock:
            return self.metrics.get_health_status()
    
    def is_available(self) -> bool:
        """Check if layer is available."""
        return self.circuit_breaker.is_available()
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get metrics snapshot."""
        with self.lock:
            return PerformanceMetrics(
                name=self.metrics.name,
                success_rate=self.metrics.success_rate,
                avg_latency_ms=self.metrics.avg_latency_ms,
                throughput_mbps=self.metrics.throughput_mbps,
                error_count=self.metrics.error_count,
                request_count=self.metrics.request_count,
                fallback_count=self.metrics.fallback_count
            )


# ============================================================================
# ADAPTIVE PIPELINE WITH STABILITY
# ============================================================================


class AdaptivePipeline(HardwareOptimizedPipeline):
    """Extended pipeline with adaptive strategies and health monitoring."""
    
    def __init__(self):
        super().__init__()
        self.monitors = [
            LayerHealthMonitor(layer)
            for layer in self.layers
        ]
        self.lock = threading.Lock()
        self.stats_history: deque = deque(maxlen=1000)
        self.optimization_hints: List[str] = []
    
    def compress_with_monitoring(self, data: bytes, 
                                adaptive: bool = True) -> Tuple[bytes, Dict[str, Any]]:
        """Compress with health monitoring and adaptation."""
        metadata = {
            "start_time": time.time(),
            "input_size": len(data),
            "per_layer_stats": [],
            "errors": [],
            "total_time_ms": 0.0,
        }
        
        current = data
        
        # Adaptive: Check if layers can be skipped
        if adaptive and len(self.layers) > 4:
            layer5 = self.layers[4]
            if isinstance(layer5, object) and hasattr(layer5, 'should_skip_expensive_layers'):
                if layer5.should_skip_expensive_layers(current):
                    logger.info("Layer 5: Skipping expensive layers due to high entropy")
                    # Skip layers 6-7
                    layer_range = range(5)
                else:
                    layer_range = range(6)
            else:
                layer_range = range(6)
        else:
            layer_range = range(6)
        
        # Process layers with monitoring
        for i in layer_range:
            layer = self.layers[i]
            monitor = self.monitors[i]
            
            if not monitor.is_available():
                logger.warning(f"Layer {i+1} unavailable, skipping")
                metadata["errors"].append(f"Layer {i+1} unavailable")
                continue
            
            try:
                start_time = time.time()
                current = layer.encode(current)
                duration_ms = (time.time() - start_time) * 1000
                
                bytes_processed = len(current) if isinstance(current, bytes) else current.nbytes
                
                monitor.record_encode(duration_ms, bytes_processed, success=True)
                
                metadata["per_layer_stats"].append({
                    "layer": i+1,
                    "duration_ms": duration_ms,
                    "size": bytes_processed,
                    "health": monitor.get_health_status().value
                })
                
            except Exception as e:
                logger.error(f"Layer {i+1} error: {e}")
                monitor.record_encode(0, 0, success=False)
                monitor.record_fallback()
                metadata["errors"].append(f"Layer {i+1}: {str(e)}")
        
        # Final layers
        try:
            start_time = time.time()
            current = self.layers[6].encode(current)
            duration_ms = (time.time() - start_time) * 1000
            self.monitors[6].record_encode(duration_ms, len(current), success=True)
        except Exception as e:
            logger.error(f"Layer 7 error: {e}")
            self.monitors[6].record_encode(0, 0, success=False)
            metadata["errors"].append(f"Layer 7: {str(e)}")
        
        try:
            start_time = time.time()
            current = self.layers[7].encode(current)
            duration_ms = (time.time() - start_time) * 1000
            self.monitors[7].record_encode(duration_ms, len(current), success=True)
        except Exception as e:
            logger.error(f"Layer 8 error: {e}")
            self.monitors[7].record_encode(0, 0, success=False)
            metadata["errors"].append(f"Layer 8: {str(e)}")
        
        metadata["total_time_ms"] = (time.time() - metadata["start_time"]) * 1000
        metadata["output_size"] = len(current) if isinstance(current, bytes) else current.nbytes
        
        with self.lock:
            self.stats_history.append(metadata)
            self.optimization_hints = self._generate_optimization_hints()
        
        return current, metadata
    
    def decompress_with_monitoring(self, data: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Decompress with health monitoring."""
        metadata = {
            "start_time": time.time(),
            "input_size": len(data),
            "errors": [],
        }
        
        current = data
        
        # Reverse order decompression
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            monitor = self.monitors[i]
            
            if not monitor.is_available():
                logger.warning(f"Layer {i+1} unavailable for decompression")
                metadata["errors"].append(f"Layer {i+1} unavailable")
                continue
            
            try:
                start_time = time.time()
                current = layer.decode(current)
                duration_ms = (time.time() - start_time) * 1000
                
                bytes_processed = len(current) if isinstance(current, bytes) else current.nbytes
                monitor.record_encode(duration_ms, bytes_processed, success=True)
                
                if isinstance(current, object) and hasattr(current, 'nbytes'):
                    current = bytes(current)
                
            except Exception as e:
                logger.error(f"Layer {i+1} decode error: {e}")
                monitor.record_encode(0, 0, success=False)
                metadata["errors"].append(f"Layer {i+1}: {str(e)}")
        
        metadata["output_size"] = len(current)
        metadata["total_time_ms"] = (time.time() - metadata["start_time"]) * 1000
        
        return current, metadata
    
    def _generate_optimization_hints(self) -> List[str]:
        """Analyze metrics and generate optimization hints."""
        hints = []
        
        for i, monitor in enumerate(self.monitors):
            metrics = monitor.get_metrics()
            
            if metrics.get_health_score() < 50:
                hints.append(f"Layer {i+1}: Consider disabling or reoptimizing "
                           f"(score: {metrics.get_health_score():.1f})")
            
            if metrics.fallback_count > metrics.request_count * 0.1:
                hints.append(f"Layer {i+1}: High fallback rate "
                           f"({metrics.fallback_count}/{metrics.request_count})")
            
            if metrics.throughput_mbps > 0 and metrics.throughput_mbps < 10:
                hints.append(f"Layer {i+1}: Low throughput ({metrics.throughput_mbps:.1f} MB/s), "
                           f"consider GPU acceleration")
        
        return hints
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health."""
        return {
            "timestamp": time.time(),
            "overall_score": sum(m.get_health_score() for m in self.monitors) / len(self.monitors),
            "layer_scores": {
                i+1: m.get_health_score()
                for i, m in enumerate(self.monitors)
            },
            "layer_statuses": {
                i+1: m.get_health_status().value
                for i, m in enumerate(self.monitors)
            },
            "available_layers": {
                i+1: m.is_available()
                for i, m in enumerate(self.monitors)
            },
            "optimization_hints": self.optimization_hints,
            "recent_stats": list(self.stats_history)[-5:] if self.stats_history else []
        }
    
    def get_detailed_metrics(self) -> Dict[int, Dict[str, Any]]:
        """Get detailed metrics for all layers."""
        return {
            i+1: {
                "metrics": monitor.get_metrics().__dict__,
                "health_status": monitor.get_health_status().value,
                "circuit_breaker_state": monitor.circuit_breaker.get_state().value,
            }
            for i, monitor in enumerate(self.monitors)
        }


# ============================================================================
# STABILITY MANAGER
# ============================================================================


class StabilityManager:
    """Manage system stability and recovery."""
    
    def __init__(self, pipeline: AdaptivePipeline):
        self.pipeline = pipeline
        self.health_check_interval = 30.0  # seconds
        self.last_health_check = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
    
    def check_health(self) -> bool:
        """Perform health check."""
        elapsed = time.time() - self.last_health_check
        
        if elapsed < self.health_check_interval:
            return True
        
        self.last_health_check = time.time()
        health = self.pipeline.get_system_health()
        
        if health["overall_score"] < 50:
            self.consecutive_failures += 1
            logger.warning(f"System health degraded: score={health['overall_score']:.1f}, "
                         f"failures={self.consecutive_failures}")
            
            if self.consecutive_failures >= self.max_consecutive_failures:
                logger.error("System health critical, triggering recovery")
                return False
        else:
            self.consecutive_failures = 0
        
        return True
    
    def get_recovery_action(self) -> Optional[Callable[[], None]]:
        """Get recommended recovery action."""
        health = self.pipeline.get_system_health()
        
        # Find worst-performing layer
        worst_layer = min(
            health["layer_scores"].items(),
            key=lambda x: x[1]
        )
        
        if worst_layer[1] < 30:
            logger.warning(f"Layer {worst_layer[0]} score too low, consider disabling")
            return lambda: logger.info(f"Disabling Layer {worst_layer[0]}")
        
        return None
    
    def apply_recovery(self):
        """Apply recommended recovery."""
        action = self.get_recovery_action()
        if action:
            try:
                action()
                logger.info("Recovery action applied")
            except Exception as e:
                logger.error(f"Recovery action failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test adaptive pipeline
    pipeline = AdaptivePipeline()
    stability = StabilityManager(pipeline)
    
    print(pipeline.get_hardware_summary())
    
    # Test compression
    test_data = b"COBOL" * 10000
    print(f"\nTest data: {len(test_data)} bytes")
    
    compressed, metadata = pipeline.compress_with_monitoring(test_data)
    print(f"Compressed: {len(compressed)} bytes")
    print(f"Time: {metadata['total_time_ms']:.2f}ms")
    print(f"Errors: {len(metadata['errors'])}")
    
    # System health
    print("\nSystem Health:")
    health = pipeline.get_system_health()
    print(f"Overall Score: {health['overall_score']:.1f}/100")
    for layer, score in sorted(health['layer_scores'].items()):
        status = health['layer_statuses'][layer]
        print(f"  Layer {layer}: {score:.1f}/100 ({status})")
    
    print("\nOptimization Hints:")
    for hint in health['optimization_hints']:
        print(f"  - {hint}")
