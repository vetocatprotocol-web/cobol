"""
COBOL v1.5: Hardware Optimization - Quick Reference for Developers

Fast lookup for common tasks and API usage.
"""

quick_ref = """
# COBOL v1.5 Hardware Optimization - Developer Quick Reference

## 🚀 5-Minute Quick Start

### Import & Initialize
```python
from hardware_abstraction_layer import get_hardware_context
from adaptive_pipeline import AdaptivePipeline

# Initialize (auto-detects hardware)
context = get_hardware_context()
pipeline = AdaptivePipeline()
```

### Compress with Monitoring
```python
data = b"COBOL Protocol Data"
compressed, metadata = pipeline.compress_with_monitoring(data)

print(f"Input: {len(data)} bytes")
print(f"Output: {len(compressed)} bytes")
print(f"Time: {metadata['total_time_ms']:.2f}ms")
```

### Check System Health
```python
health = pipeline.get_system_health()
print(f"Overall Score: {health['overall_score']:.1f}/100")

for layer, score in health['layer_scores'].items():
    print(f"Layer {layer}: {score:.1f}")
```

---

## 📋 API Reference

### Hardware Detection
```python
# Get hardware context (singleton)
context = get_hardware_context()

# Detect all hardware
devices = context.get_all_devices()

# Get primary device
primary = context.get_primary_device()
print(primary.hardware_type)        # HardwareType enum
print(primary.capability_level)     # ComputeCapability enum
print(primary.score())              # 0-100 score

# Check GPU/FPGA availability
has_gpu = context.can_use_gpu()     # bool
has_fpga = context.can_use_fpga()   # bool

# Get optimization strategies
strategies = context.get_all_layer_strategies()
print(strategies[1])  # OptimizationStrategy for layer 1

# Print summary
print(context.summary())  # Nice formatted summary
```

### Per-Layer Operations
```python
# Create individual layer
from hardware_optimized_layers import HardwareOptimizedLayer1

layer = HardwareOptimizedLayer1()

# Encode
data = b"test"
encoded = layer.encode(data)        # np.ndarray output

# Decode
decoded = layer.decode(encoded)     # np.ndarray output

# Get statistics
stats = layer.get_stats()
print(stats['calls'])               # Number of calls
print(stats['bytes'])               # Bytes processed
print(stats['fallbacks'])           # Fallback count
```

### Pipeline Operations
```python
from hardware_optimized_layers import HardwareOptimizedPipeline

pipeline = HardwareOptimizedPipeline()

# Compress (all 8 layers)
compressed = pipeline.compress(data)

# Decompress (all 8 layers, reverse order)
original = pipeline.decompress(compressed)

# Get statistics
stats = pipeline.get_compression_stats()
for layer_num, layer_stats in stats.items():
    print(f"Layer {layer_num}: {layer_stats}")
```

### Adaptive Pipeline (Recommended)
```python
from adaptive_pipeline import AdaptivePipeline, StabilityManager

pipeline = AdaptivePipeline()
stability = StabilityManager(pipeline)

# Compress with monitoring
compressed, metadata = pipeline.compress_with_monitoring(
    data,
    adaptive=True  # Enable entropy-based layer skipping
)

# Metadata contains:
# - start_time, input_size, output_size
# - per_layer_stats (list of per-layer stats)
# - errors (list of error messages if any)
# - total_time_ms

# Get detailed metrics
details = pipeline.get_detailed_metrics()
for layer_num, info in details.items():
    print(f"Layer {layer_num}:")
    print(f"  Metrics: {info['metrics']}")
    print(f"  Status: {info['health_status']}")
    print(f"  CB State: {info['circuit_breaker_state']}")

# Get system health
health = pipeline.get_system_health()
# Contains: overall_score, layer_scores, layer_statuses,
#           available_layers, optimization_hints, recent_stats

# Check stability
is_stable = stability.check_health()

# Get recovery action if needed
action = stability.get_recovery_action()
if action:
    action()  # Execute recovery
```

---

## 🔍 Common Tasks

### Task: Detect Hardware Capabilities
```python
context = get_hardware_context()
primary = context.get_primary_device()

print(f"Type: {primary.hardware_type.value}")
print(f"Score: {primary.score()}/100")

if primary.cpu_caps:
    print(f"CPU: {primary.cpu_caps.cores} cores @ {primary.cpu_caps.frequency_ghz} GHz")
    print(f"SIMD: {primary.cpu_caps.simd_type}")

if primary.gpu_caps:
    print(f"GPU: {primary.gpu_caps.device_name}")
    print(f"Memory: {primary.gpu_caps.total_memory_gb}GB")
    print(f"Compute: {primary.gpu_caps.compute_capability}")
```

### Task: List All Devices
```python
context = get_hardware_context()
devices = context.get_all_devices()

for i, device in enumerate(devices):
    print(f"{i+1}. {device.hardware_type.value} (score: {device.score()})")
```

### Task: Get Layer Strategy
```python
context = get_hardware_context()

for layer_num in range(1, 9):
    strategy = context.get_layer_strategy(layer_num)
    print(f"Layer {layer_num}: {strategy.value}")
```

### Task: Monitor Single Layer
```python
from hardware_optimized_layers import HardwareOptimizedLayer3
from adaptive_pipeline import LayerHealthMonitor

layer = HardwareOptimizedLayer3()
monitor = LayerHealthMonitor(layer)

# Do some work
for _ in range(10):
    data = b"test" * 100
    encoded = layer.encode(data)
    monitor.record_encode(5.0, len(encoded), success=True)

# Check health
print(f"Score: {monitor.get_health_score():.1f}/100")
print(f"Status: {monitor.get_health_status().value}")
print(f"Metrics: {monitor.get_metrics().__dict__}")
```

### Task: Handle Circuit Breaker
```python
monitor = LayerHealthMonitor(layer)

if monitor.circuit_breaker.is_available():
    try:
        result = layer.encode(data)
        monitor.circuit_breaker.record_success()
    except Exception as e:
        monitor.circuit_breaker.record_failure()
        if not monitor.circuit_breaker.is_available():
            print("Circuit opened, layer unavailable")
else:
    print(f"Layer circuit is {monitor.circuit_breaker.get_state().value}")
```

### Task: Process Large Data Adaptively
```python
pipeline = AdaptivePipeline()

# Process in chunks
chunk_size = 1024 * 1024  # 1 MB
results = []

for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    compressed, metadata = pipeline.compress_with_monitoring(chunk, adaptive=True)
    results.append(compressed)
    
    if metadata['errors']:
        print(f"Chunk {i}: Errors - {metadata['errors']}")

# Combine results
final = b"".join(results)
```

### Task: Get Optimization Recommendations
```python
pipeline = AdaptivePipeline()

# Do multiple compressions to gather statistics
for _ in range(10):
    _ = pipeline.compress_with_monitoring(test_data)

# Get health and hints
health = pipeline.get_system_health()

print("Optimization Hints:")
for hint in health['optimization_hints']:
    print(f"  - {hint}")

print("\nLayer Scores:")
for layer, score in health['layer_scores'].items():
    status = "✓" if score >= 80 else "⚠" if score >= 50 else "✗"
    print(f"  {status} Layer {layer}: {score:.1f}/100")
```

---

## 📊 Data Structures

### HardwareProfile
```python
{
    "hardware_type": HardwareType.CPU,      # enum
    "capability_level": ComputeCapability.STANDARD,  # enum
    "cpu_caps": CPUCapabilities(...),       # if CPU
    "gpu_caps": GPUCapabilities(...),       # if GPU
    "fpga_caps": FPGACapabilities(...),     # if FPGA
    "available_memory_gb": 64.0,
    "preferred_strategy": OptimizationStrategy.CPU_PARALLEL,
    "fallback_strategies": [...]
}
```

### PerformanceMetrics
```python
{
    "name": "Layer 5",
    "success_rate": 0.95,       # 0-1
    "avg_latency_ms": 15.5,
    "throughput_mbps": 50.0,
    "error_count": 5,
    "request_count": 100,
    "fallback_count": 2
}
```

### Compression Metadata
```python
{
    "start_time": 1709065400.5,
    "input_size": 10485760,
    "output_size": 2097152,
    "total_time_ms": 523.4,
    "per_layer_stats": [
        {
            "layer": 1,
            "duration_ms": 10.2,
            "size": 10485760,
            "health": "healthy"
        },
        ... (layers 2-8)
    ],
    "errors": []  # Empty if no errors
}
```

### System Health Report
```python
{
    "timestamp": 1709065500.0,
    "overall_score": 92.5,      # 0-100
    "layer_scores": {
        1: 100.0,
        2: 98.5,
        ...,
        8: 75.0
    },
    "layer_statuses": {
        1: "healthy",
        2: "healthy",
        ...,
        8: "degraded"
    },
    "available_layers": {
        1: true,
        2: true,
        ...,
        8: true
    },
    "optimization_hints": [
        "Layer 8: Low throughput (5.2 MB/s), consider GPU",
        "Layer 7: High fallback rate (3/20)"
    ],
    "recent_stats": [...]  # Last 5 compression sessions
}
```

---

## ⚙️ Configuration Parameters

### Circuit Breaker
```python
CircuitBreaker(
    name="Layer 1",
    failure_threshold=5,        # Open after N failures
    recovery_timeout=60.0       # Recovery test after N seconds
)
```

### Layer Health Monitor
```python
LayerHealthMonitor(
    layer=layer_instance,
    window_size=100            # Recent calls to track
)
```

### Stability Manager
```python
StabilityManager(
    pipeline=pipeline_instance,
    health_check_interval=30.0,      # Seconds
    max_consecutive_failures=3       # Before critical
)
```

### Adaptive Layer 5
```python
layer5.entropy_threshold = 7.5   # Skip expensive if entropy > this
```

### Huffman Layer 7
```python
layer7.num_workers = 4           # Parallel workers
layer7.chunk_size = 64 * 1024    # 64 KB chunks
```

---

## 🐛 Debugging Tips

### Check Hardware Detection
```python
from hardware_abstraction_layer import get_hardware_context
context = get_hardware_context()
print(context.summary())
```

### Enable Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Profile Performance
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
compressed = pipeline.compress_with_monitoring(data)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Check Individual Layer Performance
```python
stats = pipeline.get_compression_stats()
for layer, data in sorted(stats.items()):
    throughput = (data['bytes'] / (data['avg_duration_ms'] / 1000)) / 1_000_000
    print(f"Layer {layer}: {throughput:.1f} MB/s ({data['calls']} calls, "
          f"{data['fallbacks']} fallbacks)")
```

### Monitor Health During Operation
```python
import time

while True:
    health = pipeline.get_system_health()
    if health['overall_score'] < 80:
        print(f"⚠️  System health: {health['overall_score']:.1f}/100")
        for hint in health['optimization_hints']:
            print(f"   {hint}")
    time.sleep(60)
```

---

## 🔗 File Structure

```
hardware_abstraction_layer.py      (500+ lines)
├─ HardwareDetector               # Hardware detection
├─ HardwareProfile                # Capability storage
├─ HardwareContext                # Global singleton
├─ HardwareOptimizer              # Strategy selection
└─ get_hardware_context()          # Singleton accessor

hardware_optimized_layers.py       (700+ lines)
├─ HardwareOptimizedLayer1-8       # Per-layer classes
├─ HardwareOptimizedPipeline       # Unified pipeline
└─ encode/decode                   # Common interface

adaptive_pipeline.py               (600+ lines)
├─ PerformanceMetrics             # Metrics storage
├─ CircuitBreaker                 # Failure detection
├─ LayerHealthMonitor             # Per-layer health
├─ AdaptivePipeline               # Monitoring pipeline
└─ StabilityManager               # Overall health

tests/test_hardware_optimization.py (300+ lines)
├─ TestHardwareDetection          # 6 tests
├─ TestLayer1-8                   # 40+ tests
├─ TestLayerHealthMonitor         # 6 tests
├─ TestCircuitBreaker             # 6 tests
├─ TestAdaptivePipeline           # 8 tests
├─ TestStabilityManager           # 6 tests
└─ TestIntegration                # 4 tests

HARDWARE_OPTIMIZATION_GUIDE.md     (18.8 KB)
├─ Executive Summary
├─ Architecture Overview
├─ Component Details
├─ Health Monitoring
├─ Performance Metrics
├─ Configuration
├─ Testing Guide
├─ Deployment Guide
├─ Benchmarks
├─ Troubleshooting
└─ Future Roadmap
```

---

## 🎯 Use Cases

### Use Case 1: Basic Compression
```python
from adaptive_pipeline import AdaptivePipeline

pipeline = AdaptivePipeline()
compressed = pipeline.compress(data)  # Simple, no monitoring
```

### Use Case 2: Monitored Compression (Recommended)
```python
from adaptive_pipeline import AdaptivePipeline

pipeline = AdaptivePipeline()
compressed, metadata = pipeline.compress_with_monitoring(data)

if metadata['errors']:
    log.error(f"Compression errors: {metadata['errors']}")
```

### Use Case 3: Production with Stability Checks
```python
from adaptive_pipeline import AdaptivePipeline, StabilityManager

pipeline = AdaptivePipeline()
stability = StabilityManager(pipeline)

if not stability.check_health():
    raise RuntimeError("System health critical")

compressed, metadata = pipeline.compress_with_monitoring(data)
```

### Use Case 4: Real-Time Monitoring Daemon
```python
from adaptive_pipeline import AdaptivePipeline
import logging
import time

pipeline = AdaptivePipeline()

while True:
    health = pipeline.get_system_health()
    
    if health['overall_score'] < 50:
        logging.warning(f"System degraded: {health['overall_score']:.1f}/100")
        
        for hint in health['optimization_hints']:
            logging.warning(f"Hint: {hint}")
    
    time.sleep(300)  # Every 5 minutes
```

---

## 📖 More Information

- Full guide: `HARDWARE_OPTIMIZATION_GUIDE.md`
- Implementation: `hardware_optimized_layers.py`
- Monitoring: `adaptive_pipeline.py`
- Tests: `tests/test_hardware_optimization.py`
- Summary: `LAYER_OPTIMIZATION_HARDWARE_SUMMARY.md`

---

**Last Updated:** February 28, 2026  
**Version:** 1.5.0  
**Status:** Production-Ready
"""

# Write file
with open("/workspaces/cobol/HARDWARE_QUICK_REFERENCE.md", "w") as f:
    f.write(quick_ref)

print("Quick reference created: HARDWARE_QUICK_REFERENCE.md")
