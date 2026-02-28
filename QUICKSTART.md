# COBOL v1.5 FPGA Control - Quick Start Guide

**Status:** ✅ Full Stack Implementation Complete

---

## 📋 What's Included

✓ **Python Control Layer** (`fpga_controller.py`)
- FPGAController class for device management
- FPGASimulator for testing without hardware
- FPGACluster for 5,000-device orchestration
- Real-time metrics collection

✓ **REST API Server** (`fpga_api_server.py`)
- FastAPI with async/WebSocket support
- Device management endpoints
- CAM configuration & Huffman table loading
- Metrics & health status monitoring

✓ **Real-time Dashboard** (`dashboard.html`)
- Live metrics visualization with Chart.js
- Device & cluster health monitoring
- WebSocket-driven updates (500ms/1000ms)
- Control panel for device management

✓ **Test Suite** (`test_api_client.py`)
- 18 pytest methods in fpga_controller tests
- API endpoint testing
- WebSocket stress testing
- Performance benchmarks

✓ **RTL Testbenches** (`tb/*.sv`)
- SystemVerilog CAM_BANK testbench
- DECOMPRESSOR pipeline testbench
- Timing & throughput validation

✓ **Documentation**
- API Reference (API_DOCUMENTATION.md)
- Quick Start (this file)
- Architecture overview

---

## 🚀 Getting Started (5 minutes)

### Option 1: Automated Quick Start

```bash
# Make script executable
chmod +x start_cobol.sh

# Run the setup script
./start_cobol.sh
```

The script will:
1. ✓ Check Python installation
2. ✓ Install dependencies
3. ✓ Start API server
4. ✓ Open dashboard
5. ✓ Run tests

### Option 2: Manual Setup

#### Step 1: Install Dependencies
```bash
pip install -r requirements_api.txt
```

#### Step 2: Start API Server
```bash
python3 fpga_api_server.py 0.0.0.0 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     COBOL v1.5 FPGA Control API Starting...
INFO:     Cluster initialized: FPGACluster(num_devices=5000)
```

#### Step 3: Open Dashboard
```
http://localhost:8000/dashboard.html
```

#### Step 4: Run Tests (in another terminal)
```bash
python3 test_api_client.py
```

---

## 🎯 Quick Examples

### Python API Client

```python
from test_api_client import FPGAAPIClient

# Create client
client = FPGAAPIClient("http://localhost:8000/api")

# 1. List devices
devices = client.list_devices(limit=32)
print(f"Initialized: {devices['initialized_devices']} devices")

# 2. Configure CAM patterns
pattern = b"my_data_pattern"
client.configure_cam(device_id=0, pattern=pattern, match_id=42)
client.flush_cam(device_id=0)

# 3. Load Huffman table
client.load_huffman(device_id=0, chunk_id=0)

# 4. Get metrics
metrics = client.get_cluster_metrics()
print(f"Cluster Health: {metrics['cluster_health_score']:.0f}/100")

# 5. Stream real-time metrics
import asyncio
from test_api_client import WebSocketTester

async def monitor():
    samples = await WebSocketTester.stream_device_metrics(0, duration=5.0)
    for s in samples:
        print(f"Input: {s['input_rate_gb_s']:.1f} GB/s, "
              f"Hit Rate: {s['cam_hit_rate']:.1f}%")

asyncio.run(monitor())
```

### cURL Examples

```bash
# List devices
curl http://localhost:8000/api/devices

# Get device status
curl http://localhost:8000/api/devices/0

# Get cluster metrics
curl http://localhost:8000/api/metrics/cluster

# Configure CAM
curl -X POST http://localhost:8000/api/devices/0/cam/config \
  -H "Content-Type: application/json" \
  -d '{"pattern": "48656c6c6f", "match_id": 1, "chunk_size": 16}'

# Flush CAM
curl -X POST http://localhost:8000/api/devices/0/cam/flush

# Load Huffman table
curl -X POST http://localhost:8000/api/devices/0/huffman/load \
  -H "Content-Type: application/json" \
  -d '{
    "chunk_id": 0, 
    "table": {
      "code_lengths": [3,3,3,...,8],
      "code_values": [0,1,2,...,255],
      "symbols": [0,1,2,...,255]
    }
  }'
```

### JavaScript WebSocket

```javascript
// Connect to device metrics
const ws = new WebSocket('ws://localhost:8000/ws/metrics/0');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Throughput: ${data.input_rate_gb_s.toFixed(1)} GB/s`);
  console.log(`Hit Rate: ${data.cam_hit_rate.toFixed(1)}%`);
  console.log(`Health: ${data.pipeline_depth} cycles`);
};

ws.onerror = (error) => console.error('Error:', error);

// Connect to cluster metrics
const clusterWS = new WebSocket('ws://localhost:8000/ws/cluster');
clusterWS.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Cluster Health: ${data.cluster_health_score.toFixed(0)}/100`);
};
```

---

## 📊 Dashboard Features

### Tab 1: Cluster Metrics
- Total devices & health score
- Cluster-wide CAM entries & Huffman tables
- Active issues with severity indicators
- Real-time health gauge

### Tab 2: Device Metrics
- Per-device throughput trends (60-point graph)
- CAM hit rate visualization
- Resource utilization (HBM/BRAM)
- Device status & health

### Tab 3: Control Panel
- Device selector dropdown
- Soft reset button
- CAM flush button
- Operation log with timestamps

---

## 🧪 Running Tests

### Full Test Suite
```bash
python3 test_api_client.py
```

Runs:
- ✓ API information retrieval
- ✓ Health check
- ✓ Device listing & status
- ✓ CAM configuration workflow
- ✓ Huffman table loading
- ✓ Cluster metrics
- ✓ Device history
- ✓ WebSocket device streaming
- ✓ WebSocket cluster streaming
- ✓ API stress test (50 RPS)
- ✓ WebSocket load test (10 concurrent)

### Run Specific Tests
```bash
# Only REST API tests
python3 -c "from test_api_client import *; test_api_info(); test_health_check()"

# Only WebSocket tests
python3 -c "import asyncio; from test_api_client import *; asyncio.run(test_websocket_device())"

# Only stress tests
python3 -c "from test_api_client import *; stress_test_api()"
```

### pytest Integration Tests
```bash
# Run all integration tests
pytest tests/test_integration.py -v

# Run specific test class
pytest tests/test_integration.py::TestCAMDictionary -v

# Run with coverage
pytest tests/test_integration.py --cov=fpga_controller
```

---

## 🔧 Configuration

### API Server Options

```bash
# Custom host & port
python3 fpga_api_server.py 192.168.1.100 9000

# Default: 0.0.0.0:8000
python3 fpga_api_server.py
```

### Dashboard Configuration

Edit `dashboard.html` to change:
```javascript
// Line ~25
const API_BASE = 'http://localhost:8000/api';
const WS_BASE = 'ws://localhost:8000';
```

### Cluster Configuration

Edit `fpga_api_server.py` to change:
```python
# Line ~35
def get_cluster(num_devices: int = 5000) -> FPGACluster:
    # Change 5000 to desired cluster size
    _cluster = FPGACluster(num_devices=5000)
```

---

## 📈 Performance Expectations

### Throughput
- REST API: 1000+ RPS
- WebSocket: 50+ concurrent connections
- CAM flush: 512-4096 entries/POST
- Metrics collection: 50+ samples/sec

### Latency
- GET endpoints: <10 ms
- POST endpoints: 10-50 ms
- WebSocket latency: <1 ms (local)
- History retrieval: <20 ms (3600 samples)

### Resource Usage
- Memory: ~50 MB per 32 devices
- CPU: <5% idle
- Network: ~5 KB/s per device

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Find process on port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Try different port
python3 fpga_api_server.py 0.0.0.0 9000
```

### WebSocket Connection Failed
```javascript
// Check 1: Is API running?
curl http://localhost:8000/health

// Check 2: Can reach WebSocket?
// Browser DevTools → Network → WS → check connection status
```

### Dashboard Not Loading
1. Check browser console (F12) for CORS errors
2. Verify API_BASE and WS_BASE match your server
3. Clear browser cache and reload

### Tests Failing
```bash
# Ensure API is running first
python3 fpga_api_server.py 0.0.0.0 8000

# Then in another terminal
python3 test_api_client.py
```

---

## 📚 Documentation

- **Full API Reference:** [API_DOCUMENTATION.md](API_DOCUMENTATION.md)
- **Architecture Guide:** Inside fpga_controller.py docstrings
- **RTL Specifications:** See RTL design files
- **Test Coverage:** In tests/test_integration.py

---

## 🔗 API Quick Reference

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/devices` | List FPGA devices |
| GET | `/api/devices/{id}` | Get device status |
| POST | `/api/devices/{id}/cam/config` | Configure CAM |
| POST | `/api/devices/{id}/cam/flush` | Flush CAM to device |
| POST | `/api/devices/{id}/huffman/load` | Load Huffman table |
| GET | `/api/metrics/cluster` | Cluster metrics |
| GET | `/api/metrics/device/{id}/history` | Device history |
| GET | `/health` | API health |
| GET | `/api/info` | API information |
| WS | `/ws/metrics/{id}` | Device metrics stream |
| WS | `/ws/cluster` | Cluster metrics stream |

---

## 📞 Support

### Check Logs
```bash
# API server logs (if running in foreground)
python3 fpga_api_server.py

# Or view log file
tail -f logs/*
```

### Test Connectivity
```bash
# Check API is responding
curl -v http://localhost:8000/health

# Check dashboard loads
curl http://localhost:8000/dashboard.html
```

### Get API Documentation
```
http://localhost:8000/docs  # Swagger UI
http://localhost:8000/redoc # ReDoc
```

---

## 🎓 Next Steps

1. **Learn the API:** Read API_DOCUMENTATION.md
2. **Explore Dashboard:** Open dashboard.html in browser
3. **Test Endpoints:** Run test_api_client.py
4. **Write Custom Code:** Use FPGAAPIClient in your application
5. **Deploy:** See API_DOCUMENTATION.md Deployment section

---

## 📦 File Structure

```
.
├── fpga_controller.py          # Core Python control layer
├── fpga_api_server.py          # FastAPI + WebSocket server
├── dashboard.html              # Web dashboard
├── test_api_client.py          # API test suite
├── tests/
│   └── test_integration.py     # pytest integration tests
├── tb/
│   ├── cam_bank_tb.sv         # CAM testbench
│   └── decompressor_tb.sv     # Decompressor testbench
├── API_DOCUMENTATION.md        # Full API reference
├── QUICKSTART.md              # This file
├── requirements_api.txt        # Python dependencies
└── start_cobol.sh             # Automated startup script
```

---

## ✅ Verification Checklist

Before running in production:

- [ ] All dependencies installed: `pip list | grep -E "fastapi|uvicorn|websockets"`
- [ ] API server starts: `python3 fpga_api_server.py`
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Dashboard loads: Open browser to http://localhost:8000/dashboard.html
- [ ] Tests pass: `python3 test_api_client.py`
- [ ] WebSocket works: Check network tab in browser DevTools
- [ ] Metrics update: Watch graphs in dashboard update in real-time

---

**Version:** 1.5.0  
**Last Updated:** 2024-01-15  
**Status:** ✅ Production Ready
