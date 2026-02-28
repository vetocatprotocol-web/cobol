# COBOL v1.5 FPGA Control - REST API & Dashboard Documentation

## Overview

This document describes the complete REST API and WebSocket interfaces for the COBOL v1.5 FPGA cluster control system. The API provides real-time monitoring, device management, and configuration capabilities for a 5,000-device FPGA cluster.

**Architecture:**
- **Backend:** FastAPI + uvicorn (async Python)
- **Frontend:** HTML5 + Chart.js dashboard
- **Real-time:** WebSocket for metrics streaming
- **Testing:** Test client with full endpoint coverage

---

## Quick Start

### 1. Install Dependencies

```bash
pip install fastapi uvicorn websockets aiofiles
```

### 2. Start API Server

```bash
python fpga_api_server.py 0.0.0.0 8000
```

Output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     COBOL v1.5 FPGA Control API Starting...
INFO:     Cluster initialized: FPGACluster(num_devices=5000, initialized=32)
```

### 3. Open Dashboard

```
http://localhost:8000/dashboard.html
```

### 4. Run Tests

```bash
python test_api_client.py
```

---

## API Endpoints

### Device Management

#### List Devices
```
GET /api/devices?limit=32
```

**Response:**
```json
{
  "total_devices": 5000,
  "initialized_devices": 32,
  "devices": [
    {
      "device_id": 0,
      "state": "IDLE",
      "health_score": 85,
      "cam_entries": 256,
      "huffman_tables": 10
    }
  ]
}
```

#### Get Device Status
```
GET /api/devices/{device_id}
```

**Response:**
```json
{
  "device_id": 0,
  "state": "IDLE",
  "health_score": 85,
  "issues": [],
  "timestamp": "2024-01-15T12:34:56.789Z",
  "metrics": {
    "input_rate_gb_s": 25.3,
    "decomp_rate_gb_s": 12500,
    "output_rate_gb_s": 12.5,
    "cam_hit_rate": 78.5,
    "compression_ratio": 500,
    "pipeline_depth": 16,
    "active_clients": 3
  },
  "resources": {
    "hbm_utilization_mb": 15000,
    "bram_utilization_pct": 45.2,
    "cam_entries": 256,
    "huffman_tables": 10
  }
}
```

---

### CAM Configuration

#### Configure CAM Entry
```
POST /api/devices/{device_id}/cam/config
```

**Request Body:**
```json
{
  "pattern": "48656c6c6f",
  "match_id": 42,
  "chunk_size": 16
}
```

**Response:**
```json
{
  "status": "configured",
  "device_id": 0,
  "match_id": 42,
  "pattern_len": 5
}
```

#### Flush CAM Configuration
```
POST /api/devices/{device_id}/cam/flush
```

**Response:**
```json
{
  "status": "flushed",
  "device_id": 0,
  "entries_written": 512,
  "total_cam_entries": 512
}
```

**Example Workflow:**

```python
# 1. Configure 100 patterns
for i in range(100):
    pattern = f"pattern_{i:04d}".encode()
    client.configure_cam(device_id=0, pattern=pattern, match_id=i)

# 2. Flush all to device
result = client.flush_cam(device_id=0)
print(f"Wrote {result['entries_written']} entries")

# 3. Verify
status = client.get_device_status(device_id=0)
print(f"CAM entries: {status['resources']['cam_entries']}")
```

---

### Huffman Decompression

#### Load Huffman Table
```
POST /api/devices/{device_id}/huffman/load
```

**Request Body:**
```json
{
  "chunk_id": 0,
  "table": {
    "code_lengths": [3, 3, 3, ..., 8],
    "code_values": [0, 1, 2, ..., 255],
    "symbols": [0, 1, 2, ..., 255]
  }
}
```

**Response:**
```json
{
  "status": "loaded",
  "device_id": 0,
  "chunk_id": 0,
  "total_huffman_tables": 1
}
```

---

### Metrics & Monitoring

#### Get Cluster Metrics
```
GET /api/metrics/cluster
```

**Response:**
```json
{
  "timestamp": "2024-01-15T12:34:56.789Z",
  "cluster_health_score": 82.5,
  "total_devices": 5000,
  "total_cam_entries": 128000,
  "total_huffman_tables": 5000,
  "critical_issues": 0
}
```

#### Get Device Metrics History
```
GET /api/metrics/device/{device_id}/history?last_n=100
```

**Response:**
```json
{
  "device_id": 0,
  "samples": 100,
  "metrics": [
    {
      "timestamp": "2024-01-15T12:34:50",
      "input_rate_gb_s": 24.8,
      "decomp_rate_gb_s": 12400,
      "output_rate_gb_s": 12.4,
      "cam_hit_rate": 77.2,
      "compression_ratio": 496
    }
  ]
}
```

---

### Health & Information

#### Health Check
```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:34:56.789Z",
  "cluster_health": 82.5,
  "active_ws_connections": 5
}
```

#### API Information
```
GET /api/info
```

**Response:**
```json
{
  "name": "COBOL v1.5 FPGA Control API",
  "version": "1.5.0",
  "description": "REST + WebSocket API for FPGA cluster management",
  "endpoints": {
    "devices": "/api/devices",
    "device_status": "/api/devices/{device_id}",
    "cam_config": "POST /api/devices/{device_id}/cam/config",
    ...
  }
}
```

---

## WebSocket Endpoints

### Device Metrics Stream
```
ws://host:8000/ws/metrics/{device_id}
```

**Message Format (emitted every 500ms):**
```json
{
  "type": "metrics",
  "device_id": 0,
  "timestamp": "2024-01-15T12:34:56.789",
  "input_rate_gb_s": 25.3,
  "decomp_rate_gb_s": 12500,
  "output_rate_gb_s": 12.5,
  "cam_hit_rate": 78.5,
  "compression_ratio": 500,
  "hbm_utilization_pct": 75.0,
  "bram_utilization_pct": 45.2,
  "pipeline_depth": 16,
  "active_clients": 3
}
```

**JavaScript Example:**

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics/0');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Input rate: ${data.input_rate_gb_s.toFixed(1)} GB/s`);
  console.log(`Hit rate: ${data.cam_hit_rate.toFixed(1)}%`);
};

ws.onerror = (error) => console.error('WebSocket error:', error);
```

### Cluster Metrics Stream
```
ws://host:8000/ws/cluster
```

**Message Format (emitted every 1000ms):**
```json
{
  "type": "cluster_metrics",
  "timestamp": "2024-01-15T12:34:56.789",
  "total_devices": 5000,
  "healthy_devices": 4985,
  "cluster_health_score": 82.5,
  "total_cam_entries": 128000,
  "total_huffman_tables": 5000,
  "issues_count": 15
}
```

---

## Dashboard Features

### Real-time Monitoring
- **Cluster Overview:** Total devices, CAM entries, Huffman tables
- **Health Dashboard:** Cluster health score with issue tracking
- **Device Metrics:** Per-device throughput, hit rate, resource utilization
- **History Graphs:** 60-point sliding window of metrics

### Device Control
- **Soft Reset:** Recover from error states
- **Flush CAM:** Write queued patterns to device
- **View History:** Load and display historical metrics
- **Operation Log:** Real-time event tracking

### Real-time Updates
- WebSocket-driven metrics streaming
- 500ms device update rate
- 1000ms cluster update rate
- Auto-refresh on tabs

---

## Client Library

### Python Example

```python
from test_api_client import FPGAAPIClient
import asyncio
from test_api_client import WebSocketTester

# Create client
client = FPGAAPIClient("http://localhost:8000/api")

# List devices
devices = client.list_devices(limit=32)
print(f"Initialized: {devices['initialized_devices']} devices")

# Configure CAM
pattern = b"my_pattern"
client.configure_cam(0, pattern, match_id=100)
client.flush_cam(0)

# Load Huffman table
client.load_huffman(0, chunk_id=0)

# Get metrics
metrics = client.get_cluster_metrics()
print(f"Health: {metrics['cluster_health_score']:.0f}/100")

# Stream device metrics
async def monitor():
    samples = await WebSocketTester.stream_device_metrics(0, duration=10.0)
    for s in samples:
        print(f"{s['input_rate_gb_s']:.1f} GB/s")

asyncio.run(monitor())
```

---

## Performance Characteristics

### Throughput
- **REST endpoints:** 1000+ RPS per single-threaded client
- **WebSocket:** 50+ concurrent connections per server (uvicorn)
- **Metrics collection:** 3600 samples per device (1 hour at 1 Hz)
- **CAM flush:** 512-4096 entries in single POST

### Latency
- **GET endpoints:** <10 ms (without network)
- **POST endpoints:** 10-50 ms
- **WebSocket message latency:** <1 ms (local network)
- **Metrics history retrieval:** <20 ms for 3600 samples

### Resource Usage
- **Memory:** ~50 MB per 32 initialized devices
- **CPU:** <5% during idle (metrics collection interval 1.0s)
- **Network:** ~5 KB/s per real-time device connection

---

## Error Handling

### HTTP Status Codes
- **200:** Success
- **404:** Device not found
- **400:** Invalid request (bad CAM config, invalid Huffman table)
- **500:** Internal server error

### Error Response Format
```json
{
  "detail": "Device 500 not found",
  "error_type": "HTTPException"
}
```

### WebSocket Errors
- **404 Close Code:** Device not found (connection rejected)
- **1011 Close Code:** Server error during streaming

---

## Deployment Guide

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY fpga_controller.py fpga_api_server.py ./
COPY dashboard.html ./

EXPOSE 8000
CMD ["python", "fpga_api_server.py", "0.0.0.0", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  fpga-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NUM_DEVICES=5000
    volumes:
      - ./logs:/app/logs
```

**Run:**
```bash
docker-compose up -d
```

### Kubernetes Deployment

**fpga-api-deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fpga-api
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fpga-api
  template:
    metadata:
      labels:
        app: fpga-api
    spec:
      containers:
      - name: fpga-api
        image: cobol/fpga-api:1.5.0
        ports:
        - containerPort: 8000
        env:
        - name: NUM_DEVICES
          value: "5000"
        resources:
          requests:
            memory: "500Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2"
---
apiVersion: v1
kind: Service
metadata:
  name: fpga-api
spec:
  type: LoadBalancer
  ports:
  - port: 8000
    targetPort: 8000
  selector:
    app: fpga-api
```

**Deploy:**
```bash
kubectl apply -f fpga-api-deployment.yaml
```

---

## Monitoring & Logging

### Log Levels
- **INFO:** API startup, WebSocket connections, requests
- **WARNING:** High latency operations
- **ERROR:** WebSocket disconnections, API errors
- **DEBUG:** Full request/response bodies (not production)

### Metrics to Monitor
- **cluster_health_score:** Target > 80
- **active_ws_connections:** Monitor connection churn
- **issues_count:** Alert if > 50
- **device_state:** Track transitions to ERROR

### Alerting

Recommended thresholds:
```
cluster_health_score < 70    → Page on-call
device_state == ERROR         → Alert
issues_count > 100            → Alert
ws_connection_errors > 10/min → Alert
API_latency > 100ms           → Warn
```

---

## Troubleshooting

### API Won't Start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill existing process
kill -9 <PID>

# Try different port
python fpga_api_server.py 0.0.0.0 9000
```

### WebSocket Connection Refused
```
Error: Connection refused at ws://localhost:8000/ws/metrics/0
```

**Solution:** Ensure API server is running and accessible
```bash
curl http://localhost:8000/health
```

### Dashboard Not Loading
- Check browser console for CORS errors
- Ensure API_BASE and WS_BASE are correct in dashboard.html
- Verify API server allows CORS (enabled by default)

### Metrics Not Updating
- Check WebSocket connection in browser DevTools (Network tab)
- Verify device exists: `GET /api/devices/0`
- Check API logs for errors: `docker logs <container>`

---

## Scaling Considerations

### Single Server Limits
- **Concurrent WebSocket:** 50-100 connections
- **Devices:** 5000 (lazy initialized)
- **RPS:** 1000+ (REST)

### Horizontal Scaling
For 1000+ concurrent clients:

1. **Load Balancer:** nginx with WebSocket support
2. **Multiple API Servers:** 3-5 instances
3. **Shared State:** Redis for distributed metrics cache
4. **Cluster Sync:** Shared metrics collection backend

**Multi-server Setup:**
```
┌─ Load Balancer (nginx) ─┐
│                         │
├─ API Server 1 (8000)   │
├─ API Server 2 (8001)   │
├─ API Server 3 (8002)   │
│                         │
└─ Metrics Backend (Redis)┘
```

---

## Support & Documentation

- **API Docs:** http://localhost:8000/docs (auto-generated by FastAPI)
- **Redoc:** http://localhost:8000/redoc
- **Dashboard:** http://localhost:8000/dashboard.html
- **Tests:** `python test_api_client.py`

---

---
**Version:** 1.5.0  
**Last Updated:** 2024-01-15  
**Maintainer:** COBOL Hardware Architect Team
