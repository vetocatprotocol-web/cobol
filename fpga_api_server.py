"""
REST API & WebSocket Server for COBOL v1.5 FPGA Monitoring
Provides real-time metrics, health status, and cluster management
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import asdict
from datetime import datetime

from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import websockets

from fpga_controller import FPGAController, FPGACluster, HuffmanTable, FPGAMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# GLOBAL CLUSTER & CONTROL
# ============================================================================

# Initialize cluster (lazy load on first request)
_cluster: Optional[FPGACluster] = None
_active_ws_connections: List[WebSocket] = []


def get_cluster(num_devices: int = 5000) -> FPGACluster:
    """Get or create global cluster"""
    global _cluster
    if _cluster is None:
        logger.info(f"Initializing FPGA cluster with {num_devices} devices")
        _cluster = FPGACluster(num_devices=num_devices)
        # Pre-initialize devices 0-31 for demo
        for i in range(min(32, num_devices)):
            _cluster.initialize_device(i, use_simulator=True)
        _cluster.start_all_metrics()
    return _cluster


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="COBOL v1.5 FPGA Control API",
    description="Real-time monitoring and control for FPGA cluster",
    version="1.5.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# DEVICE ENDPOINTS
# ============================================================================

@app.get("/api/devices")
async def list_devices(limit: int = Query(32, ge=1, le=5000)) -> Dict:
    """List available FPGA devices"""
    cluster = get_cluster()
    devices = []
    
    for dev_id in range(min(limit, len(cluster.devices))):
        dev = cluster.get_device(dev_id)
        if dev:
            status = dev.get_status()
            devices.append({
                'device_id': dev_id,
                'state': status['state'],
                'health_score': status['health_score'],
                'cam_entries': status['cam_entries_loaded'],
                'huffman_tables': status['huffman_tables_loaded'],
            })
    
    return {
        'total_devices': cluster.num_devices,
        'initialized_devices': len(cluster.devices),
        'devices': devices
    }


@app.get("/api/devices/{device_id}")
async def get_device_status(device_id: int) -> Dict:
    """Get detailed status of specific device"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    status = dev.get_status()
    metrics = dev.get_metrics()
    
    return {
        'device_id': device_id,
        'state': status['state'],
        'health_score': status['health_score'],
        'issues': status['issues'],
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'input_rate_gb_s': metrics.input_rate_gb_s,
            'decomp_rate_gb_s': metrics.decomp_rate_gb_s,
            'output_rate_gb_s': metrics.output_rate_gb_s,
            'cam_hit_rate': metrics.cam_hit_rate,
            'compression_ratio': metrics.compression_ratio,
            'pipeline_depth': metrics.pipeline_depth,
            'active_clients': metrics.active_clients,
        },
        'resources': {
            'hbm_utilization_mb': metrics.hbm_utilization_mb,
            'bram_utilization_pct': metrics.bram_utilization_pct,
            'cam_entries': status['cam_entries_loaded'],
            'huffman_tables': status['huffman_tables_loaded'],
        }
    }


@app.post("/api/devices/{device_id}/cam/config")
async def configure_cam(device_id: int, request: Dict) -> Dict:
    """Configure CAM entry on device"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    pattern = bytes.fromhex(request.get('pattern', ''))
    match_id = request.get('match_id', 0)
    chunk_size = request.get('chunk_size', len(pattern))
    
    dev.configure_cam_entry(pattern, match_id, chunk_size)
    
    return {
        'status': 'configured',
        'device_id': device_id,
        'match_id': match_id,
        'pattern_len': len(pattern),
    }


@app.post("/api/devices/{device_id}/cam/flush")
async def flush_cam(device_id: int) -> Dict:
    """Flush CAM configuration to device"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    written = dev.flush_cam_config()
    
    return {
        'status': 'flushed',
        'device_id': device_id,
        'entries_written': written,
        'total_cam_entries': len(dev.cam_entries),
    }


@app.post("/api/devices/{device_id}/huffman/load")
async def load_huffman(device_id: int, request: Dict) -> Dict:
    """Load Huffman table for chunk"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    chunk_id = request.get('chunk_id', 0)
    table_data = request.get('table', {})
    
    table = HuffmanTable(
        chunk_id=chunk_id,
        code_length_bits=table_data.get('code_lengths', [3]*256),
        code_values=table_data.get('code_values', list(range(256))),
        symbols=table_data.get('symbols', list(range(256))),
        total_entries=256
    )
    
    result = dev.load_huffman_table(chunk_id, table)
    
    return {
        'status': 'loaded' if result else 'failed',
        'device_id': device_id,
        'chunk_id': chunk_id,
        'total_huffman_tables': len(dev.huffman_tables),
    }


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@app.get("/api/metrics/cluster")
async def get_cluster_metrics() -> Dict:
    """Get aggregate cluster metrics"""
    cluster = get_cluster()
    status = cluster.get_aggregate_status()
    
    return {
        'timestamp': datetime.now().isoformat(),
        'cluster_health_score': status['cluster_health_score'],
        'total_devices': status['total_devices'],
        'total_cam_entries': status['total_cam_entries'],
        'total_huffman_tables': status['total_huffman_tables'],
        'critical_issues': status['critical_issues'],
    }


@app.get("/api/metrics/device/{device_id}/history")
async def get_device_metrics_history(
    device_id: int,
    last_n: int = Query(100, ge=1, le=3600)
) -> Dict:
    """Get metrics history for device"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    history = dev.get_metrics_history(last_n)
    
    return {
        'device_id': device_id,
        'samples': len(history),
        'metrics': [
            {
                'timestamp': m.timestamp,
                'input_rate_gb_s': round(m.input_rate_gb_s, 2),
                'decomp_rate_gb_s': round(m.decomp_rate_gb_s, 0),
                'output_rate_gb_s': round(m.output_rate_gb_s, 2),
                'cam_hit_rate': round(m.cam_hit_rate, 1),
                'compression_ratio': round(m.compression_ratio, 0),
            }
            for m in history
        ]
    }


# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/metrics/{device_id}")
async def websocket_metrics(websocket: WebSocket, device_id: int):
    """WebSocket for real-time metrics streaming"""
    cluster = get_cluster()
    dev = cluster.get_device(device_id)
    
    if dev is None:
        await websocket.close(code=404, reason="Device not found")
        return
    
    await websocket.accept()
    _active_ws_connections.append(websocket)
    
    logger.info(f"WebSocket connected: device {device_id}")
    
    try:
        while True:
            # Send metrics every 500ms
            metrics = dev.get_metrics()
            message = {
                'type': 'metrics',
                'device_id': device_id,
                'timestamp': metrics.timestamp,
                'input_rate_gb_s': round(metrics.input_rate_gb_s, 2),
                'decomp_rate_gb_s': round(metrics.decomp_rate_gb_s, 0),
                'output_rate_gb_s': round(metrics.output_rate_gb_s, 2),
                'cam_hit_rate': round(metrics.cam_hit_rate, 1),
                'compression_ratio': round(metrics.compression_ratio, 0),
                'hbm_utilization_pct': round(100 * metrics.hbm_utilization_mb / 20000, 1),
                'bram_utilization_pct': round(metrics.bram_utilization_pct, 1),
                'pipeline_depth': metrics.pipeline_depth,
                'active_clients': metrics.active_clients,
            }
            
            await websocket.send_json(message)
            await asyncio.sleep(0.5)
    
    except Exception as e:
        logger.error(f"WebSocket error on device {device_id}: {e}")
    
    finally:
        _active_ws_connections.remove(websocket)
        logger.info(f"WebSocket disconnected: device {device_id}")


@app.websocket("/ws/cluster")
async def websocket_cluster_metrics(websocket: WebSocket):
    """WebSocket for cluster-level metrics"""
    cluster = get_cluster()
    
    await websocket.accept()
    _active_ws_connections.append(websocket)
    
    logger.info("WebSocket connected: cluster metrics")
    
    try:
        while True:
            # Send cluster metrics every 1 second
            status = cluster.get_aggregate_status()
            message = {
                'type': 'cluster_metrics',
                'timestamp': datetime.now().isoformat(),
                'total_devices': status['total_devices'],
                'healthy_devices': status['total_devices'] - len(status['critical_issues']),
                'cluster_health_score': round(status['cluster_health_score'], 1),
                'total_cam_entries': status['total_cam_entries'],
                'total_huffman_tables': status['total_huffman_tables'],
                'issues_count': len(status['critical_issues']),
            }
            
            await websocket.send_json(message)
            await asyncio.sleep(1.0)
    
    except Exception as e:
        logger.error(f"WebSocket error on cluster: {e}")
    
    finally:
        _active_ws_connections.remove(websocket)
        logger.info("WebSocket disconnected: cluster metrics")


# ============================================================================
# HEALTH & INFO ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check() -> Dict:
    """Health check endpoint"""
    try:
        cluster = get_cluster()
        status = cluster.get_aggregate_status()
        
        return {
            'status': 'healthy' if status['cluster_health_score'] > 70 else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'cluster_health': status['cluster_health_score'],
            'active_ws_connections': len(_active_ws_connections),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


@app.get("/api/info")
async def get_api_info() -> Dict:
    """Get API information"""
    return {
        'name': 'COBOL v1.5 FPGA Control API',
        'version': '1.5.0',
        'description': 'REST + WebSocket API for FPGA cluster management',
        'endpoints': {
            'devices': '/api/devices',
            'device_status': '/api/devices/{device_id}',
            'cam_config': 'POST /api/devices/{device_id}/cam/config',
            'cam_flush': 'POST /api/devices/{device_id}/cam/flush',
            'huffman_load': 'POST /api/devices/{device_id}/huffman/load',
            'cluster_metrics': '/api/metrics/cluster',
            'device_history': '/api/metrics/device/{device_id}/history',
            'ws_metrics': 'ws://host/ws/metrics/{device_id}',
            'ws_cluster': 'ws://host/ws/cluster',
        }
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error_type": type(exc).__name__}
    )


# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize cluster on startup"""
    logger.info("COBOL v1.5 FPGA Control API Starting...")
    cluster = get_cluster()
    logger.info(f"Cluster initialized: {cluster}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("COBOL v1.5 FPGA Control API Shutting down...")
    global _cluster
    if _cluster:
        _cluster.stop_all_metrics()
    logger.info("Cleanup complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "0.0.0.0"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    logger.info(f"Starting API server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
