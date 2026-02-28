"""
REST API Client Tester for COBOL v1.5 FPGA Control
Tests all API endpoints and WebSocket connections
"""

import asyncio
import json
import time
import requests
import websockets
from typing import Dict, List

# Configuration
API_BASE = "http://localhost:8000/api"
WS_BASE = "ws://localhost:8000"


class FPGAAPIClient:
    """Client for FPGA Control API"""
    
    def __init__(self, base_url: str = API_BASE, ws_base: str = WS_BASE):
        self.base_url = base_url
        self.ws_base = ws_base
        self.session = requests.Session()
    
    # ========================================================================
    # DEVICE ENDPOINTS
    # ========================================================================
    
    def list_devices(self, limit: int = 32) -> Dict:
        """List available FPGA devices"""
        response = self.session.get(f"{self.base_url}/devices", params={"limit": limit})
        response.raise_for_status()
        return response.json()
    
    def get_device_status(self, device_id: int) -> Dict:
        """Get detailed device status"""
        response = self.session.get(f"{self.base_url}/devices/{device_id}")
        response.raise_for_status()
        return response.json()
    
    def configure_cam(self, device_id: int, pattern: bytes, match_id: int = 0) -> Dict:
        """Configure CAM entry"""
        data = {
            'pattern': pattern.hex(),
            'match_id': match_id,
            'chunk_size': len(pattern)
        }
        response = self.session.post(
            f"{self.base_url}/devices/{device_id}/cam/config",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def flush_cam(self, device_id: int) -> Dict:
        """Flush CAM configuration to device"""
        response = self.session.post(f"{self.base_url}/devices/{device_id}/cam/flush")
        response.raise_for_status()
        return response.json()
    
    def load_huffman(self, device_id: int, chunk_id: int, table_dict: Dict = None) -> Dict:
        """Load Huffman table"""
        if table_dict is None:
            table_dict = {
                'code_lengths': [3] * 256,
                'code_values': list(range(256)),
                'symbols': list(range(256))
            }
        
        data = {
            'chunk_id': chunk_id,
            'table': table_dict
        }
        response = self.session.post(
            f"{self.base_url}/devices/{device_id}/huffman/load",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # METRICS ENDPOINTS
    # ========================================================================
    
    def get_cluster_metrics(self) -> Dict:
        """Get cluster-wide metrics"""
        response = self.session.get(f"{self.base_url}/metrics/cluster")
        response.raise_for_status()
        return response.json()
    
    def get_device_history(self, device_id: int, last_n: int = 100) -> Dict:
        """Get device metrics history"""
        response = self.session.get(
            f"{self.base_url}/metrics/device/{device_id}/history",
            params={"last_n": last_n}
        )
        response.raise_for_status()
        return response.json()
    
    # ========================================================================
    # HEALTH & INFO
    # ========================================================================
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url.replace('/api', '')}/health")
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict:
        """Get API information"""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


class WebSocketTester:
    """Test WebSocket connections"""
    
    @staticmethod
    async def stream_device_metrics(device_id: int, duration: float = 5.0):
        """Stream metrics from device via WebSocket"""
        url = f"{WS_BASE}/ws/metrics/{device_id}"
        samples = []
        
        try:
            async with websockets.connect(url) as ws:
                start = time.time()
                while time.time() - start < duration:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(message)
                        samples.append(data)
                        print(f"  Sample {len(samples)}: {data['input_rate_gb_s']:.1f} GB/s input")
                    except asyncio.TimeoutError:
                        break
        
        except Exception as e:
            print(f"  WebSocket error: {e}")
        
        return samples
    
    @staticmethod
    async def stream_cluster_metrics(duration: float = 3.0):
        """Stream cluster metrics via WebSocket"""
        url = f"{WS_BASE}/ws/cluster"
        samples = []
        
        try:
            async with websockets.connect(url) as ws:
                start = time.time()
                while time.time() - start < duration:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=2.0)
                        data = json.loads(message)
                        samples.append(data)
                        print(f"  Sample {len(samples)}: Health {data['cluster_health_score']:.0f}/100")
                    except asyncio.TimeoutError:
                        break
        
        except Exception as e:
            print(f"  WebSocket error: {e}")
        
        return samples


# ============================================================================
# TEST SUITE
# ============================================================================

def test_api_info():
    """Test API info endpoint"""
    print("\n[TEST] API Information")
    client = FPGAAPIClient()
    try:
        info = client.get_info()
        print(f"  ✓ API Name: {info['name']}")
        print(f"  ✓ Version: {info['version']}")
        print(f"  ✓ Endpoints: {len(info['endpoints'])} available")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_health_check():
    """Test health endpoint"""
    print("\n[TEST] Health Check")
    client = FPGAAPIClient()
    try:
        health = client.health_check()
        print(f"  ✓ Status: {health['status']}")
        print(f"  ✓ Cluster Health: {health['cluster_health']:.0f}/100")
        print(f"  ✓ Active WS Connections: {health['active_ws_connections']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_list_devices():
    """Test device listing"""
    print("\n[TEST] List Devices")
    client = FPGAAPIClient()
    try:
        result = client.list_devices(limit=8)
        print(f"  ✓ Total Devices: {result['total_devices']}")
        print(f"  ✓ Initialized: {result['initialized_devices']}")
        for dev in result['devices'][:3]:
            print(f"    - Device {dev['device_id']}: {dev['state']} (Health: {dev['health_score']})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_device_status():
    """Test individual device status"""
    print("\n[TEST] Device Status")
    client = FPGAAPIClient()
    try:
        status = client.get_device_status(0)
        print(f"  ✓ Device 0: {status['state']}")
        print(f"    - Health Score: {status['health_score']}")
        print(f"    - Input Rate: {status['metrics']['input_rate_gb_s']:.1f} GB/s")
        print(f"    - Hit Rate: {status['metrics']['cam_hit_rate']:.1f}%")
        print(f"    - HBM Usage: {status['resources']['hbm_utilization_mb']} MB")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_cam_workflow():
    """Test CAM configuration workflow"""
    print("\n[TEST] CAM Configuration Workflow")
    client = FPGAAPIClient()
    try:
        # Configure pattern
        print("  Step 1: Configure CAM entry...")
        pattern = b"test_pattern_123"
        result = client.configure_cam(0, pattern, match_id=42)
        print(f"  ✓ Configured: {result['pattern_len']} bytes, Match ID: {result['match_id']}")
        
        # Flush to device
        print("  Step 2: Flush configuration...")
        flush_result = client.flush_cam(0)
        print(f"  ✓ Flushed: {flush_result['entries_written']} entries")
        
        # Verify
        status = client.get_device_status(0)
        print(f"  ✓ Verification: {status['resources']['cam_entries']} CAM entries loaded")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_huffman_workflow():
    """Test Huffman table loading"""
    print("\n[TEST] Huffman Table Workflow")
    client = FPGAAPIClient()
    try:
        for chunk in range(3):
            result = client.load_huffman(0, chunk_id=chunk)
            print(f"  ✓ Chunk {chunk}: Loaded (Total: {result['total_huffman_tables']})")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_cluster_metrics():
    """Test cluster metrics"""
    print("\n[TEST] Cluster Metrics")
    client = FPGAAPIClient()
    try:
        metrics = client.get_cluster_metrics()
        print(f"  ✓ Cluster Health: {metrics['cluster_health_score']:.0f}/100")
        print(f"  ✓ Total Devices: {metrics['total_devices']}")
        print(f"  ✓ CAM Entries: {metrics['total_cam_entries']}")
        print(f"  ✓ Huffman Tables: {metrics['total_huffman_tables']}")
        print(f"  ✓ Critical Issues: {metrics['critical_issues']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_containers():
    """Test container listing endpoint"""
    print("\n[TEST] Container Listing")
    client = FPGAAPIClient()
    try:
        result = client.session.get(f"{client.base_url}/containers").json()
        print(f"  ✓ Total Containers: {result.get('total_containers')}")
        assert 'containers' in result
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_economics():
    """Test economics endpoints"""
    print("\n[TEST] Economic Calculations")
    client = FPGAAPIClient()
    try:
        tco = client.session.get(f"{client.base_url}/economics/tco?eb=15").json()
        print(f"  ✓ Cloud cost/yr: {tco['cloud_cost_per_year']}")
        be = client.session.get(f"{client.base_url}/economics/break_even?eb=15").json()
        print(f"  ✓ Break-even year: {be['break_even_year']}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


def test_device_history():
    """Test metrics history"""
    print("\n[TEST] Device Metrics History")
    client = FPGAAPIClient()
    try:
        history = client.get_device_history(0, last_n=20)
        print(f"  ✓ Retrieved {history['samples']} samples")
        if history['metrics']:
            latest = history['metrics'][-1]
            print(f"    - Latest input rate: {latest['input_rate_gb_s']:.1f} GB/s")
            print(f"    - Latest hit rate: {latest['cam_hit_rate']:.1f}%")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


async def test_websocket_device():
    """Test device metrics WebSocket"""
    print("\n[TEST] WebSocket Device Metrics")
    try:
        samples = await WebSocketTester.stream_device_metrics(0, duration=3.0)
        print(f"  ✓ Received {len(samples)} samples")
        if samples:
            avg_rate = sum(s['input_rate_gb_s'] for s in samples) / len(samples)
            print(f"  ✓ Average input rate: {avg_rate:.1f} GB/s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


async def test_websocket_cluster():
    """Test cluster metrics WebSocket"""
    print("\n[TEST] WebSocket Cluster Metrics")
    try:
        samples = await WebSocketTester.stream_cluster_metrics(duration=2.0)
        print(f"  ✓ Received {len(samples)} samples")
        if samples:
            avg_health = sum(s['cluster_health_score'] for s in samples) / len(samples)
            print(f"  ✓ Average health score: {avg_health:.0f}/100")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


# ============================================================================
# STRESS TEST
# ============================================================================

def stress_test_api():
    """Stress test API endpoints"""
    print("\n[STRESS TEST] API Load Test")
    client = FPGAAPIClient()
    
    iterations = 50
    start = time.time()
    
    try:
        for i in range(iterations):
            client.get_cluster_metrics()
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{iterations} requests")
        
        duration = time.time() - start
        rps = iterations / duration
        print(f"  ✓ Completed {iterations} requests in {duration:.2f}s ({rps:.0f} RPS)")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


async def stress_test_websocket():
    """Stress test WebSocket connection"""
    print("\n[STRESS TEST] WebSocket Load Test")
    
    async def connect_device(device_id: int, count: int):
        try:
            for _ in range(count):
                await WebSocketTester.stream_device_metrics(device_id, duration=1.0)
        except Exception as e:
            print(f"  Error on device {device_id}: {e}")
    
    try:
        start = time.time()
        # Connect 10 devices concurrently for 1 second each
        await asyncio.gather(*[
            connect_device(i, 1) for i in range(10)
        ])
        duration = time.time() - start
        print(f"  ✓ 10 concurrent WebSocket connections completed in {duration:.2f}s")
    except Exception as e:
        print(f"  ✗ Failed: {e}")


# ============================================================================
# MAIN
# ============================================================================

async def run_async_tests():
    """Run all async tests"""
    await test_websocket_device()
    await test_websocket_cluster()
    await stress_test_websocket()


def main():
    """Run full test suite"""
    print("=" * 70)
    print("COBOL v1.5 FPGA API Test Suite")
    print("=" * 70)
    
    # Check connectivity
    print("\n[STARTUP] Checking API connectivity...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        print(f"✓ API is running on http://localhost:8000")
    except:
        print("✗ API is not running. Start with: python fpga_api_server.py")
        return
    
    # Sync tests
    print("\n" + "=" * 70)
    print("SYNCHRONOUS TESTS")
    print("=" * 70)
    
    test_api_info()
    test_health_check()
    test_list_devices()
    test_device_status()
    test_cam_workflow()
    test_huffman_workflow()
    test_cluster_metrics()
    test_device_history()
    
    # Async tests
    print("\n" + "=" * 70)
    print("ASYNCHRONOUS TESTS")
    print("=" * 70)
    
    asyncio.run(run_async_tests())
    
    # Stress tests
    print("\n" + "=" * 70)
    print("STRESS TESTS")
    print("=" * 70)
    
    stress_test_api()
    asyncio.run(stress_test_websocket())
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Open dashboard: http://localhost:8000/dashboard.html")
    print("  2. Monitor real-time metrics via WebSocket")
    print("  3. Test device control endpoints")
    print("\nFor more info: python fpga_api_server.py --help")


if __name__ == "__main__":
    main()
