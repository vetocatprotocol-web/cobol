"""
COBOL v1.5 Cluster Orchestration Framework
Manages deployment of 10 mobile container data centers with 5,000 FPGAs
Handles federation, replication, cost optimization, and geographic distribution
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentRegion(Enum):
    """Geographic regions for MCDC deployment"""
    NORTH_AMERICA = "NA"
    EUROPE = "EU"
    ASIA_PACIFIC = "APAC"
    MIDDLE_EAST = "ME"
    SOUTH_AMERICA = "SA"


class CostOptimizationStrategy(Enum):
    """Cost optimization strategies"""
    MINIMIZE_POWER = "power"
    MINIMIZE_LATENCY = "latency"
    MINIMIZE_TRANSFER = "transfer"
    BALANCED = "balanced"


@dataclass
class MCDCLocation:
    """Physical location of a Mobile Container Data Center"""
    location_id: str
    region: DeploymentRegion
    latitude: float
    longitude: float
    city: str
    country: str
    distance_to_nearest_km: float = 0.0


@dataclass
class NetworkLink:
    """Network connection between MCDCs or to cloud"""
    source_mcdc: str
    dest_mcdc: str
    bandwidth_mbps: float
    latency_ms: float
    reliability_pct: float = 99.9
    cost_per_gb: float = 0.01


@dataclass
class ReplicationStrategy:
    """Data replication policy across MCDCs"""
    min_replicas: int = 2
    max_replicas: int = 3
    geographic_diversity: bool = True
    sync_interval_hours: float = 1.0


class MCDCOrchestrator:
    """Master orchestrator for all MCDCs in the cluster"""
    
    def __init__(self, num_mcdc: int = 10, cost_strategy: CostOptimizationStrategy = CostOptimizationStrategy.BALANCED):
        self.num_mcdc = num_mcdc
        self.mcdc_list: Dict[str, 'MobileContainerDC'] = {}
        self.network_graph: Dict[str, List[NetworkLink]] = {}
        self.replication_policy = ReplicationStrategy()
        self.cost_strategy = cost_strategy
        self.total_compression_ratio = 500.0  # 500:1 compression
        
    def add_mcdc(self, mcdc_id: str, location: MCDCLocation, num_fpgas: int = 500):
        """Add a mobile container to the cluster"""
        mcdc = MobileContainerDC(
            mcdc_id=mcdc_id,
            location=location,
            num_fpgas=num_fpgas
        )
        self.mcdc_list[mcdc_id] = mcdc
        self.network_graph[mcdc_id] = []
        logger.info(f"Added MCDC {mcdc_id} at {location.city}, {location.country}")
        return mcdc
    
    def add_network_link(self, link: NetworkLink):
        """Establish network connection between MCDCs"""
        self.network_graph[link.source_mcdc].append(link)
        # Add reverse link (symmetric)
        reverse = NetworkLink(
            source_mcdc=link.dest_mcdc,
            dest_mcdc=link.source_mcdc,
            bandwidth_mbps=link.bandwidth_mbps,
            latency_ms=link.latency_ms,
            reliability_pct=link.reliability_pct,
            cost_per_gb=link.cost_per_gb
        )
        self.network_graph[link.dest_mcdc].append(reverse)
    
    def calculate_geo_distance(self, loc1: MCDCLocation, loc2: MCDCLocation) -> float:
        """Approximate geodesic distance in km using Haversine formula"""
        from math import radians, cos, sin, asin, sqrt
        
        lon1, lat1, lon2, lat2 = map(radians, [loc1.longitude, loc1.latitude, 
                                                 loc2.longitude, loc2.latitude])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in km
        
        return c * r
    
    def optimize_placement(self, data_location: str, data_size_tb: float) -> Dict:
        """Recommend optimal MCDC placement for data based on cost strategy"""
        if data_location not in self.mcdc_list:
            return {'error': f'Location {data_location} not found'}
        
        costs = {}
        for mcdc_id, mcdc in self.mcdc_list.items():
            # Cost components
            power_cost = mcdc.get_power_cost()
            transfer_cost = self._estimate_transfer_cost(data_location, mcdc_id, data_size_tb)
            latency = self._estimate_latency(data_location, mcdc_id)
            
            # Composite score based on strategy
            if self.cost_strategy == CostOptimizationStrategy.MINIMIZE_POWER:
                score = power_cost
            elif self.cost_strategy == CostOptimizationStrategy.MINIMIZE_TRANSFER:
                score = transfer_cost
            elif self.cost_strategy == CostOptimizationStrategy.MINIMIZE_LATENCY:
                score = latency
            else:  # BALANCED
                score = (power_cost * 0.3) + (transfer_cost * 0.4) + (latency * 0.3)
            
            costs[mcdc_id] = {
                'power_cost': power_cost,
                'transfer_cost': transfer_cost,
                'latency_ms': latency,
                'composite_score': score
            }
        
        # Return top 3 recommendations
        sorted_costs = sorted(costs.items(), key=lambda x: x[1]['composite_score'])
        return {
            'strategy': self.cost_strategy.value,
            'recommendations': [
                {'mcdc': mcdc_id, **metrics} 
                for mcdc_id, metrics in sorted_costs[:3]
            ]
        }
    
    def _estimate_transfer_cost(self, from_mcdc: str, to_mcdc: str, data_size_tb: float) -> float:
        """Estimate data transfer cost between MCDCs"""
        if from_mcdc == to_mcdc:
            return 0.0
        
        # Find cheapest path using Dijkstra-like logic
        links = self.network_graph.get(from_mcdc, [])
        min_cost = float('inf')
        
        for link in links:
            if link.dest_mcdc == to_mcdc:
                cost = link.cost_per_gb * data_size_tb * 1024  # TB to GB
                min_cost = min(min_cost, cost)
        
        return min_cost if min_cost != float('inf') else 1000.0  # Default high cost
    
    def _estimate_latency(self, from_mcdc: str, to_mcdc: str) -> float:
        """Estimate network latency between MCDCs"""
        if from_mcdc == to_mcdc:
            return 1.0  # Local latency
        
        links = self.network_graph.get(from_mcdc, [])
        for link in links:
            if link.dest_mcdc == to_mcdc:
                return link.latency_ms
        
        return 500.0  # Default high latency
    
    def get_cluster_status(self) -> Dict:
        """Get overall cluster health and capacity"""
        total_fpga = sum(mcdc.num_fpgas for mcdc in self.mcdc_list.values())
        total_power = sum(mcdc.get_power_cost() for mcdc in self.mcdc_list.values())
        total_capacity_eb = (total_fpga * 6.0) / 1000  # 6 EB per FPGA
        
        return {
            'num_mcdc': len(self.mcdc_list),
            'total_fpga': total_fpga,
            'total_power_kw': total_power,
            'total_capacity_eb': total_capacity_eb,
            'compression_ratio': self.total_compression_ratio,
            'effective_capacity_eb': total_capacity_eb / self.total_compression_ratio,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_replication_distribution(self, data_hash: str, data_size_tb: float) -> Dict:
        """Calculate optimal replication distribution"""
        # Deterministic MCDC selection based on hash
        mcdc_ids = list(self.mcdc_list.keys())
        primary_idx = int(data_hash[:8], 16) % len(mcdc_ids)
        
        replication_plan = {
            'data_hash': data_hash,
            'primary_mcdc': mcdc_ids[primary_idx],
            'replicas': [],
            'total_replicas': self.replication_policy.min_replicas
        }
        
        # Select replicas in different regions if possible
        selected_regions = set()
        primary_region = self.mcdc_list[mcdc_ids[primary_idx]].location.region
        selected_regions.add(primary_region)
        
        for i in range(1, self.replication_policy.min_replicas):
            # Find next MCDC in different region
            for mcdc_id in mcdc_ids:
                if mcdc_id != mcdc_ids[primary_idx]:
                    region = self.mcdc_list[mcdc_id].location.region
                    if region not in selected_regions:
                        replication_plan['replicas'].append(mcdc_id)
                        selected_regions.add(region)
                        break
            else:
                # Fallback if not enough regions
                fallback_idx = (primary_idx + i) % len(mcdc_ids)
                if mcdc_ids[fallback_idx] != mcdc_ids[primary_idx]:
                    replication_plan['replicas'].append(mcdc_ids[fallback_idx])
        
        return replication_plan


@dataclass
class MobileContainerDC:
    """Individual Mobile Container Data Center"""
    mcdc_id: str
    location: MCDCLocation
    num_fpgas: int = 500
    power_consumption_kw: float = 400.0
    cooling_capacity_kw: float = 420.0
    uptime_sla_pct: float = 99.95
    
    def get_power_cost(self) -> float:
        """Estimated annual power cost (~$0.15/kWh, 8760 hours/year)"""
        return self.power_consumption_kw * 0.15 * 8760 / 1000000  # In millions
    
    def get_summary(self) -> Dict:
        return {
            'mcdc_id': self.mcdc_id,
            'location': {
                'city': self.location.city,
                'country': self.location.country,
                'region': self.location.region.value
            },
            'num_fpgas': self.num_fpgas,
            'power_kw': self.power_consumption_kw,
            'cooling_kw': self.cooling_capacity_kw,
            'uptime_sla': f"{self.uptime_sla_pct}%",
            'annual_power_cost_m': f"${self.get_power_cost():.3f}M"
        }


class FederationProtocol:
    """Inter-MCDC federation and synchronization protocol"""
    
    def __init__(self, orchestrator: MCDCOrchestrator):
        self.orchestrator = orchestrator
        self.sync_queue: List[Dict] = []
        self.ledger: List[Dict] = []
    
    def broadcast_dictionary(self, dictionary_hash: str, mcdc_origin: str) -> Dict:
        """Broadcast dictionary update to all MCDCs (gossip protocol)"""
        message = {
            'type': 'dictionary_update',
            'hash': dictionary_hash,
            'origin': mcdc_origin,
            'timestamp': datetime.now().isoformat(),
            'ttl': 10  # hops to live
        }
        
        # Simulate gossip propagation
        propagated = [mcdc_origin]
        for mcdc_id in self.orchestrator.mcdc_list.keys():
            if mcdc_id != mcdc_origin:
                propagated.append(mcdc_id)
        
        self.ledger.append({
            'event': 'dictionary_broadcast',
            'message': message,
            'propagated_to': propagated
        })
        
        return {'broadcast': dictionary_hash, 'reached': propagated}
    
    def sync_metrics(self) -> Dict:
        """Synchronize performance metrics across cluster"""
        metrics_sync = {}
        for mcdc_id, mcdc in self.orchestrator.mcdc_list.items():
            metrics_sync[mcdc_id] = {
                'power_usage': mcdc.power_consumption_kw,
                'fpga_count': mcdc.num_fpgas,
                'uptime': mcdc.uptime_sla_pct
            }
        
        self.ledger.append({
            'event': 'metrics_sync',
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_sync
        })
        
        return metrics_sync


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize orchestrator
    orch = MCDCOrchestrator(
        num_mcdc=10,
        cost_strategy=CostOptimizationStrategy.BALANCED
    )
    
    # Define MCDC locations
    locations = [
        MCDCLocation("MCDC-1", DeploymentRegion.NORTH_AMERICA, 40.7128, -74.0060, "New York", "USA"),
        MCDCLocation("MCDC-2", DeploymentRegion.EUROPE, 51.5074, -0.1278, "London", "UK"),
        MCDCLocation("MCDC-3", DeploymentRegion.ASIA_PACIFIC, 35.6762, 139.6503, "Tokyo", "Japan"),
        MCDCLocation("MCDC-4", DeploymentRegion.ASIA_PACIFIC, 1.3521, 103.8198, "Singapore", "Singapore"),
        MCDCLocation("MCDC-5", DeploymentRegion.EUROPE, 48.8566, 2.3522, "Paris", "France"),
        MCDCLocation("MCDC-6", DeploymentRegion.MIDDLE_EAST, 25.2048, 55.2708, "Dubai", "UAE"),
        MCDCLocation("MCDC-7", DeploymentRegion.ASIA_PACIFIC, -33.8688, 151.2093, "Sydney", "Australia"),
        MCDCLocation("MCDC-8", DeploymentRegion.NORTH_AMERICA, 37.7749, -122.4194, "San Francisco", "USA"),
        MCDCLocation("MCDC-9", DeploymentRegion.SOUTH_AMERICA, -23.5505, -46.6333, "São Paulo", "Brazil"),
        MCDCLocation("MCDC-10", DeploymentRegion.ASIA_PACIFIC, 28.7041, 77.1025, "New Delhi", "India"),
    ]
    
    # Add MCDCs
    for loc in locations:
        orch.add_mcdc(f"mcdc_{loc.location_id}", loc, num_fpgas=500)
    
    # Add network links (sample)
    orch.add_network_link(NetworkLink(
        source_mcdc="mcdc_MCDC-1", 
        dest_mcdc="mcdc_MCDC-8",
        bandwidth_mbps=100000,
        latency_ms=10,
        cost_per_gb=0.001
    ))
    
    # Print cluster status
    print("\n=== COBOL v1.5 Cluster Orchestration ===\n")
    print(json.dumps(orch.get_cluster_status(), indent=2))
    
    # Print MCDC locations
    print("\n=== Mobile Container Locations ===\n")
    for mcdc_id, mcdc in orch.mcdc_list.items():
        print(json.dumps(mcdc.get_summary(), indent=2))
    
    # Placement optimization
    print("\n=== Cost Optimization Example ===\n")
    placement = orch.optimize_placement("mcdc_MCDC-1", 15.0)
    print(json.dumps(placement, indent=2))
    
    # Federation test
    fed = FederationProtocol(orch)
    print("\n=== Federation Broadcast ===\n")
    result = fed.broadcast_dictionary("abc123def456", "mcdc_MCDC-1")
    print(json.dumps(result, indent=2))
