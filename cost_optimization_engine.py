"""
COBOL v1.5 Cost Optimization & Economics Engine
Detailed TCO analysis, ROI calculation, and cost-benefit trade-off analysis
"""

import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class CloudProvider(Enum):
    """Cloud storage providers"""
    GOOGLE_CLOUD = "google_cloud"
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    CUSTOM = "custom"


class FPGABoardModel(Enum):
    """FPGA board types and costs"""
    XILINX_U50 = ("Xilinx U50", 12000)  # (name, cost per unit)
    XILINX_U55C = ("Xilinx U55C", 14000)
    XILINX_U200 = ("Xilinx U200", 15000)
    XILINX_U280 = ("Xilinx U280", 18000)


@dataclass
class StorageCost:
    """Cloud storage pricing model"""
    provider: CloudProvider
    hot_tier_per_gb_month: float
    warm_tier_per_gb_month: float
    cold_tier_per_gb_month: float
    egress_per_gb: float
    ingest_per_gb: float = 0.0  # Often free


class ComprehensiveEconomicModel:
    """Detailed cost estimation for FPGA vs Cloud comparison"""
    
    # Default cloud pricing (USD)
    CLOUD_PRICING = {
        CloudProvider.GOOGLE_CLOUD: StorageCost(
            provider=CloudProvider.GOOGLE_CLOUD,
            hot_tier_per_gb_month=0.020,
            warm_tier_per_gb_month=0.010,
            cold_tier_per_gb_month=0.004,
            egress_per_gb=0.12,
            ingest_per_gb=0.0
        ),
        CloudProvider.AWS_S3: StorageCost(
            provider=CloudProvider.AWS_S3,
            hot_tier_per_gb_month=0.023,
            warm_tier_per_gb_month=0.0125,
            cold_tier_per_gb_month=0.004,
            egress_per_gb=0.09,
            ingest_per_gb=0.0
        ),
        CloudProvider.AZURE_BLOB: StorageCost(
            provider=CloudProvider.AZURE_BLOB,
            hot_tier_per_gb_month=0.021,
            warm_tier_per_gb_month=0.012,
            cold_tier_per_gb_month=0.005,
            egress_per_gb=0.087,
            ingest_per_gb=0.0
        ),
    }
    
    def __init__(
        self,
        num_fpga: int = 5000,
        fpga_board_model: FPGABoardModel = FPGABoardModel.XILINX_U50,
        num_containers: int = 10,
        lifespan_years: int = 5,
        annual_electricity_cost_per_kw: float = 1314,  # ~$0.15/kWh
        power_per_fpga_kw: float = 0.08,
        maintenance_per_year_pct: float = 0.05  # 5% of CAPEX
    ):
        self.num_fpga = num_fpga
        self.fpga_board_model = fpga_board_model
        self.num_containers = num_containers
        self.lifespan_years = lifespan_years
        self.annual_electricity_cost_per_kw = annual_electricity_cost_per_kw
        self.power_per_fpga_kw = power_per_fpga_kw
        self.maintenance_pct = maintenance_per_year_pct
        
    def calculate_fpga_capex(self) -> Dict[str, float]:
        """Calculate capital expenditure for FPGA infrastructure"""
        board_cost = self.fpga_board_model.value[1]
        
        # Hardware costs
        fpga_boards_cost = self.num_fpga * board_cost
        
        # Container infrastructure (enclosure, power, cooling)
        container_infra_per_unit = 100000  # $100k per container
        container_cost = self.num_containers * container_infra_per_unit
        
        # Network infrastructure
        network_cost = self.num_fpga * 50  # $50 per FPGA for networking
        
        # Deployment & installation
        deployment_cost = self.num_containers * 50000  # $50k per container
        
        total_capex = fpga_boards_cost + container_cost + network_cost + deployment_cost
        
        return {
            'fpga_boards': fpga_boards_cost,
            'container_infrastructure': container_cost,
            'network': network_cost,
            'deployment': deployment_cost,
            'total_capex': total_capex
        }
    
    def calculate_fpga_opex(self, years: int = None) -> Dict[str, float]:
        """Calculate operational expenditure for FPGA infrastructure"""
        if years is None:
            years = self.lifespan_years
        
        # Power consumption
        total_power_kw = self.num_fpga * self.power_per_fpga_kw
        annual_power_cost = total_power_kw * self.annual_electricity_cost_per_kw
        total_power_cost = annual_power_cost * years
        
        # Maintenance & support
        capex = self.calculate_fpga_capex()['total_capex']
        annual_maintenance = capex * self.maintenance_pct
        total_maintenance = annual_maintenance * years
        
        # Cooling system (liquid cooling maintenance)
        annual_cooling = total_power_kw * 10  # $10 per kW per year
        total_cooling = annual_cooling * years
        
        # Personnel (2 engineers @ $150k/year for 5000 FPGAs)
        annual_personnel = 300000  # 2 engineers
        total_personnel = annual_personnel * years
        
        total_opex = total_power_cost + total_maintenance + total_cooling + total_personnel
        
        return {
            'annual_power': annual_power_cost,
            'total_power_over_life': total_power_cost,
            'annual_maintenance': annual_maintenance,
            'total_maintenance_over_life': total_maintenance,
            'annual_cooling': annual_cooling,
            'total_cooling_over_life': total_cooling,
            'annual_personnel': annual_personnel,
            'total_personnel_over_life': total_personnel,
            'total_annual_opex': annual_power_cost + annual_maintenance + annual_cooling + annual_personnel,
            'total_opex_over_life': total_opex
        }
    
    def calculate_cloud_costs(
        self,
        data_size_eb: float,
        years: int = None,
        provider: CloudProvider = CloudProvider.GOOGLE_CLOUD,
        access_pattern: str = "hot"
    ) -> Dict[str, float]:
        """Calculate cloud storage costs"""
        if years is None:
            years = self.lifespan_years
        
        pricing = self.CLOUD_PRICING[provider]
        data_size_gb = data_size_eb * 1_000_000_000
        
        # Select pricing tier
        if access_pattern == "hot":
            monthly_per_gb = pricing.hot_tier_per_gb_month
        elif access_pattern == "warm":
            monthly_per_gb = pricing.warm_tier_per_gb_month
        else:  # cold
            monthly_per_gb = pricing.cold_tier_per_gb_month
        
        # Storage costs
        monthly_storage = data_size_gb * monthly_per_gb
        annual_storage = monthly_storage * 12
        total_storage = annual_storage * years
        
        # Egress costs (assume 10% of data egressed per year for archival)
        egress_per_year = data_size_gb * pricing.egress_per_gb * 0.1
        total_egress = egress_per_year * years
        
        total_cloud = total_storage + total_egress
        
        return {
            'provider': provider.value,
            'access_pattern': access_pattern,
            'monthly_storage': monthly_storage,
            'annual_storage': annual_storage,
            'total_storage_over_life': total_storage,
            'annual_egress': egress_per_year,
            'total_egress_over_life': total_egress,
            'total_cloud_cost_over_life': total_cloud
        }
    
    def comparative_analysis(
        self,
        data_size_eb: float,
        cloud_provider: CloudProvider = CloudProvider.GOOGLE_CLOUD,
        access_pattern: str = "hot"
    ) -> Dict:
        """Comprehensive comparative analysis"""
        fpga_capex = self.calculate_fpga_capex()
        fpga_opex = self.calculate_fpga_opex()
        cloud_costs = self.calculate_cloud_costs(
            data_size_eb,
            provider=cloud_provider,
            access_pattern=access_pattern
        )
        
        total_fpga = fpga_capex['total_capex'] + fpga_opex['total_opex_over_life']
        total_cloud = cloud_costs['total_cloud_cost_over_life']
        
        savings = total_cloud - total_fpga
        roi_months = (fpga_capex['total_capex'] / (cloud_costs['annual_storage'] - fpga_opex['total_annual_opex'] / 12)) if (cloud_costs['annual_storage'] > fpga_opex['total_annual_opex'] / 12) else float('inf')
        
        return {
            'summary': {
                'data_size_eb': data_size_eb,
                'analysis_period_years': self.lifespan_years,
                'timestamp': '2026-02-28'
            },
            'fpga_infrastructure': {
                'capex': fpga_capex['total_capex'],
                'opex_over_life': fpga_opex['total_opex_over_life'],
                'total_cost': total_fpga,
                'annual_opex': fpga_opex['total_annual_opex']
            },
            'cloud_storage': {
                'provider': cloud_provider.value,
                'access_pattern': access_pattern,
                'total_cost': total_cloud,
                'annual_cost': cloud_costs['annual_storage'] + cloud_costs['annual_egress']
            },
            'economics': {
                'fpga_cheaper_by': max(0, savings),
                'cloud_cheaper_by': max(0, -savings),
                'savings_percent': (savings / total_cloud * 100) if total_cloud > 0 else 0,
                'payback_period_months': roi_months if roi_months != float('inf') else 999,
                'annual_savings': max(0, (cloud_costs['annual_storage'] + cloud_costs['annual_egress']) - fpga_opex['total_annual_opex'])
            },
            'recommendation': self._make_recommendation(savings, roi_months)
        }
    
    def _make_recommendation(self, savings: float, roi_months: float) -> str:
        """Make recommendation based on economics"""
        if savings > 0 and roi_months < 48:
            return "BUILD FPGA INFRASTRUCTURE: Strong economic case with <4 year ROI"
        elif savings > 0 and roi_months < 60:
            return "BUILD FPGA INFRASTRUCTURE: Positive ROI within 5 years"
        elif savings > 0:
            return "HYBRID APPROACH: FPGA infrastructure economically viable for long-term"
        else:
            return "USE CLOUD: Current cloud solution more cost-effective"
    
    def sensitivity_analysis(
        self,
        data_size_eb: float,
        param_name: str,
        param_range: List[float]
    ) -> List[Dict]:
        """Sensitivity analysis on key parameters"""
        results = []
        
        for param_value in param_range:
            # Temporarily set parameter
            if param_name == "power_per_fpga_kw":
                original = self.power_per_fpga_kw
                self.power_per_fpga_kw = param_value
            elif param_name == "fpga_lifespan":
                original = self.lifespan_years
                self.lifespan_years = int(param_value)
            elif param_name == "electricity_cost":
                original = self.annual_electricity_cost_per_kw
                self.annual_electricity_cost_per_kw = param_value
            
            analysis = self.comparative_analysis(data_size_eb)
            results.append({
                'parameter': param_name,
                'parameter_value': param_value,
                'fpga_total_cost': analysis['fpga_infrastructure']['total_cost'],
                'cloud_total_cost': analysis['cloud_storage']['total_cost'],
                'savings': analysis['economics']['fpga_cheaper_by']
            })
            
            # Restore original
            if param_name == "power_per_fpga_kw":
                self.power_per_fpga_kw = original
            elif param_name == "fpga_lifespan":
                self.lifespan_years = original
            elif param_name == "electricity_cost":
                self.annual_electricity_cost_per_kw = original
        
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n=== COBOL v1.5 Comprehensive Cost Analysis ===\n")
    
    model = ComprehensiveEconomicModel(
        num_fpga=5000,
        fpga_board_model=FPGABoardModel.XILINX_U50,
        num_containers=10,
        lifespan_years=5
    )
    
    # Comparative analysis for 15 EB
    analysis = model.comparative_analysis(
        data_size_eb=15.0,
        cloud_provider=CloudProvider.GOOGLE_CLOUD,
        access_pattern="hot"
    )
    
    print(json.dumps(analysis, indent=2))
    
    # Sensitivity analysis
    print("\n=== Sensitivity Analysis: Power Cost ===\n")
    sensitivity = model.sensitivity_analysis(
        data_size_eb=15.0,
        param_name="power_per_fpga_kw",
        param_range=[0.05, 0.08, 0.10, 0.12, 0.15]
    )
    
    for result in sensitivity:
        print(f"Power: {result['parameter_value']} kW/FPGA | "
              f"FPGA Cost: ${result['fpga_total_cost']/1e6:.1f}M | "
              f"Cloud Cost: ${result['cloud_total_cost']/1e6:.1f}M | "
              f"Savings: ${result['savings']/1e6:.1f}M")
