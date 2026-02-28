"""
Test suite for cluster orchestrator and cost optimization
"""

import pytest
import json
from cluster_orchestrator import (
    MCDCOrchestrator, MCDCLocation, NetworkLink,
    DeploymentRegion, CostOptimizationStrategy,
    FederationProtocol, MobileContainerDC
)
from cost_optimization_engine import (
    ComprehensiveEconomicModel, CloudProvider, FPGABoardModel
)


class TestMCDCOrchestrator:
    """Test cluster orchestrator functionality"""
    
    @pytest.fixture
    def orchestrator(self):
        return MCDCOrchestrator(num_mcdc=10)
    
    def test_add_mcdc(self, orchestrator):
        """Test adding MCDC to cluster"""
        loc = MCDCLocation(
            location_id="TEST-1",
            region=DeploymentRegion.NORTH_AMERICA,
            latitude=40.7128,
            longitude=-74.0060,
            city="New York",
            country="USA"
        )
        mcdc = orchestrator.add_mcdc("test_mcdc", loc, num_fpgas=500)
        assert "test_mcdc" in orchestrator.mcdc_list
        assert mcdc.num_fpgas == 500
    
    def test_cluster_status(self, orchestrator):
        """Test cluster status report"""
        loc = MCDCLocation(
            location_id="TEST-2",
            region=DeploymentRegion.EUROPE,
            latitude=51.5074,
            longitude=-0.1278,
            city="London",
            country="UK"
        )
        orchestrator.add_mcdc("london_mcdc", loc, num_fpgas=500)
        
        status = orchestrator.get_cluster_status()\n        assert status['num_mcdc'] == 1\n        assert status['total_fpga'] == 500\n        assert status['compression_ratio'] == 500.0\n    \n    def test_network_links(self, orchestrator):\n        \"\"\"Test network link management\"\"\"\n        loc1 = MCDCLocation(\"A\", DeploymentRegion.NORTH_AMERICA, 40.7128, -74.0060, \"NYC\", \"USA\")\n        loc2 = MCDCLocation(\"B\", DeploymentRegion.EUROPE, 51.5074, -0.1278, \"LON\", \"UK\")\n        \n        orchestrator.add_mcdc(\"mcdc_a\", loc1)\n        orchestrator.add_mcdc(\"mcdc_b\", loc2)\n        \n        link = NetworkLink(\n            source_mcdc=\"mcdc_a\",\n            dest_mcdc=\"mcdc_b\",\n            bandwidth_mbps=100000,\n            latency_ms=10\n        )\n        orchestrator.add_network_link(link)\n        \n        # Check bidirectional link was created\n        assert len(orchestrator.network_graph[\"mcdc_a\"]) >= 1\n        assert len(orchestrator.network_graph[\"mcdc_b\"]) >= 1\n    \n    def test_cost_optimization(self, orchestrator):\n        \"\"\"Test placement optimization\"\"\"\n        loc = MCDCLocation(\n            location_id=\"OPT-1\",\n            region=DeploymentRegion.NORTH_AMERICA,\n            latitude=40.7128,\n            longitude=-74.0060,\n            city=\"New York\",\n            country=\"USA\"\n        )\n        orchestrator.add_mcdc(\"opt_mcdc\", loc)\n        \n        result = orchestrator.optimize_placement(\"opt_mcdc\", 15.0)\n        assert 'recommendations' in result\n        assert result['strategy'] == 'balanced'\n    \n    def test_federation_broadcast(self, orchestrator):\n        \"\"\"Test federation protocol\"\"\"\n        loc = MCDCLocation(\"FED\", DeploymentRegion.ASIA_PACIFIC, 35.6762, 139.6503, \"Tokyo\", \"Japan\")\n        orchestrator.add_mcdc(\"fed_mcdc\", loc)\n        \n        fed = FederationProtocol(orchestrator)\n        result = fed.broadcast_dictionary(\"test_hash_123\", \"fed_mcdc\")\n        \n        assert 'propagated_to' in result\n        assert \"fed_mcdc\" in result['propagated_to']\n\n\nclass TestCostOptimization:\n    \"\"\"Test cost optimization engine\"\"\"\n    \n    @pytest.fixture\n    def model(self):\n        return ComprehensiveEconomicModel(\n            num_fpga=5000,\n            fpga_board_model=FPGABoardModel.XILINX_U50,\n            lifespan_years=5\n        )\n    \n    def test_fpga_capex(self, model):\n        \"\"\"Test FPGA capital expenditure calculation\"\"\"\n        capex = model.calculate_fpga_capex()\n        \n        assert 'fpga_boards' in capex\n        assert 'container_infrastructure' in capex\n        assert 'total_capex' in capex\n        assert capex['total_capex'] > 0\n    \n    def test_fpga_opex(self, model):\n        \"\"\"Test FPGA operational expenditure\"\"\"\n        opex = model.calculate_fpga_opex(years=5)\n        \n        assert 'total_power_over_life' in opex\n        assert 'total_maintenance_over_life' in opex\n        assert 'total_opex_over_life' in opex\n        assert opex['total_opex_over_life'] > 0\n    \n    def test_cloud_costs(self, model):\n        \"\"\"Test cloud storage cost calculation\"\"\"\n        costs = model.calculate_cloud_costs(\n            data_size_eb=15.0,\n            provider=CloudProvider.GOOGLE_CLOUD,\n            access_pattern=\"hot\"\n        )\n        \n        assert 'total_storage_over_life' in costs\n        assert 'total_egress_over_life' in costs\n        assert 'total_cloud_cost_over_life' in costs\n    \n    def test_comparative_analysis(self, model):\n        \"\"\"Test comprehensive cost comparison\"\"\"\n        analysis = model.comparative_analysis(\n            data_size_eb=15.0,\n            cloud_provider=CloudProvider.GOOGLE_CLOUD\n        )\n        \n        assert 'fpga_infrastructure' in analysis\n        assert 'cloud_storage' in analysis\n        assert 'economics' in analysis\n        assert 'recommendation' in analysis\n        \n        # Check economics calculations\n        econ = analysis['economics']\n        assert 'fpga_cheaper_by' in econ\n        assert 'savings_percent' in econ\n    \n    def test_sensitivity_analysis(self, model):\n        \"\"\"Test sensitivity to parameter changes\"\"\"\n        # Test power sensitivity\n        sensitivity = model.sensitivity_analysis(\n            data_size_eb=15.0,\n            param_name=\"power_per_fpga_kw\",\n            param_range=[0.05, 0.08, 0.10]\n        )\n        \n        assert len(sensitivity) == 3\n        for result in sensitivity:\n            assert 'fpga_total_cost' in result\n            assert 'cloud_total_cost' in result\n            assert 'savings' in result\n    \n    def test_board_model_pricing(self):\n        \"\"\"Test different FPGA board model costs\"\"\"\n        models = [FPGABoardModel.XILINX_U50, FPGABoardModel.XILINX_U55C, FPGABoardModel.XILINX_U280]\n        \n        previous_cost = 0\n        for model_type in models:\n            m = ComprehensiveEconomicModel(fpga_board_model=model_type)\n            capex = m.calculate_fpga_capex()\n            # Verify costs increase with higher-end boards\n            assert capex['total_capex'] >= previous_cost\n            previous_cost = capex['total_capex']\n\n\nclass TestMobileContainerDC:\n    \"\"\"Test individual MCDC functionality\"\"\"\n    \n    def test_mcdc_creation(self):\n        \"\"\"Test MCDC instantiation\"\"\"\n        loc = MCDCLocation(\n            location_id=\"M-1\",\n            region=DeploymentRegion.ASIA_PACIFIC,\n            latitude=1.3521,\n            longitude=103.8198,\n            city=\"Singapore\",\n            country=\"Singapore\"\n        )\n        mcdc = MobileContainerDC(\n            mcdc_id=\"sg_mcdc\",\n            location=loc,\n            num_fpgas=500\n        )\n        \n        assert mcdc.mcdc_id == \"sg_mcdc\"\n        assert mcdc.num_fpgas == 500\n        assert mcdc.uptime_sla_pct == 99.95\n    \n    def test_mcdc_power_cost(self):\n        \"\"\"Test MCDC power cost calculation\"\"\"\n        loc = MCDCLocation(\"M\", DeploymentRegion.NORTH_AMERICA, 0, 0, \"Test\", \"Test\")\n        mcdc = MobileContainerDC(\n            mcdc_id=\"test\",\n            location=loc,\n            power_consumption_kw=400.0\n        )\n        \n        cost = mcdc.get_power_cost()\n        # ~$0.15/kWh × 400 kW × 8760 h/year\n        expected = 400.0 * 0.15 * 8760 / 1000000\n        assert abs(cost - expected) < 0.001\n    \n    def test_mcdc_summary(self):\n        \"\"\"Test MCDC summary output\"\"\"\n        loc = MCDCLocation(\n            location_id=\"S\",\n            region=DeploymentRegion.EUROPE,\n            latitude=48.8566,\n            longitude=2.3522,\n            city=\"Paris\",\n            country=\"France\"\n        )\n        mcdc = MobileContainerDC(\n            mcdc_id=\"paris_1\",\n            location=loc,\n            num_fpgas=500\n        )\n        \n        summary = mcdc.get_summary()\n        assert summary['mcdc_id'] == \"paris_1\"\n        assert summary['location']['city'] == \"Paris\"\n        assert summary['num_fpgas'] == 500\n\n\nif __name__ == \"__main__\":\n    pytest.main([__file__, \"-v\", \"--tb=short\"])\n