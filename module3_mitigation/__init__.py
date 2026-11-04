"""
Module 3: Attack-Specific Mitigation and Verification (Windows Simulation).

Components:
- MitigationEngine: Master coordinator
- MitigationPolicyEngine: Policy lookup & validation
- SimulatedNetworkController: In-memory network controller for Windows
- TrafficSimulator: Reproducible pre/post traffic telemetry
- MitigationVerifier: Verification engine (% reduction math)
- MitigationDatabase: SQLite persistence
- Scenarios: 8 reproducible simulation scenarios
"""

from module3_mitigation.models import (
    AttackEvent,
    MitigationPolicy,
    ExecutionResult,
    VerificationResult,
    MitigationRecord,
    ExecutionStatus,
    VerificationStatus,
    RemediationType,
)
from module3_mitigation.mitigation_policies import MitigationPolicyEngine
from module3_mitigation.simulated_network_controller import (
    BaseNetworkController,
    SimulatedNetworkController,
)
from module3_mitigation.traffic_simulator import TrafficSimulator
from module3_mitigation.mitigation_verifier import MitigationVerifier
from module3_mitigation.database import MitigationDatabase
from module3_mitigation.mitigation_engine import MitigationEngine
from module3_mitigation.scenarios import (
    SCENARIOS_CATALOG,
    get_scenario,
    list_all_scenarios,
)

__all__ = [
    "AttackEvent",
    "MitigationPolicy",
    "ExecutionResult",
    "VerificationResult",
    "MitigationRecord",
    "ExecutionStatus",
    "VerificationStatus",
    "RemediationType",
    "MitigationPolicyEngine",
    "BaseNetworkController",
    "SimulatedNetworkController",
    "TrafficSimulator",
    "MitigationVerifier",
    "MitigationDatabase",
    "MitigationEngine",
    "SCENARIOS_CATALOG",
    "get_scenario",
    "list_all_scenarios",
]

