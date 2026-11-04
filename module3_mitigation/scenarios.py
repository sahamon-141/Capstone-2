"""
Reproducible Simulation Scenarios for Module 3.

Contains the 8 required testable simulation scenarios:
1. Deauth attack – successful simulated mitigation (PMF enabled, 97% reduction)
2. Botnet attack – successful simulated device isolation (100% reduction)
3. SSH attack – partial mitigation (Rate limiting, 55% reduction)
4. Rogue AP – administrator alert (Manual sweep required, REQUIRES_ADMIN_APPROVAL)
5. Malware – simulated quarantine (Outbound C2 blocked, 97.5% reduction)
6. Unknown attack – no matching policy (Fallback policy, UNSUPPORTED_ACTION)
7. Low-confidence prediction – require confirmation (Confidence < threshold)
8. Failed mitigation – verification reports failure (Reduction < 30%)
"""

from typing import Dict, Any, List, Optional
from module3_mitigation.models import (
    AttackEvent,
    ExecutionStatus,
    VerificationStatus,
)


SCENARIOS_CATALOG: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "Deauth Attack (Successful PMF Activation)",
        "description": "Simulates 802.11 deauthentication flood mitigation by enabling PMF on simulated AP.",
        "attack_type": "Deauth",
        "confidence": 0.97,
        "source_mac": "00:14:22:01:23:45",
        "source_ip": None,
        "destination_ip": None,
        "sensor_id": "ESP32_01",
        "traffic_before": 500.0,
        "traffic_after": 15.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_EFFECTIVE,
        "expected_reduction_pct": 97.0,
    },
    2: {
        "id": 2,
        "name": "Mirai / Botnet Attack (Device Isolation)",
        "description": "Simulates infected IoT device isolation by assigning its MAC to quarantine VLAN 999.",
        "attack_type": "Botnet",
        "confidence": 0.98,
        "source_mac": "E4:5F:01:AB:CD:EF",
        "source_ip": "192.168.1.185",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_02",
        "traffic_before": 1200.0,
        "traffic_after": 0.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_EFFECTIVE,
        "expected_reduction_pct": 100.0,
    },
    3: {
        "id": 3,
        "name": "SSH Brute-Force Attack (Partial Rate Limiting)",
        "description": "Simulates SSH connection throttling on TCP port 22, resulting in partial suppression.",
        "attack_type": "SSH",
        "confidence": 0.92,
        "source_mac": "B8:27:EB:12:34:56",
        "source_ip": "192.168.1.140",
        "destination_ip": "192.168.1.10",
        "sensor_id": "GATEWAY_WIN",
        "traffic_before": 400.0,
        "traffic_after": 180.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.PARTIALLY_EFFECTIVE,
        "expected_reduction_pct": 55.0,
    },
    4: {
        "id": 4,
        "name": "Rogue AP Detected (Administrator Alert)",
        "description": "Simulates detection of unauthorized access point. Requires manual site investigation.",
        "attack_type": "Rogue_AP",
        "confidence": 0.94,
        "source_mac": "68:DB:F5:88:99:AA",
        "source_ip": None,
        "destination_ip": None,
        "sensor_id": "ESP32_01",
        "traffic_before": 180.0,
        "traffic_after": 180.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
    },
    5: {
        "id": 5,
        "name": "Malware / C2 Beaconing (Endpoint Quarantine)",
        "description": "Simulates endpoint quarantine and dropping outbound command-and-control communication.",
        "attack_type": "Malware",
        "confidence": 0.95,
        "source_mac": "AC:DE:48:00:11:22",
        "source_ip": "192.168.1.205",
        "destination_ip": "198.51.100.77",
        "sensor_id": "ESP32_03",
        "traffic_before": 800.0,
        "traffic_after": 20.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_EFFECTIVE,
        "expected_reduction_pct": 97.5,
    },
    6: {
        "id": 6,
        "name": "Unknown Exploit / Zero-Day (No Matching Policy)",
        "description": "Simulates arrival of an uncataloged exploit category. Fallback policy logs forensic alert.",
        "attack_type": "Unknown_Novel_Exploit_X",
        "confidence": 0.70,
        "source_mac": "52:54:00:12:34:56",
        "source_ip": "192.168.1.99",
        "destination_ip": "192.168.1.1",
        "sensor_id": "GATEWAY_WIN",
        "traffic_before": 300.0,
        "traffic_after": 300.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.UNSUPPORTED_ACTION,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
    },
    7: {
        "id": 7,
        "name": "Low-Confidence Prediction (Admin Confirmation Required)",
        "description": "Prediction confidence (0.42) falls below threshold (0.60). System requires human verification.",
        "attack_type": "SQL_Injection",
        "confidence": 0.42,
        "source_mac": "70:85:C2:55:66:77",
        "source_ip": "192.168.1.112",
        "destination_ip": "192.168.1.80",
        "sensor_id": "ESP32_01",
        "traffic_before": 350.0,
        "traffic_after": 350.0,
        "force_status": None,
        "expected_execution_status": ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
    },
    8: {
        "id": 8,
        "name": "Failed Mitigation (Verification Reports Failure)",
        "description": "Simulates mitigation failure where suspicious traffic remains unsuppressed (< 30% reduction).",
        "attack_type": "SSDP",
        "confidence": 0.88,
        "source_mac": "F0:9F:C2:33:44:55",
        "source_ip": "192.168.1.160",
        "destination_ip": "239.255.255.250",
        "sensor_id": "ESP32_01",
        "traffic_before": 1000.0,
        "traffic_after": 950.0,  # Only 5% reduction
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 5.0,
    },
}


def get_scenario(scenario_id: int) -> Dict[str, Any]:
    """Retrieves scenario details by ID (1..8)."""
    if scenario_id not in SCENARIOS_CATALOG:
        raise KeyError(f"Invalid scenario ID {scenario_id}. Available IDs: 1..8")
    return SCENARIOS_CATALOG[scenario_id].copy()


def list_all_scenarios() -> List[Dict[str, Any]]:
    """Returns list of all 8 scenario definitions."""
    return list(SCENARIOS_CATALOG.values())

