"""
Reproducible demonstration scenarios for Module 3.

These drive the dashboard's one-click buttons. They cover the CICIoT2023
classes Module 2 actually predicts, and between them exercise every outcome the
mitigation stage can produce: an automated filter, an automated block, the
zero-day behaviour route, a low-confidence hand-off to an operator, an action
that needs administrator approval, and a window that correctly receives no
action at all.

Where a scenario's class appears in the evaluation stream, ``traffic_before`` is
the **median observed packet rate** for that class in
``results/window_features.csv`` rather than an invented number, and
``traffic_after`` follows the filter model in ``escalation_controller.py``: a
cap at the normal 75th-percentile rate (495 pkt/s on this baseline) for a
filter, and zero for a block. Classes that the sampled stream does not contain
are marked ``illustrative`` and use plausible values instead; they demonstrate
the policy path, not a measurement.
"""

from typing import Dict, Any, List

from module3_mitigation.models import (
    AttackEvent,
    ExecutionStatus,
    VerificationStatus,
)

# Normal-traffic rate cap used by the filter model, in packets per second.
RATE_CAP_P75 = 495.0


SCENARIOS_CATALOG: Dict[int, Dict[str, Any]] = {
    1: {
        "id": 1,
        "name": "DDoS ICMP Flood (per-source rate limit)",
        "description": (
            "Volumetric flood family. A token bucket caps the source at the normal "
            "75th-percentile rate. Median observed rate for this class in the dataset."
        ),
        "attack_type": "DDoS-ICMP_Flood",
        "confidence": 0.96,
        "source_mac": "E4:5F:01:AB:CD:EF",
        "source_ip": "192.168.1.185",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_01",
        "traffic_before": 2367.46,
        "traffic_after": RATE_CAP_P75,
        "data_source": "median of 1,138 DDoS-ICMP_Flood windows",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.PARTIALLY_EFFECTIVE,
        "expected_reduction_pct": 79.09,
    },
    2: {
        "id": 2,
        "name": "DDoS SYN Flood (SYN cookies)",
        "description": (
            "TCP state exhaustion family. SYN cookies are enabled and half-open "
            "connections expire early. Median observed rate for this class."
        ),
        "attack_type": "DDoS-SYN_Flood",
        "confidence": 0.94,
        "source_mac": "B8:27:EB:12:34:56",
        "source_ip": "192.168.1.140",
        "destination_ip": "192.168.1.10",
        "sensor_id": "GATEWAY_WIN",
        "traffic_before": 2356.68,
        "traffic_after": RATE_CAP_P75,
        "data_source": "median of 162 DDoS-SYN_Flood windows",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.PARTIALLY_EFFECTIVE,
        "expected_reduction_pct": 79.00,
    },
    3: {
        "id": 3,
        "name": "Mirai flood, held out as zero-day (behaviour route)",
        "description": (
            "The Mirai family is excluded from training, so no class policy matches "
            "and containment is chosen from the measured behaviour: CV of 0.0 at an "
            "above-median rate is a constant-rate flood, so the source is rate limited."
        ),
        "attack_type": "Mirai-greeth_flood",
        "confidence": 0.20,
        "source_mac": "AC:DE:48:00:11:22",
        "source_ip": "192.168.1.205",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_02",
        "traffic_before": 654.69,
        "traffic_after": RATE_CAP_P75,
        "data_source": "median of 552 Mirai-greeth_flood windows",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 24.39,
    },
    4: {
        "id": 4,
        "name": "Recon port scan (block the scanning source)",
        "description": (
            "Reconnaissance family. A rate limit does not help a low-rate probe, so "
            "the policy blocks the source outright and redirects it to a honeypot. "
            "Illustrative: this class is not present in the sampled stream."
        ),
        "attack_type": "Recon-PortScan",
        "confidence": 0.91,
        "source_mac": "52:54:00:12:34:56",
        "source_ip": "192.168.1.99",
        "destination_ip": "192.168.1.0/24",
        "sensor_id": "GATEWAY_WIN",
        "traffic_before": 120.0,
        "traffic_after": 0.0,
        "data_source": "illustrative",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_EFFECTIVE,
        "expected_reduction_pct": 100.0,
    },
    5: {
        "id": 5,
        "name": "ARP spoofing / MITM (ARP and DNS integrity)",
        "description": (
            "Spoofing family. Dynamic ARP inspection with static bindings and DNSSEC "
            "validation drop the forged frames. Illustrative: not in the sampled stream."
        ),
        "attack_type": "MITM-ArpSpoofing",
        "confidence": 0.89,
        "source_mac": "70:85:C2:55:66:77",
        "source_ip": "192.168.1.112",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_03",
        "traffic_before": 240.0,
        "traffic_after": 0.0,
        "data_source": "illustrative",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_EFFECTIVE,
        "expected_reduction_pct": 100.0,
    },
    6: {
        "id": 6,
        "name": "SQL injection at low confidence (operator confirmation)",
        "description": (
            "Web application family. Confidence of 0.42 is below the 0.70 threshold "
            "for this policy, so the WAF rule is held for an operator to confirm "
            "rather than applied automatically. Illustrative."
        ),
        "attack_type": "SqlInjection",
        "confidence": 0.42,
        "source_mac": "70:85:C2:55:66:77",
        "source_ip": "192.168.1.112",
        "destination_ip": "192.168.1.80",
        "sensor_id": "ESP32_01",
        "traffic_before": 350.0,
        "traffic_after": 350.0,
        "data_source": "illustrative",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
    },
    7: {
        "id": 7,
        "name": "Backdoor malware (quarantine needs approval)",
        "description": (
            "Malware family. Quarantining an endpoint cuts off all of its traffic, so "
            "the policy requires an administrator to approve it even at high "
            "confidence. Illustrative."
        ),
        "attack_type": "Backdoor_Malware",
        "confidence": 0.93,
        "source_mac": "F0:9F:C2:33:44:55",
        "source_ip": "192.168.1.160",
        "destination_ip": "198.51.100.77",
        "sensor_id": "ESP32_03",
        "traffic_before": 800.0,
        "traffic_after": 800.0,
        "data_source": "illustrative",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
    },
    8: {
        "id": 8,
        "name": "Benign window flagged by Module 1 (no action)",
        "description": (
            "A Module 1 false positive. Module 2 attributes it to the benign class, so "
            "Module 3 correctly takes no action and the traffic is untouched. Median "
            "observed rate for benign windows."
        ),
        "attack_type": "BenignTraffic",
        "confidence": 0.88,
        "source_mac": "00:14:22:01:23:45",
        "source_ip": "192.168.1.52",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_01",
        "traffic_before": 203.58,
        "traffic_after": 203.58,
        "data_source": "median of 1,501 BenignTraffic windows",
        "force_status": None,
        "expected_execution_status": ExecutionStatus.SIMULATED_SUCCESS,
        "expected_verification_status": VerificationStatus.MITIGATION_FAILED,
        "expected_reduction_pct": 0.0,
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
