"""
Traffic Simulator for Module 3.

Provides configurable, reproducible synthetic network traffic measurements
before and after mitigation. Kept strictly separate from the verification engine
to ensure verification mathematics remain unpolluted and testable across
different outcomes (effective, partial, failed, zero-traffic, etc.).
"""

from typing import Dict, Any, Optional, Tuple


class TrafficSimulator:
    """Generates synthetic pre- and post-mitigation traffic observations."""

    # Default baseline profiles per attack category (units: packets/minute over 60s window)
    DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
        "Deauth": {
            "traffic_before": 500.0,
            "traffic_after": 15.0,
            "legitimate_traffic_impact": 0.0,
        },
        "Disas": {
            "traffic_before": 450.0,
            "traffic_after": 12.0,
            "legitimate_traffic_impact": 0.0,
        },
        "(Re)Assoc": {
            "traffic_before": 620.0,
            "traffic_after": 25.0,
            "legitimate_traffic_impact": 1.0,
        },
        "Rogue_AP": {
            "traffic_before": 180.0,
            "traffic_after": 180.0,  # Requires physical/manual admin removal
            "legitimate_traffic_impact": 0.0,
        },
        "Krack": {
            "traffic_before": 220.0,
            "traffic_after": 220.0,  # Firmware update required
            "legitimate_traffic_impact": 0.0,
        },
        "Kr00k": {
            "traffic_before": 210.0,
            "traffic_after": 210.0,  # Firmware update required
            "legitimate_traffic_impact": 0.0,
        },
        "SSH": {
            "traffic_before": 400.0,
            "traffic_after": 180.0,  # Rate limiting partially throttles brute force
            "legitimate_traffic_impact": 2.0,
        },
        "Botnet": {
            "traffic_before": 1200.0,
            "traffic_after": 0.0,  # Full isolation
            "legitimate_traffic_impact": 100.0,  # Total isolation halts all device traffic
        },
        "Malware": {
            "traffic_before": 800.0,
            "traffic_after": 20.0,
            "legitimate_traffic_impact": 85.0,
        },
        "SQL_Injection": {
            "traffic_before": 350.0,
            "traffic_after": 10.0,  # WAF filters payloads
            "legitimate_traffic_impact": 0.5,
        },
        "SSDP": {
            "traffic_before": 650.0,
            "traffic_after": 8.0,  # Port 1900 restricted
            "legitimate_traffic_impact": 5.0,  # May affect UPnP discovery
        },
        "Evil_Twin": {
            "traffic_before": 240.0,
            "traffic_after": 240.0,  # Requires manual operator triage
            "legitimate_traffic_impact": 0.0,
        },
        "Website_spoofing": {
            "traffic_before": 420.0,
            "traffic_after": 5.0,  # DNS sinkhole blocks requests
            "legitimate_traffic_impact": 0.0,
        },
        "Unknown": {
            "traffic_before": 300.0,
            "traffic_after": 300.0,
            "legitimate_traffic_impact": 0.0,
        },
    }

    def __init__(self, default_window_seconds: int = 60):
        self.default_window_seconds = default_window_seconds

    def generate_measurements(
        self,
        attack_type: str,
        traffic_before: Optional[float] = None,
        traffic_after: Optional[float] = None,
        outcome_mode: Optional[str] = None,
        observation_window_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates simulated pre- and post-mitigation traffic measurements.

        Args:
            attack_type: Name of attack (used to select default profile)
            traffic_before: Explicit pre-mitigation traffic (if provided)
            traffic_after: Explicit post-mitigation traffic (if provided)
            outcome_mode: "effective", "partial", "failed", "zero_traffic", or "increased"
            observation_window_seconds: Equal observation window length (default: 60s)
        """
        window_sec = observation_window_seconds or self.default_window_seconds

        # 1. Start from profile or generic baseline
        profile = self.DEFAULT_PROFILES.get(
            attack_type, self.DEFAULT_PROFILES["Unknown"]
        ).copy()

        before = float(profile["traffic_before"] if traffic_before is None else traffic_before)
        after = float(profile["traffic_after"] if traffic_after is None else traffic_after)
        legit_impact = float(profile["legitimate_traffic_impact"])

        # 2. Apply outcome mode override if requested
        if outcome_mode == "effective":
            before = max(before, 500.0)
            after = before * 0.05  # 95% reduction
        elif outcome_mode == "partial":
            before = max(before, 500.0)
            after = before * 0.45  # 55% reduction
        elif outcome_mode == "failed":
            before = max(before, 500.0)
            after = before * 0.95  # 5% reduction (mitigation failed)
        elif outcome_mode == "zero_traffic":
            before = 0.0
            after = 0.0
        elif outcome_mode == "increased":
            before = 300.0
            after = 450.0  # Traffic surged

        return {
            "traffic_before": round(before, 2),
            "traffic_after": round(after, 2),
            "observation_window_seconds": window_sec,
            "legitimate_traffic_impact": round(legit_impact, 2),
            "unit": "packets_per_minute",
            "is_simulation": True,
        }

