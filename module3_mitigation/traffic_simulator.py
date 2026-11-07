"""
Illustrative traffic profiles for the Module 3 dashboard.

These drive the single-event demo path. They are **not** the evaluation: the
authoritative numbers come from ``pipeline_evaluation.py``, which uses the
packet rates measured on every window of the real stream. Anything produced
here carries ``is_simulation=True``, and nothing in it should be quoted as a
result.

The profiles follow the same filter model as ``escalation_controller.py`` so the
dashboard and the batch evaluation agree:

* a *filter* caps the source at the normal 75th-percentile rate (495 pkt/s on
  the CICIoT2023 baseline), or the median (202 pkt/s) for the tighter caps;
* a *block* drops the source to zero;
* an action that only raises an alert leaves the rate unchanged.

Where a family appears in the evaluation stream, ``traffic_before`` is the
median rate measured for it in ``results/window_features.csv``. The rest are
marked illustrative.

One caveat worth repeating in any write-up: percentage traffic reduction only
describes volumetric attacks well. A WAF rule or an authentication back-off
stops the attack without necessarily moving the packet rate much, so a low
reduction there does not mean the mitigation failed.
"""

from typing import Dict, Any, Optional, Tuple

# Normal-traffic rate caps used by the filter model, in packets per second.
CAP_P75 = 495.0
CAP_P50 = 202.0


class TrafficSimulator:
    """Generates illustrative pre- and post-mitigation traffic observations."""

    # Per-family profiles. "measured" marks a traffic_before taken from the
    # dataset; "illustrative" marks a family the sampled stream does not contain.
    DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
        "Benign": {
            "traffic_before": 203.58,      # measured: median benign window
            "traffic_after": 203.58,       # no action taken
            "legitimate_traffic_impact": 0.0,
        },
        "Volumetric_Flood": {
            "traffic_before": 2367.46,     # measured: median DDoS-ICMP_Flood window
            "traffic_after": CAP_P75,      # per-source token bucket
            "legitimate_traffic_impact": 0.0,
        },
        "TCP_State_Exhaustion": {
            "traffic_before": 2356.68,     # measured: median DDoS-SYN_Flood window
            "traffic_after": CAP_P75,      # SYN cookies plus early expiry
            "legitimate_traffic_impact": 0.0,
        },
        "Fragmentation_Attack": {
            "traffic_before": 2000.0,      # illustrative
            "traffic_after": CAP_P75,
            "legitimate_traffic_impact": 0.0,
        },
        "Slow_Application_Attack": {
            "traffic_before": 300.0,       # illustrative; low rate by nature
            "traffic_after": CAP_P50,      # connection cap plus shorter timeouts
            "legitimate_traffic_impact": 2.0,
        },
        "Reconnaissance": {
            "traffic_before": 120.0,       # illustrative
            "traffic_after": 0.0,          # source blocked
            "legitimate_traffic_impact": 0.0,
        },
        "Spoofing_MITM": {
            "traffic_before": 240.0,       # illustrative
            "traffic_after": 0.0,          # forged frames dropped
            "legitimate_traffic_impact": 0.0,
        },
        "Web_Application_Attack": {
            "traffic_before": 350.0,       # illustrative
            "traffic_after": CAP_P50,      # WAF drops the malicious requests
            "legitimate_traffic_impact": 0.5,
        },
        "Credential_Attack": {
            "traffic_before": 400.0,       # illustrative
            "traffic_after": CAP_P50,      # authentication back-off
            "legitimate_traffic_impact": 2.0,
        },
        "Malware_Backdoor": {
            "traffic_before": 800.0,       # illustrative
            "traffic_after": 0.0,          # endpoint quarantined
            "legitimate_traffic_impact": 100.0,  # the device loses all traffic
        },
        "Botnet": {
            "traffic_before": 1200.0,      # illustrative
            "traffic_after": 0.0,          # device isolated
            "legitimate_traffic_impact": 100.0,
        },
        "Unknown": {
            "traffic_before": 654.69,      # measured: median held-out Mirai window
            "traffic_after": CAP_P75,      # behaviour route starts with a rate cap
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

