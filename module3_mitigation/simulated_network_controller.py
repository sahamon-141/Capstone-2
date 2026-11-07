"""
Network Controller interface and Simulated Network Controller for Windows.

Implements an interface/adapter pattern so that when Raspberry Pi and real network
hardware become available, a real network controller (e.g., using iptables, hostapd_cli,
or ebtables) can seamlessly replace the simulator.

IMPORTANT: SimulatedNetworkController strictly operates in-memory.
It NEVER executes actual firewall commands, blocks real devices, modifies router
settings, disconnects real clients, or changes Windows network configuration.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set

from module3_mitigation.models import (
    ExecutionResult,
    ExecutionStatus,
    MitigationPolicy,
    RemediationType,
    AttackEvent,
)


class BaseNetworkController(ABC):
    """Abstract interface defining contract for network controllers."""

    @abstractmethod
    def execute_mitigation(
        self,
        policy: MitigationPolicy,
        event: AttackEvent,
        force_status: Optional[ExecutionStatus] = None,
    ) -> ExecutionResult:
        """Executes or simulates the specified mitigation action."""
        pass

    @abstractmethod
    def rollback_mitigation(self, action: str, target: Dict[str, Any]) -> ExecutionResult:
        """Rolls back an executed or simulated mitigation action."""
        pass

    @abstractmethod
    def get_active_mitigations(self) -> Dict[str, Any]:
        """Returns the current active mitigation state."""
        pass


class SimulatedNetworkController(BaseNetworkController):
    """
    In-memory simulation controller running safely on Windows.

    Models network security primitives:
    - 802.11w Protected Management Frames (PMF) activation
    - MAC address quarantine / station isolation
    - IP rate limiting / firewall rules
    - Web Application Firewall (WAF) regex inspection
    - UDP port restriction (SSDP port 1900)
    - DNS resolver destination blacklisting
    - Rogue AP / Evil Twin administrator alerts
    """

    def __init__(self):
        # In-memory simulated network state
        self.pmf_enabled: bool = False
        self.isolated_macs: Set[str] = set()
        self.rate_limited_ips: Dict[str, Dict[str, Any]] = {}
        self.active_waf_rules: List[Dict[str, Any]] = []
        self.restricted_ports: Set[int] = set()
        self.blocked_destinations: Set[str] = set()
        self.admin_alerts: List[Dict[str, Any]] = []
        self.action_history: List[Dict[str, Any]] = []

    def reset_simulated_state(self) -> None:
        """Resets all simulated network state to baseline defaults."""
        self.pmf_enabled = False
        self.isolated_macs.clear()
        self.rate_limited_ips.clear()
        self.active_waf_rules.clear()
        self.restricted_ports.clear()
        self.blocked_destinations.clear()
        self.admin_alerts.clear()
        self.action_history.clear()

    def get_active_mitigations(self) -> Dict[str, Any]:
        """Returns snapshot of current simulated network modifications."""
        return {
            "pmf_enabled": self.pmf_enabled,
            "isolated_macs": sorted(list(self.isolated_macs)),
            "rate_limited_ips": self.rate_limited_ips,
            "active_waf_rules": self.active_waf_rules,
            "restricted_ports": sorted(list(self.restricted_ports)),
            "blocked_destinations": sorted(list(self.blocked_destinations)),
            "pending_admin_alerts": len(self.admin_alerts),
            "total_actions_recorded": len(self.action_history),
        }

    def execute_mitigation(
        self,
        policy: MitigationPolicy,
        event: AttackEvent,
        force_status: Optional[ExecutionStatus] = None,
    ) -> ExecutionResult:
        """
        Simulates the execution of a mitigation action.

        Returns one of four exact statuses:
        - SIMULATED_SUCCESS
        - REQUIRES_ADMIN_APPROVAL
        - SIMULATION_FAILED
        - UNSUPPORTED_ACTION
        """
        # Allow testing scenario 8 (forced failure) or manual test overrides
        if force_status is not None:
            return self._build_result(
                action=policy.mitigation_action,
                status=force_status,
                message=f"[SIMULATED] Action outcome explicitly forced to {force_status.value}",
                target=self._extract_target(event),
                details={"forced": True},
            )

        target = self._extract_target(event)

        # 1. Alert-only policies change no traffic and need an operator.
        # "Unknown" is no longer routed here: a zero-day window gets a
        # behaviour-based containment action instead of just an alert.
        if policy.mitigation_action == "ALERT_ADMIN_AND_LOG":
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "UNSUPPORTED_ATTACK_ALERT",
                "attack_type": event.attack_type,
                "confidence": event.confidence,
                "target": target,
                "note": "Unknown attack detected. Queued for manual triage.",
            }
            self.admin_alerts.append(alert)
            return self._build_result(
                action=policy.mitigation_action,
                status=ExecutionStatus.UNSUPPORTED_ACTION,
                message=(
                    f"[SIMULATED] No automated mitigation available for '{event.attack_type}'. "
                    "Forensic alert queued for security administrator."
                ),
                target=target,
                details={"alert": alert},
            )

        # 2. Check if action requires administrator approval (e.g., Rogue AP, KRACK, Kr00k, Evil Twin)
        # or if low confidence triggered an admin approval requirement
        if (
            policy.remediation_type == RemediationType.RECOMMENDED_MANUAL
            or policy.requires_admin_approval
        ):
            alert = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": policy.mitigation_action,
                "attack_type": policy.attack_type,
                "description": policy.description,
                "target": target,
                "requires_approval": True,
            }
            self.admin_alerts.append(alert)
            return self._build_result(
                action=policy.mitigation_action,
                status=ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
                message=(
                    f"[SIMULATED] Mitigation requires administrator confirmation: {policy.description}."
                ),
                target=target,
                details={"alert": alert, "remediation_type": policy.remediation_type.value},
            )

        # 3. Simulate Automated Actions
        action = policy.mitigation_action

        if action in ("ENABLE_PMF_AND_MONITOR", "ENABLE_PMF_MONITOR_DISAS"):
            self.pmf_enabled = True
            msg = (
                f"[SIMULATED] 802.11w Protected Management Frames (PMF) set to REQUIRED on simulated AP. "
                f"Monitoring management frames from MAC {target.get('source_mac', 'Unknown')}."
            )
            return self._record_success(action, target, msg, {"pmf_status": "REQUIRED"})

        elif action == "ENFORCE_AUTH_ASSOC_POLICY":
            mac = target.get("source_mac", "Unknown")
            msg = (
                f"[SIMULATED] AP Association rate limit enforced (max 5 frames/sec). "
                f"Strict association policy applied for MAC {mac}."
            )
            return self._record_success(action, target, msg, {"assoc_rate_limit_fps": 5})

        elif action == "APPLY_SSH_RATE_LIMIT":
            ip = target.get("source_ip", "192.168.1.105")
            self.rate_limited_ips[ip] = {
                "port": 22,
                "protocol": "TCP",
                "limit": "3_conns_per_minute",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            msg = (
                f"[SIMULATED] SSH connection rate limit applied on TCP port 22 for source IP {ip} "
                "(3 connections / minute max)."
            )
            return self._record_success(action, target, msg, {"rate_limit": "3/min", "port": 22})

        elif action == "ISOLATE_INFECTED_DEVICE":
            mac = target.get("source_mac", "AA:BB:CC:DD:EE:FF")
            self.isolated_macs.add(mac)
            msg = (
                f"[SIMULATED] Device with MAC {mac} assigned to isolated quarantine VLAN (VLAN 999). "
                "Intra-LAN forwarding dropped."
            )
            return self._record_success(action, target, msg, {"quarantine_vlan": 999, "isolated_mac": mac})

        elif action == "QUARANTINE_ENDPOINT":
            mac = target.get("source_mac", "AA:BB:CC:DD:EE:FF")
            ip = target.get("source_ip", "192.168.1.105")
            self.isolated_macs.add(mac)
            if ip:
                self.rate_limited_ips[ip] = {"policy": "DROP_ALL", "port": "ALL"}
            msg = (
                f"[SIMULATED] Endpoint quarantined: MAC {mac}, IP {ip}. Outbound C2 access blocked."
            )
            return self._record_success(action, target, msg, {"endpoint_quarantined": True})

        elif action == "ACTIVATE_WAF_RULE":
            ip = target.get("source_ip", "192.168.1.105")
            rule = {
                "rule_id": f"WAF_SQLI_{len(self.active_waf_rules) + 1}",
                "target_ip": ip,
                "pattern": "(UNION|SELECT|DROP|INSERT|--|OR 1=1)",
                "action": "DROP_REQUEST_AND_LOG",
            }
            self.active_waf_rules.append(rule)
            msg = (
                f"[SIMULATED] WAF Rule {rule['rule_id']} activated. SQL injection signatures filtered on port 80/443."
            )
            return self._record_success(action, target, msg, {"waf_rule": rule})

        elif action == "RESTRICT_UDP_PORT_1900":
            self.restricted_ports.add(1900)
            msg = (
                "[SIMULATED] Inbound and forwarding traffic for UDP port 1900 (SSDP/UPnP) dropped at gateway router."
            )
            return self._record_success(action, target, msg, {"restricted_port": 1900, "protocol": "UDP"})

        elif action == "BLOCK_MALICIOUS_DESTINATION":
            dest_ip = target.get("destination_ip", "198.51.100.42")
            self.blocked_destinations.add(dest_ip)
            msg = (
                f"[SIMULATED] Malicious destination {dest_ip} blocked in DNS sinkhole and gateway route table."
            )
            return self._record_success(action, target, msg, {"blocked_destination": dest_ip})

        # ------------------------------------------------------------------
        # CICIoT2023 attack-family actions (policies.json v2)
        # ------------------------------------------------------------------
        elif action == "NO_ACTION":
            return self._record_success(
                action, target,
                "[SIMULATED] Traffic attributed to the benign class. No mitigation applied.",
                {"traffic_affected": False},
            )

        elif action == "RATE_LIMIT_SOURCE":
            ip = target.get("source_ip", "192.168.1.105")
            self.rate_limited_ips[ip] = {
                "policy": "TOKEN_BUCKET",
                "cap": "normal_baseline_p95",
                "port": "ALL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            msg = (
                f"[SIMULATED] Per-source token bucket applied to {ip}, capped at the "
                "normal-traffic 95th percentile rate. Excess packets dropped."
            )
            return self._record_success(action, target, msg, {"cap": "p95", "source_ip": ip})

        elif action == "ENABLE_SYN_COOKIES":
            msg = (
                "[SIMULATED] SYN cookies enabled on the gateway, half-open connection "
                "timeout reduced and packets with invalid flag combinations dropped."
            )
            return self._record_success(action, target, msg, {"syn_cookies": True})

        elif action == "DROP_MALFORMED_FRAGMENTS":
            msg = (
                "[SIMULATED] Per-source IP reassembly buffer limit enforced. Overlapping "
                "and incomplete fragments dropped before reassembly."
            )
            return self._record_success(action, target, msg, {"fragment_filter": True})

        elif action == "LIMIT_CONCURRENT_CONNECTIONS":
            ip = target.get("source_ip", "192.168.1.105")
            msg = (
                f"[SIMULATED] Concurrent connection cap applied to {ip} with shortened "
                "header and body read timeouts, so partial requests are closed."
            )
            return self._record_success(action, target, msg, {"conn_cap": True, "source_ip": ip})

        elif action == "THROTTLE_AUTH_ATTEMPTS":
            ip = target.get("source_ip", "192.168.1.105")
            self.rate_limited_ips[ip] = {"policy": "AUTH_BACKOFF", "port": "AUTH"}
            msg = (
                f"[SIMULATED] Exponential back-off applied to authentication attempts from {ip}. "
                "Targeted account locked after repeated failures."
            )
            return self._record_success(action, target, msg, {"auth_backoff": True})

        elif action in ("BLOCK_SOURCE", "BLOCK_SCANNING_SOURCE"):
            ip = target.get("source_ip", "192.168.1.105")
            self.rate_limited_ips[ip] = {"policy": "DROP_ALL", "port": "ALL"}
            extra = (
                " Further probes redirected to a honeypot for observation."
                if action == "BLOCK_SCANNING_SOURCE" else ""
            )
            msg = f"[SIMULATED] All traffic from source {ip} dropped at the gateway.{extra}"
            return self._record_success(action, target, msg, {"blocked_source": ip})

        elif action == "UPSTREAM_BLACKHOLE":
            ip = target.get("source_ip", "192.168.1.105")
            self.blocked_destinations.add(ip)
            msg = (
                f"[SIMULATED] Upstream blackhole route installed for {ip}, so the flood is "
                "discarded before it reaches the local link."
            )
            return self._record_success(action, target, msg, {"blackhole_source": ip})

        elif action == "ISOLATE_SEGMENT":
            mac = target.get("source_mac", "AA:BB:CC:DD:EE:FF")
            self.isolated_macs.add(mac)
            msg = (
                "[SIMULATED] Affected network segment moved to an isolated VLAN and packet "
                "capture started for forensic analysis."
            )
            return self._record_success(action, target, msg, {"segment_isolated": True})

        elif action == "ENFORCE_ARP_DNS_INTEGRITY":
            msg = (
                "[SIMULATED] Dynamic ARP inspection enabled with static bindings for known "
                "devices, and DNSSEC validation enforced on resolver responses."
            )
            return self._record_success(action, target, msg, {"arp_inspection": True, "dnssec": True})

        elif action == "BEHAVIOUR_BASED_CONTAINMENT":
            # The concrete rung is chosen by ZeroDayContainment and executed by
            # the escalation controller; this records that the route was taken.
            msg = (
                "[SIMULATED] No class-specific policy matched. Containment selected from the "
                "traffic behaviour measured by Module 1."
            )
            return self._record_success(action, target, msg, {"zero_day_route": True})

        else:
            return self._build_result(
                action=action,
                status=ExecutionStatus.UNSUPPORTED_ACTION,
                message=f"[SIMULATED] Unknown action string '{action}'. Cannot simulate.",
                target=target,
                details={},
            )

    def rollback_mitigation(self, action: str, target: Dict[str, Any]) -> ExecutionResult:
        """Simulates rollback of an active mitigation."""
        mac = target.get("source_mac")
        ip = target.get("source_ip")

        if mac and mac in self.isolated_macs:
            self.isolated_macs.remove(mac)
        if ip and ip in self.rate_limited_ips:
            del self.rate_limited_ips[ip]

        msg = f"[SIMULATED] Mitigation '{action}' rolled back for target {target}."
        return self._build_result(
            action=action,
            status=ExecutionStatus.SIMULATED_SUCCESS,
            message=msg,
            target=target,
            details={"rollback": True},
        )

    def _extract_target(self, event: AttackEvent) -> Dict[str, Any]:
        """Extracts addressing information from event."""
        return {
            "source_mac": event.source_mac or "AA:BB:CC:DD:EE:FF",
            "source_ip": event.source_ip,
            "destination_ip": event.destination_ip,
            "sensor_id": event.sensor_id,
        }

    def _record_success(
        self, action: str, target: Dict[str, Any], message: str, details: Dict[str, Any]
    ) -> ExecutionResult:
        res = self._build_result(
            action=action,
            status=ExecutionStatus.SIMULATED_SUCCESS,
            message=message,
            target=target,
            details=details,
        )
        self.action_history.append(res.to_dict())
        return res

    def _build_result(
        self,
        action: str,
        status: ExecutionStatus,
        message: str,
        target: Dict[str, Any],
        details: Dict[str, Any],
    ) -> ExecutionResult:
        return ExecutionResult(
            action=action,
            status=status,
            message=message,
            target=target,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_simulation=True,
            details=details,
        )

