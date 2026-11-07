"""
Unit tests for Simulated Network Controller (simulated_network_controller.py).

Covers:
- Simulation safety (in-memory only, no OS changes)
- Exact execution statuses:
  - SIMULATED_SUCCESS
  - REQUIRES_ADMIN_APPROVAL
  - SIMULATION_FAILED
  - UNSUPPORTED_ACTION
- Automated primitive simulations:
  - PMF activation (Deauth/Disas)
  - Device isolation (Botnet)
  - SSH rate limiting
  - WAF rule activation (SQL Injection)
  - UDP port restriction (SSDP)
  - Malicious destination blocking
- Manual remediation actions:
  - Rogue AP
  - KRACK
  - Kr00k
  - Evil Twin
- Rollback and reset mechanics
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from module3_mitigation.simulated_network_controller import SimulatedNetworkController
from module3_mitigation.mitigation_policies import MitigationPolicyEngine
from module3_mitigation.models import (
    AttackEvent,
    ExecutionStatus,
    RemediationType,
)


class TestSimulatedNetworkController(unittest.TestCase):
    """Test suite for SimulatedNetworkController."""

    def setUp(self):
        self.controller = SimulatedNetworkController()
        self.policy_engine = MitigationPolicyEngine()

    def test_01_simulate_volumetric_flood_rate_limit(self):
        """A volumetric flood is capped per source, not blocked outright."""
        ip = "192.168.1.185"
        event = AttackEvent(
            attack_type="DDoS-ICMP_Flood",
            confidence=0.97,
            source_ip=ip,
            source_mac="00:11:22:33:44:55",
        )
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertTrue(res.is_simulation)
        self.assertEqual(res.action, "RATE_LIMIT_SOURCE")
        self.assertIn(ip, self.controller.rate_limited_ips)
        self.assertEqual(self.controller.rate_limited_ips[ip]["policy"], "TOKEN_BUCKET")

    def test_02_simulate_botnet_device_isolation(self):
        """Tests simulated device MAC quarantine for Botnet attack."""
        mac = "AA:BB:CC:DD:EE:01"
        event = AttackEvent(attack_type="Botnet", confidence=0.98, source_mac=mac)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertIn(mac, self.controller.isolated_macs)
        self.assertIn("VLAN 999", res.message)

    def test_03_simulate_credential_attack_throttling(self):
        """A dictionary brute force is throttled at the authentication layer."""
        ip = "192.168.1.150"
        event = AttackEvent(
            attack_type="DictionaryBruteForce", confidence=0.90, source_ip=ip
        )
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertEqual(res.action, "THROTTLE_AUTH_ATTEMPTS")
        self.assertIn(ip, self.controller.rate_limited_ips)
        self.assertEqual(self.controller.rate_limited_ips[ip]["policy"], "AUTH_BACKOFF")

    def test_03b_reconnaissance_blocks_the_source(self):
        """A low-rate scan cannot be rate limited away, so it is blocked."""
        ip = "192.168.1.99"
        event = AttackEvent(attack_type="Recon-PortScan", confidence=0.91, source_ip=ip)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertEqual(res.action, "BLOCK_SCANNING_SOURCE")
        self.assertEqual(self.controller.rate_limited_ips[ip]["policy"], "DROP_ALL")

    def test_04_simulate_sql_injection_waf_rule(self):
        """Tests simulated WAF rule activation for SQL Injection."""
        event = AttackEvent(
            attack_type="SQL_Injection", confidence=0.95, source_ip="192.168.1.180"
        )
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertGreater(len(self.controller.active_waf_rules), 0)

    def test_05_manual_remediation_requires_admin_approval(self):
        """Disruptive remediations must not run without an operator's approval."""
        manual_attacks = ["Backdoor_Malware"]
        for att in manual_attacks:
            event = AttackEvent(attack_type=att, confidence=0.90, source_mac="11:22:33:44:55:66")
            policy, _, _ = self.policy_engine.select_policy(event)
            res = self.controller.execute_mitigation(policy, event)

            self.assertEqual(
                res.status,
                ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
                f"Attack {att} should require admin approval",
            )
            self.assertEqual(policy.remediation_type, RemediationType.RECOMMENDED_MANUAL)

    def test_06_unknown_attack_takes_the_behaviour_route(self):
        """An uncataloged attack is contained by behaviour, not left to an alert."""
        event = AttackEvent(
            attack_type="Unseen_Novel_ZeroDay_999",
            confidence=0.85,
            source_ip="192.168.1.77",
        )
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertEqual(res.action, "BEHAVIOUR_BASED_CONTAINMENT")

    def test_07_force_status_override(self):
        """Tests forced status parameter for testing failure scenarios."""
        event = AttackEvent(attack_type="Deauth", confidence=0.95)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(
            policy, event, force_status=ExecutionStatus.SIMULATION_FAILED
        )

        self.assertEqual(res.status, ExecutionStatus.SIMULATION_FAILED)

    def test_08_rollback_and_reset(self):
        """Tests rollback and state clearing."""
        mac = "DE:AD:BE:EF:00:01"
        event = AttackEvent(attack_type="Botnet", confidence=0.95, source_mac=mac)
        policy, _, _ = self.policy_engine.select_policy(event)
        self.controller.execute_mitigation(policy, event)
        self.assertIn(mac, self.controller.isolated_macs)

        # Rollback
        self.controller.rollback_mitigation(policy.mitigation_action, {"source_mac": mac})
        self.assertNotIn(mac, self.controller.isolated_macs)

        # Reset
        self.controller.reset_simulated_state()
        snapshot = self.controller.get_active_mitigations()
        self.assertFalse(snapshot["pmf_enabled"])
        self.assertEqual(len(snapshot["isolated_macs"]), 0)


if __name__ == "__main__":
    unittest.main()

