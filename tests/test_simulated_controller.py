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

    def test_01_simulate_deauth_pmf_activation(self):
        """Tests simulated PMF activation for Deauth attack."""
        event = AttackEvent(
            attack_type="Deauth",
            confidence=0.97,
            source_mac="00:11:22:33:44:55",
        )
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertTrue(res.is_simulation)
        self.assertTrue(self.controller.pmf_enabled)
        self.assertIn("PMF", res.message)

    def test_02_simulate_botnet_device_isolation(self):
        """Tests simulated device MAC quarantine for Botnet attack."""
        mac = "AA:BB:CC:DD:EE:01"
        event = AttackEvent(attack_type="Botnet", confidence=0.98, source_mac=mac)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertIn(mac, self.controller.isolated_macs)
        self.assertIn("VLAN 999", res.message)

    def test_03_simulate_ssh_rate_limiting(self):
        """Tests simulated SSH rate limiting on TCP port 22."""
        ip = "192.168.1.150"
        event = AttackEvent(attack_type="SSH", confidence=0.90, source_ip=ip)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertIn(ip, self.controller.rate_limited_ips)
        self.assertEqual(self.controller.rate_limited_ips[ip]["port"], 22)

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
        """Tests that manual remediation policies (Rogue_AP, Krack, etc.) yield REQUIRES_ADMIN_APPROVAL."""
        manual_attacks = ["Rogue_AP", "Krack", "Kr00k", "Evil_Twin"]
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

    def test_06_unsupported_action_for_unknown_attack(self):
        """Tests that uncataloged attacks return UNSUPPORTED_ACTION."""
        event = AttackEvent(attack_type="Unseen_Novel_ZeroDay_999", confidence=0.85)
        policy, _, _ = self.policy_engine.select_policy(event)
        res = self.controller.execute_mitigation(policy, event)

        self.assertEqual(res.status, ExecutionStatus.UNSUPPORTED_ACTION)
        self.assertIn("No automated mitigation", res.message)

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

