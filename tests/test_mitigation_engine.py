"""
Comprehensive test suite for Module 3 Mitigation Engine (mitigation_engine.py).

Covers:
- Every supported attack category (13 AWID3 + CICIoT2023 mappings)
- Correct policy selection and alias resolution
- Low-confidence prediction handling
- Unknown attack categories
- Successful, partial, and failed simulated mitigations
- Zero baseline traffic handling
- Missing traffic telemetry
- SQLite database persistence & retrieval
- All 8 predefined simulation scenarios
"""

import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from module3_mitigation.mitigation_engine import MitigationEngine
from module3_mitigation.mitigation_policies import MitigationPolicyEngine
from module3_mitigation.simulated_network_controller import SimulatedNetworkController
from module3_mitigation.traffic_simulator import TrafficSimulator
from module3_mitigation.mitigation_verifier import MitigationVerifier
from module3_mitigation.database import MitigationDatabase
from module3_mitigation.scenarios import SCENARIOS_CATALOG, get_scenario
from module3_mitigation.models import (
    AttackEvent,
    ExecutionStatus,
    VerificationStatus,
)


class TestMitigationEngine(unittest.TestCase):
    """Test suite for complete Mitigation Engine workflows."""

    @classmethod
    def setUpClass(cls):
        # Create a dedicated temporary directory for test database and logs
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_db_path = cls.temp_dir / "test_mitigation.db"
        cls.test_log_dir = cls.temp_dir / "logs"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.db = MitigationDatabase(db_path=self.test_db_path)
        self.db.clear_history()
        self.engine = MitigationEngine(
            database=self.db,
            log_dir=self.test_log_dir,
        )

    def test_01_all_13_awid3_categories_supported(self):
        """Tests that all 13 AWID3 categories match valid policies."""
        expected_attacks = [
            "Deauth",
            "Disas",
            "(Re)Assoc",
            "Rogue_AP",
            "Krack",
            "Kr00k",
            "SSH",
            "Botnet",
            "Malware",
            "SQL_Injection",
            "SSDP",
            "Evil_Twin",
            "Website_spoofing",
        ]
        policy_engine = self.engine.policy_engine
        for attack_name in expected_attacks:
            event = AttackEvent(attack_type=attack_name, confidence=0.95)
            policy, is_low, _ = policy_engine.select_policy(event)
            self.assertNotEqual(
                policy.attack_type,
                "Unknown",
                f"Attack category '{attack_name}' should have a specific policy",
            )
            self.assertFalse(is_low)

    def test_02_ciciot2023_aliases_resolved(self):
        """Tests that common CICIoT2023 classes map cleanly to mitigation policies."""
        alias_tests = {
            "SqlInjection": "SQL_Injection",
            "Backdoor_Malware": "Malware",
            "DictionaryBruteForce": "SSH",
            "MITM-ArpSpoofing": "Evil_Twin",
            "DNS_Spoofing": "Website_spoofing",
            "Mirai-udpplain": "Botnet",
            "DDoS-UDP_Flood": "Botnet",
        }
        for ciciot_name, expected_canonical in alias_tests.items():
            event = AttackEvent(attack_type=ciciot_name, confidence=0.90)
            policy, _, _ = self.engine.policy_engine.select_policy(event)
            self.assertEqual(
                policy.attack_type,
                expected_canonical,
                f"CICIoT class '{ciciot_name}' should map to '{expected_canonical}'",
            )

    def test_03_low_confidence_requires_admin_approval(self):
        """Tests that prediction confidence below threshold requires admin approval."""
        event = AttackEvent(
            attack_type="SQL_Injection",
            confidence=0.42,  # Threshold is 0.60
            source_ip="192.168.1.50",
        )
        record = self.engine.process_attack(event)
        self.assertEqual(record.execution_status, ExecutionStatus.REQUIRES_ADMIN_APPROVAL)
        self.assertIn("LOW CONFIDENCE", record.action_description)

    def test_04_unknown_attack_category(self):
        """Tests that completely unknown attack category triggers fallback policy."""
        event = AttackEvent(
            attack_type="Unknown_Super_Exploit_9000",
            confidence=0.88,
        )
        record = self.engine.process_attack(event)
        self.assertEqual(record.execution_status, ExecutionStatus.UNSUPPORTED_ACTION)
        self.assertEqual(record.mitigation_action, "ALERT_ADMIN_AND_LOG")

    def test_05_database_persistence_and_retrieval(self):
        """Tests that mitigation records are properly stored and retrieved from SQLite."""
        event = AttackEvent(
            attack_type="Deauth",
            confidence=0.95,
            source_mac="00:11:22:33:44:55",
        )
        record = self.engine.process_attack(
            event, traffic_before=500.0, traffic_after=20.0
        )
        self.assertIsNotNone(record.id)

        # Retrieve record by ID
        saved = self.db.get_record_by_id(record.id)
        self.assertIsNotNone(saved)
        self.assertEqual(saved["attack_type"], "Deauth")
        self.assertEqual(saved["execution_status"], ExecutionStatus.SIMULATED_SUCCESS.value)
        self.assertEqual(saved["verification_status"], VerificationStatus.MITIGATION_EFFECTIVE.value)
        self.assertEqual(saved["reduction_percentage"], 96.0)

        # Retrieve recent records
        recent = self.db.get_recent_records(limit=10)
        self.assertEqual(len(recent), 1)

        # Summary stats
        stats = self.db.get_summary_stats()
        self.assertEqual(stats["total_events"], 1)
        self.assertEqual(stats["effective_count"], 1)

    def test_06_run_all_8_scenarios(self):
        """Runs and validates every one of the 8 required simulation scenarios."""
        for sc_id in range(1, 9):
            sc = get_scenario(sc_id)
            record = self.engine.run_scenario(sc_id)

            self.assertEqual(
                record.execution_status,
                sc["expected_execution_status"],
                f"Scenario {sc_id} ({sc['name']}) execution status mismatch",
            )
            self.assertEqual(
                record.verification_status,
                sc["expected_verification_status"],
                f"Scenario {sc_id} ({sc['name']}) verification status mismatch",
            )
            self.assertAlmostEqual(
                record.reduction_percentage,
                sc["expected_reduction_pct"],
                places=1,
                msg=f"Scenario {sc_id} reduction mismatch",
            )

    def test_07_zero_traffic_and_missing_telemetry(self):
        """Tests edge cases: zero traffic and missing measurements in the engine."""
        event = AttackEvent(attack_type="Deauth", confidence=0.95)

        # Zero traffic
        rec_zero = self.engine.process_attack(event, traffic_before=0.0, traffic_after=0.0)
        self.assertEqual(rec_zero.verification_status, VerificationStatus.INSUFFICIENT_DATA)

        # Missing measurement
        rec_missing = self.engine.process_attack(event, traffic_before=500.0, traffic_after=None)
        # Without explicit after traffic, traffic_simulator generates measurements
        self.assertGreater(rec_missing.traffic_before, 0.0)


if __name__ == "__main__":
    unittest.main()

