"""
End-to-End Integration Test for Module 3.

Validates the full workflow:
1. Simulated attack input ->
2. Module 2 prediction output format (from PipelineIntegrator) ->
3. Module 3 Policy Engine selection ->
4. Simulated Network Controller execution ->
5. Mitigation Verification (% reduction calculation) ->
6. SQLite database record persistence ->
7. Dashboard retrieval format check.
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
from module3_mitigation.database import MitigationDatabase
from module3_mitigation.models import ExecutionStatus, VerificationStatus


class TestModule3EndToEnd(unittest.TestCase):
    """End-to-End pipeline verification test suite."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_db_path = cls.temp_dir / "e2e_mitigation.db"
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

    def test_e2e_pipeline_integration(self):
        """
        Executes end-to-end integration:
        PipelineIntegrator format -> Module 3 Engine -> SQLite Database -> Dashboard.
        """
        # 1. Simulate exact output format from PipelineIntegrator / Module 2
        module2_prediction_output = {
            "window_id": 42,
            "module1_verdict": "ATTACK",
            "attack_risk": 0.892,
            "module2_invoked": True,
            "predicted_attack": "DDoS-UDP_Flood",  # Maps to Botnet policy
            "raw_class": "DDoS-UDP_Flood",
            "class_id": 14,
            "confidence": 0.9654,
            "is_zero_day": False,
            "module2_runtime": "INT8_TFLite",
            "module2_inference_latency_ms": 0.087,
            "total_pipeline_latency_ms": 1.250,
        }

        # 2. Hand off to Module 3 Mitigation Engine
        record = self.engine.mitigate_from_pipeline(
            pipeline_output=module2_prediction_output,
            source_mac="E4:5F:01:FE:DC:BA",
            source_ip="192.168.1.185",
            destination_ip="192.168.1.1",
            sensor_id="ESP32_01",
        )

        # 3. Verify Module 3 correctly selected policy and executed simulated mitigation
        self.assertIsNotNone(record)
        self.assertEqual(record.attack_type, "DDoS-UDP_Flood")
        # A UDP flood is a volumetric attack, usually from spoofed sources, so
        # the correct response is a per-source rate limit, not isolating a device.
        self.assertEqual(record.mitigation_action, "RATE_LIMIT_SOURCE")
        self.assertEqual(record.execution_status, ExecutionStatus.SIMULATED_SUCCESS)
        self.assertTrue(record.simulation_mode)

        # 4. Verify Traffic Verification
        # A rate limit caps the source rather than removing all of its traffic,
        # so some residual is expected and the reduction is partial, not total.
        self.assertGreater(record.traffic_before, 0.0)
        self.assertLess(record.traffic_after, record.traffic_before)
        self.assertGreater(record.reduction_percentage, 50.0)
        self.assertEqual(record.verification_status, VerificationStatus.PARTIALLY_EFFECTIVE)

        # 5. Verify Database Record Persistence
        db_records = self.db.get_recent_records(limit=10)
        self.assertEqual(len(db_records), 1)
        row = db_records[0]
        self.assertEqual(row["attack_type"], "DDoS-UDP_Flood")
        self.assertEqual(row["source_mac"], "E4:5F:01:FE:DC:BA")
        self.assertEqual(row["mitigation_action"], "RATE_LIMIT_SOURCE")
        self.assertEqual(row["execution_status"], "SIMULATED_SUCCESS")
        self.assertEqual(row["verification_status"], "PARTIALLY_EFFECTIVE")
        self.assertGreater(row["reduction_percentage"], 50.0)

        # 6. Verify Dashboard Data Readiness (dict representation)
        record_dict = record.to_dict()
        self.assertIn("execution_status", record_dict)
        self.assertIn("verification_status", record_dict)
        self.assertIn("reduction_percentage", record_dict)
        self.assertEqual(record_dict["execution_status"], "SIMULATED_SUCCESS")
        self.assertEqual(record_dict["verification_status"], "PARTIALLY_EFFECTIVE")

    def test_e2e_normal_traffic_bypasses_mitigation(self):
        """Verifies that normal traffic (Module 2 not invoked) does not trigger mitigation."""
        normal_pipeline_output = {
            "window_id": 1,
            "module1_verdict": "NORMAL",
            "attack_risk": 0.12,
            "module2_invoked": False,
            "predicted_attack": "None (Traffic is Normal)",
            "confidence": 0.88,
            "is_zero_day": False,
        }
        res = self.engine.mitigate_from_pipeline(normal_pipeline_output)
        self.assertIsNone(res, "Normal traffic should not trigger mitigation")


if __name__ == "__main__":
    unittest.main()

