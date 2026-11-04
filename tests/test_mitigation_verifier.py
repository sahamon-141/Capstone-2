"""
Unit tests for Module 3 Mitigation Verification Engine (mitigation_verifier.py).

Covers:
- Standard percentage reduction calculations
- Status thresholds (MITIGATION_EFFECTIVE, PARTIALLY_EFFECTIVE, MITIGATION_FAILED)
- Zero baseline traffic handling
- Missing telemetry / None handling
- Traffic surge / negative reduction handling
- Equal observation window handling
- Legitimate traffic impact preservation
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from module3_mitigation.mitigation_verifier import MitigationVerifier
from module3_mitigation.models import VerificationStatus


class TestMitigationVerifier(unittest.TestCase):
    """Test suite for mathematical verification logic."""

    def setUp(self):
        self.verifier = MitigationVerifier(
            effective_threshold_pct=80.0,
            partial_threshold_pct=30.0,
            default_window_seconds=60,
        )

    def test_01_effective_mitigation(self):
        """Tests that reduction >= 80% yields MITIGATION_EFFECTIVE."""
        # 500 -> 40 pkts/min = 92% reduction
        res = self.verifier.verify(traffic_before=500.0, traffic_after=40.0)
        self.assertEqual(res.status, VerificationStatus.MITIGATION_EFFECTIVE)
        self.assertAlmostEqual(res.reduction_percentage, 92.0, places=1)
        self.assertEqual(res.traffic_before, 500.0)
        self.assertEqual(res.traffic_after, 40.0)
        self.assertTrue(res.is_simulation)
        self.assertIn("[SIMULATED]", res.message)

    def test_02_partially_effective_mitigation(self):
        """Tests that 30% <= reduction < 80% yields PARTIALLY_EFFECTIVE."""
        # 400 -> 180 pkts/min = 55% reduction
        res = self.verifier.verify(traffic_before=400.0, traffic_after=180.0)
        self.assertEqual(res.status, VerificationStatus.PARTIALLY_EFFECTIVE)
        self.assertAlmostEqual(res.reduction_percentage, 55.0, places=1)

    def test_03_failed_mitigation(self):
        """Tests that reduction < 30% yields MITIGATION_FAILED."""
        # 1000 -> 950 pkts/min = 5% reduction
        res = self.verifier.verify(traffic_before=1000.0, traffic_after=950.0)
        self.assertEqual(res.status, VerificationStatus.MITIGATION_FAILED)
        self.assertAlmostEqual(res.reduction_percentage, 5.0, places=1)

    def test_04_negative_reduction_traffic_surge(self):
        """Tests that traffic increase yields negative reduction and MITIGATION_FAILED."""
        # 300 -> 450 pkts/min = -50% reduction
        res = self.verifier.verify(traffic_before=300.0, traffic_after=450.0)
        self.assertEqual(res.status, VerificationStatus.MITIGATION_FAILED)
        self.assertEqual(res.reduction_percentage, -50.0)

    def test_05_zero_baseline_traffic(self):
        """Tests that zero baseline traffic produces INSUFFICIENT_DATA."""
        res = self.verifier.verify(traffic_before=0.0, traffic_after=0.0)
        self.assertEqual(res.status, VerificationStatus.INSUFFICIENT_DATA)
        self.assertEqual(res.reduction_percentage, 0.0)
        self.assertIn("zero", res.message.lower())

    def test_06_missing_traffic_measurements(self):
        """Tests that None in before or after returns INSUFFICIENT_DATA."""
        res1 = self.verifier.verify(traffic_before=None, traffic_after=50.0)
        self.assertEqual(res1.status, VerificationStatus.INSUFFICIENT_DATA)

        res2 = self.verifier.verify(traffic_before=500.0, traffic_after=None)
        self.assertEqual(res2.status, VerificationStatus.INSUFFICIENT_DATA)

    def test_07_verification_pending(self):
        """Tests that pending flag yields VERIFICATION_PENDING."""
        res = self.verifier.verify(
            traffic_before=500.0, traffic_after=0.0, is_pending=True
        )
        self.assertEqual(res.status, VerificationStatus.VERIFICATION_PENDING)
        self.assertIn("pending", res.status.value.lower())

    def test_08_legitimate_traffic_impact_preserved(self):
        """Tests that legitimate traffic impact parameter is propagated."""
        res = self.verifier.verify(
            traffic_before=1000.0, traffic_after=0.0, legitimate_traffic_impact=100.0
        )
        self.assertEqual(res.legitimate_traffic_impact, 100.0)
        self.assertEqual(res.status, VerificationStatus.MITIGATION_EFFECTIVE)


if __name__ == "__main__":
    unittest.main()

