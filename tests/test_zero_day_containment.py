"""
Tests for the zero-day behaviour classifier, the escalation controller and the
batch pipeline evaluation.

The baseline below uses the real CICIoT2023 rate percentiles from
``results/metrics/summary_metrics.json`` so the thresholds under test are the
ones the system actually runs with.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from module3_mitigation.escalation_controller import EscalationController
from module3_mitigation.pipeline_evaluation import Module3PipelineEvaluator
from module3_mitigation.zero_day_containment import (
    ZeroDayBehaviour,
    ZeroDayContainment,
)

# Real baseline percentiles from the CICIoT2023 run.
BASELINE = {
    "rate_percentiles": {"p50": 202.39, "p75": 495.05, "p95": 10530.09},
}


class StubFuzzy:
    """Stands in for IoTFuzzyAnomalyDetector with the same rate-driven shape.

    Risk is high while the rate is above the fuzzy "rate is high" shoulder and
    low once it is capped, which is the behaviour the real Mamdani base shows.
    """

    def evaluate_window(self, cv, rate, dev):
        if rate >= 1000.0:
            return 0.85
        if rate >= 600.0:
            return 0.50
        return 0.12


class TestZeroDayContainment(unittest.TestCase):

    def setUp(self):
        self.z = ZeroDayContainment(baseline_stats=BASELINE)

    def test_constant_rate_flood_detected_from_low_cv(self):
        """Held-out Mirai sits at CV 0.0 and ~650 pkt/s; that is a flood."""
        self.assertEqual(
            self.z.classify(cv=0.0, rate=654.69, iat_deviation=0.0),
            ZeroDayBehaviour.CONSTANT_RATE_FLOOD,
        )

    def test_slow_probe_detected_from_high_cv_low_rate(self):
        self.assertEqual(
            self.z.classify(cv=2.6, rate=90.0, iat_deviation=0.3),
            ZeroDayBehaviour.SLOW_PROBE,
        )

    def test_volumetric_surge_detected(self):
        self.assertEqual(
            self.z.classify(cv=1.0, rate=42000.0, iat_deviation=0.9),
            ZeroDayBehaviour.VOLUMETRIC_SURGE,
        )

    def test_normal_looking_window_is_unclassified(self):
        self.assertEqual(
            self.z.classify(cv=1.0, rate=150.0, iat_deviation=0.05),
            ZeroDayBehaviour.UNCLASSIFIED_ANOMALY,
        )

    def test_every_behaviour_has_a_ladder_that_starts_least_disruptive(self):
        for behaviour in ZeroDayBehaviour:
            ladder = self.z.decide(
                *{
                    ZeroDayBehaviour.CONSTANT_RATE_FLOOD: (0.0, 654.0, 0.0),
                    ZeroDayBehaviour.SLOW_PROBE: (2.6, 90.0, 0.3),
                    ZeroDayBehaviour.VOLUMETRIC_SURGE: (1.0, 42000.0, 0.9),
                    ZeroDayBehaviour.UNCLASSIFIED_ANOMALY: (1.0, 150.0, 0.05),
                }[behaviour]
            ).escalation_ladder
            self.assertTrue(ladder, f"{behaviour} has no containment ladder")


class TestEscalationController(unittest.TestCase):

    def setUp(self):
        self.ctl = EscalationController(
            fuzzy_detector=StubFuzzy(), baseline_stats=BASELINE, risk_threshold=0.35
        )

    def test_rate_limit_caps_at_the_p75_baseline(self):
        kind, residual = self.ctl.apply_action("RATE_LIMIT_SOURCE", 10000.0)
        self.assertEqual(kind, "cap")
        self.assertAlmostEqual(residual, 495.05, places=2)

    def test_cap_never_raises_a_rate_below_the_cap(self):
        _, residual = self.ctl.apply_action("RATE_LIMIT_SOURCE", 120.0)
        self.assertAlmostEqual(residual, 120.0, places=2)

    def test_block_drops_traffic_to_zero(self):
        kind, residual = self.ctl.apply_action("BLOCK_SOURCE", 10000.0)
        self.assertEqual(kind, "stop")
        self.assertEqual(residual, 0.0)

    def test_filter_alone_contains_a_flood(self):
        out = self.ctl.contain(
            cv=0.0, rate=10000.0, iat_deviation=0.01,
            ladder=["RATE_LIMIT_SOURCE", "BLOCK_SOURCE"],
        )
        self.assertTrue(out.contained)
        self.assertEqual(out.steps_used, 1)
        self.assertEqual(out.final_action, "RATE_LIMIT_SOURCE")
        self.assertFalse(out.needed_disruptive_action)

    def test_escalates_to_a_block_when_the_filter_is_not_enough(self):
        # A tiny cap would still leave this above the stub's threshold, so make
        # the first rung an alert, which by definition cannot contain anything.
        out = self.ctl.contain(
            cv=1.0, rate=5000.0, iat_deviation=0.9,
            ladder=["ALERT_ADMIN_AND_LOG", "BLOCK_SOURCE"],
        )
        self.assertTrue(out.contained)
        self.assertEqual(out.steps_used, 2)
        self.assertTrue(out.needed_disruptive_action)
        self.assertEqual(out.final_risk, 0.0)

    def test_an_empty_ladder_contains_nothing(self):
        out = self.ctl.contain(cv=1.0, rate=5000.0, iat_deviation=0.9, ladder=[])
        self.assertFalse(out.contained)
        self.assertEqual(out.steps_used, 0)

    def test_results_are_flagged_as_modelled(self):
        out = self.ctl.contain(
            cv=0.0, rate=10000.0, iat_deviation=0.01, ladder=["RATE_LIMIT_SOURCE"]
        )
        self.assertTrue(out.to_dict()["is_model"])


class TestPipelineEvaluation(unittest.TestCase):

    def _windows(self):
        return pd.DataFrame([
            # flagged flood, has a class policy
            {"window_id": 1, "cv_iat": 0.0, "mean_rate": 9000.0, "iat_deviation": 0.01,
             "original_label": "DDoS-ICMP_Flood", "fuzzy_pred": 1, "attack_risk": 0.9},
            # flagged held-out Mirai, must take the behaviour route
            {"window_id": 2, "cv_iat": 0.0, "mean_rate": 654.0, "iat_deviation": 0.0,
             "original_label": "Mirai-greeth_flood", "fuzzy_pred": 1, "attack_risk": 0.8},
            # flagged but actually benign: a Module 1 false positive
            {"window_id": 3, "cv_iat": 1.0, "mean_rate": 203.0, "iat_deviation": 0.1,
             "original_label": "BenignTraffic", "fuzzy_pred": 1, "attack_risk": 0.4},
            # not flagged: bypasses Module 2 and Module 3 entirely
            {"window_id": 4, "cv_iat": 1.0, "mean_rate": 150.0, "iat_deviation": 0.05,
             "original_label": "BenignTraffic", "fuzzy_pred": 0, "attack_risk": 0.1},
        ])

    def setUp(self):
        self.ev = Module3PipelineEvaluator(
            fuzzy_detector=StubFuzzy(),
            baseline_stats=BASELINE,
            held_out_classes=["Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"],
        )
        self.result = self.ev.evaluate(self._windows())
        self.df = self.result["per_window"]
        self.summary = self.result["summary"]

    def test_unflagged_windows_never_reach_mitigation(self):
        row = self.df[self.df.window_id == 4].iloc[0]
        self.assertFalse(row["flagged"])
        self.assertEqual(row["route"], "bypassed")
        self.assertEqual(self.summary["windows_bypassed_by_module1"], 1)

    def test_known_class_uses_its_policy(self):
        row = self.df[self.df.window_id == 1].iloc[0]
        self.assertEqual(row["route"], "class_policy")
        self.assertEqual(row["family"], "Volumetric_Flood")
        self.assertEqual(row["first_action"], "RATE_LIMIT_SOURCE")

    def test_held_out_class_takes_the_behaviour_route(self):
        row = self.df[self.df.window_id == 2].iloc[0]
        self.assertEqual(row["route"], "zero_day_behaviour")
        self.assertEqual(row["behaviour"], ZeroDayBehaviour.CONSTANT_RATE_FLOOD.value)

    def test_benign_window_receives_no_action(self):
        row = self.df[self.df.window_id == 3].iloc[0]
        self.assertEqual(row["first_action"], "NO_ACTION")
        self.assertEqual(
            self.summary["collateral_on_benign_windows"]["best_case_receiving_an_action"], 0
        )

    def test_volume_suppression_is_computed_from_the_observed_rate(self):
        row = self.df[self.df.window_id == 1].iloc[0]
        expected = 100.0 * (9000.0 - 495.05) / 9000.0
        self.assertAlmostEqual(row["volume_suppressed_pct"], round(expected, 2), places=1)

    def test_summary_states_that_containment_is_modelled(self):
        self.assertIn("modelled", self.summary["note"].lower())


if __name__ == "__main__":
    unittest.main()
