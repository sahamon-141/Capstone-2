"""
End-to-end tests for the three-module pipeline (ids/).

Requires the trained artefacts:  python -m ids.prepare_data && python -m ids.train_module2
Run:  python -m pytest tests/test_pipeline.py -q
"""

import numpy as np
import pytest

from ids.config import NORMAL_LABEL, RISK_SUSPICIOUS
from ids.module1_detection import FuzzyAnomalyDetector


@pytest.fixture(scope="module")
def P():
    from ids.pipeline import IDSPipeline
    return IDSPipeline()


def test_fuzzy_matches_skfuzzy_reference():
    """The NumPy Mamdani engine (timing mode) reproduces src/fuzzy_system.py (skfuzzy)."""
    from src.fuzzy_system import IoTFuzzyAnomalyDetector
    ref = IoTFuzzyAnomalyDetector({"normal_cv_median": 1.0})
    ours = FuzzyAnomalyDetector({"mu_normal": 1.0, "cv_median": 1.0})
    for cv, rate, dev in [(1.0, 50, 0.05), (0.001, 3000, 0.01), (2.5, 100, 1.2), (1.0, 800, 0.3)]:
        assert abs(ref.evaluate_window(cv, rate, dev) - ours.infer(cv, rate, dev).risk) < 0.02


def test_benign_window_mostly_stays_in_module1(P):
    normal = sum(P.run(P.make_window(NORMAL_LABEL, seed=s)).m1.verdict == "NORMAL" for s in range(20))
    assert normal >= 16


def test_flood_is_detected_classified_and_mitigated(P):
    res = P.run(P.make_window("DDoS-SYN_Flood", 0.8, seed=1))
    assert res.m1.forward_to_module2
    assert res.m2.family in ("DDoS", "DoS")
    assert res.m3 is not None and res.m3.steps
    assert res.m3.steps[-1].risk_after < res.m3.steps[0].risk_before


def test_mirai_is_unknown_to_module2(P):
    hits = 0
    for s in range(10):
        r = P.run(P.make_window("Mirai-greeth_flood", 0.8, seed=s))
        hits += bool(r.m2 and r.m2.is_zero_day)
    assert hits >= 7


def test_mitigation_does_not_use_ground_truth(P):
    w = P.make_window("DoS-UDP_Flood", 1.0, seed=3)
    plan = P.m3.plan("DoS", "DoS-UDP_Flood", 0.9)
    # nothing targeted -> nothing filtered, whatever the labels say
    out = P.m3.execute_and_verify(plan, w, np.zeros(len(w), dtype=bool))
    assert out.steps[0].attack_flows_after == out.steps[0].attack_flows_before
    assert out.steps[0].risk_after >= RISK_SUSPICIOUS
