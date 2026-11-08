"""
End-to-end pipeline:  traffic window -> Module 1 -> (if flagged) Module 2 -> Module 3.

The demo traffic comes from the held-out CICIoT2023 test sample (data/processed/test_sample.parquet),
so neither module has seen these flows during training.
"""

import json
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ids.config import (FAMILY_OF, MODEL_DIR, METRICS_DIR, NORMAL_LABEL, TEST_SAMPLE,
                        WINDOW_SIZE)
from ids.module1_detection import FuzzyAnomalyDetector, Module1Result
from ids.module2_classification import AttackClassifier, Module2Result
from ids.module3_mitigation import MitigationEngine, MitigationPlan


@dataclass
class PipelineResult:
    window: pd.DataFrame
    true_label: str
    m1: Module1Result
    m2: Module2Result | None = None
    m3: MitigationPlan | None = None


class IDSPipeline:
    def __init__(self):
        baseline = json.loads((MODEL_DIR / "module1_baseline.json").read_text())
        self.test = pd.read_parquet(TEST_SAMPLE)
        self.benign_pool = self.test[self.test["label"] == NORMAL_LABEL]
        with open(MODEL_DIR / "module1_behaviour.pkl", "rb") as fh:
            behaviour = pickle.load(fh)
        self.m1 = FuzzyAnomalyDetector(baseline, behaviour, mode="full")
        self.m1_timing = FuzzyAnomalyDetector(baseline, behaviour, mode="timing")   # paper's IAT-only variant
        self.m2 = AttackClassifier()
        self.m3 = MitigationEngine(self.m1, self.benign_pool)
        self.metrics = json.loads((METRICS_DIR / "metrics.json").read_text())
        self.classes = sorted(self.test["label"].unique())

    # -------------------------------------------------------------------------
    def make_window(self, attack_class: str, attack_share: float = 0.8, seed: int | None = None) -> pd.DataFrame:
        """A window of 20 flows: `attack_share` of them from attack_class, the rest benign."""
        rng = np.random.default_rng(seed)
        if attack_class == NORMAL_LABEL:
            n_att = 0
        else:
            n_att = max(1, int(round(attack_share * WINDOW_SIZE)))
        pool = self.test[self.test["label"] == attack_class]
        att = pool.sample(n_att, replace=len(pool) < n_att, random_state=int(rng.integers(1e9)))
        ben = self.benign_pool.sample(WINDOW_SIZE - n_att, random_state=int(rng.integers(1e9)))
        return pd.concat([att, ben]).sample(frac=1.0, random_state=int(rng.integers(1e9))).reset_index(drop=True)

    @staticmethod
    def true_label_of(window: pd.DataFrame) -> str:
        attacks = window.loc[window["label"] != NORMAL_LABEL, "label"]
        return attacks.mode().iloc[0] if len(attacks) >= len(window) / 2 else NORMAL_LABEL

    # -------------------------------------------------------------------------
    def run(self, window: pd.DataFrame, model: str = "rf", mitigate: bool = True,
            m1_mode: str = "full") -> PipelineResult:
        det = self.m1 if m1_mode == "full" else self.m1_timing
        res = PipelineResult(window=window, true_label=self.true_label_of(window),
                             m1=det.analyse_window(window))
        if not res.m1.forward_to_module2:
            return res          # NORMAL: Module 2 and 3 stay idle (saves CPU on the IoT gateway)
        res.m2 = self.m2.classify_window(window, model=model)
        if res.m2.family == "Benign":
            return res          # Module 2 overrules a Module 1 false alarm: nothing to mitigate
        plan = self.m3.plan(res.m2.family, res.m2.attack_class, res.m2.confidence)
        if mitigate:
            plan = self.m3.execute_and_verify(plan, window, res.m2.attack_flow_mask)
        res.m3 = plan
        return res

    # -------------------------------------------------------------------------
    def stream(self, phases: list[tuple[str, int]], attack_share: float = 0.8, seed: int = 0):
        """phases: [(class, n_windows), ...]  ->  list of windows for the stream monitor."""
        rng = np.random.default_rng(seed)
        out = []
        for cls, n in phases:
            for _ in range(n):
                out.append(self.make_window(cls, attack_share, seed=int(rng.integers(1e9))))
        return out

