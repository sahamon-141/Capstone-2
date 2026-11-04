"""
Zero-Day / Unseen Attack Experimentation module.

Strictly partitions data to guarantee zero target leakage:
The held-out attack family is completely isolated from baseline computation,
fuzzy membership derivation, rule design, and parameter tuning.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

from src.config import ProjectConfig, config
from src.baseline import NormalBaselineBuilder, AdaptiveThresholdDetector
from src.fuzzy_system import IoTFuzzyAnomalyDetector


class ZeroDayExperiment:
    """
    Orchestrates the unseen-attack evaluation protocol.
    
    Scientific Note:
    In benchmark evaluations, withholding an attack family (e.g. Mirai botnet or Recon scans)
    serves as a controlled experimental proxy for 'zero-day' novel attacks. We evaluate whether
    the unsupervised statistical-fuzzy detector flags the unseen behavior purely from its
    deviation from normal traffic dynamics, without prior attack signatures.
    """

    def __init__(
        self,
        held_out_attacks: Optional[List[str]] = None,
        cfg: ProjectConfig = config
    ):
        self.cfg = cfg
        self.held_out_attacks = held_out_attacks or self.cfg.HELD_OUT_ATTACKS

    def partition_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Partition dataset into three research cohorts:
        1. Normal Traffic: Pure BenignTraffic for evaluation
        2. Known Attacks: Attacks used/allowed during dev/validation
        3. Held-out Attacks: Zero-Day proxy (strictly held out until test)
        """
        normal_mask = df["original_label"] == self.cfg.NORMAL_LABEL
        held_out_mask = df["original_label"].isin(self.held_out_attacks)
        known_attack_mask = (~normal_mask) & (~held_out_mask)

        df_normal = df[normal_mask].copy()
        df_known = df[known_attack_mask].copy()
        df_held_out = df[held_out_mask].copy()

        print("\n" + "=" * 60)
        print("STEP 10: ZERO-DAY / UNSEEN ATTACK COHORT PARTITIONING")
        print("=" * 60)
        print(f"Held-out Unseen Attacks: {self.held_out_attacks}")
        print(f"  - Normal evaluation cohort:   {len(df_normal):,} flows")
        print(f"  - Known attacks cohort:       {len(df_known):,} flows")
        print(f"  - Held-out / Unseen cohort:   {len(df_held_out):,} flows")
        print("=" * 60)

        return df_normal, df_known, df_held_out

    def evaluate_cohort(
        self,
        cohort_name: str,
        df_windows: pd.DataFrame,
        simple_detector: AdaptiveThresholdDetector,
        adaptive_detector: AdaptiveThresholdDetector,
        fuzzy_detector: IoTFuzzyAnomalyDetector,
    ) -> Dict[str, Any]:
        """
        Evaluate all three detection methods on a specific traffic cohort.
        """
        if len(df_windows) == 0:
            print(f"[Warning] Cohort {cohort_name} has 0 windows.")
            return {"cohort": cohort_name, "num_windows": 0}

        # Method 1: Simple CV
        pred_simple, _ = simple_detector.predict_simple_cv(df_windows)

        # Method 3: Fuzzy Logic
        risk_scores, categories, pred_fuzzy = fuzzy_detector.predict_batch(df_windows)

        # Method 2: Adaptive Median + MAD (with anti-poisoning guided by fuzzy risk)
        res_adaptive = adaptive_detector.run_adaptive_detection(
            df_windows,
            confidence_risk_scores=risk_scores
        )
        pred_adaptive = res_adaptive["predictions"]

        ground_truth = df_windows["binary_label"].to_numpy()

        results = {
            "cohort": cohort_name,
            "num_windows": len(df_windows),
            "ground_truth": ground_truth,
            "mean_attack_risk": float(np.mean(risk_scores)),
            "std_attack_risk": float(np.std(risk_scores)),
            "pred_simple": pred_simple,
            "pred_adaptive": pred_adaptive,
            "pred_fuzzy": pred_fuzzy,
            "fuzzy_risk_scores": risk_scores,
            "fuzzy_categories": categories,
            "adaptive_thresholds": res_adaptive["adaptive_thresholds"],
            # Detection rates (proportion classified as attack/anomaly)
            "detection_rate_simple": float(np.mean(pred_simple)),
            "detection_rate_adaptive": float(np.mean(pred_adaptive)),
            "detection_rate_fuzzy": float(np.mean(pred_fuzzy)),
        }

        print(f"\nCohort Results [{cohort_name}] (n={len(df_windows):,} windows):")
        print(f"  Mean Fuzzy Attack Risk: {results['mean_attack_risk']:.4f}")
        print(f"  Method 1 (Simple CV) Alarm Rate:     {results['detection_rate_simple'] * 100:.2f}%")
        print(f"  Method 2 (Adaptive MAD) Alarm Rate:  {results['detection_rate_adaptive'] * 100:.2f}%")
        print(f"  Method 3 (Fuzzy Logic) Alarm Rate:   {results['detection_rate_fuzzy'] * 100:.2f}%")

        return results
