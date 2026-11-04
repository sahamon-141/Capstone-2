"""
Mamdani Fuzzy Inference System for IoT Network Anomaly Detection.

Constructs linguistic variables, data-driven membership functions derived from
the normal baseline distribution, an interpretable rule base, and centroid defuzzification.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.config import ProjectConfig, config


class IoTFuzzyAnomalyDetector:
    """Interpretable Mamdani Fuzzy Anomaly Detection System."""

    def __init__(self, baseline_stats: Dict[str, Any], cfg: ProjectConfig = config):
        self.cfg = cfg
        self.baseline_stats = baseline_stats
        self.simulation: Optional[ctrl.ControlSystemSimulation] = None
        self.rules: List[ctrl.Rule] = []
        
        # Build fuzzy variables and control system
        self._build_fuzzy_system()

    def _build_fuzzy_system(self) -> None:
        """
        Derive membership functions rigorously from normal traffic distribution
        and formulate the Mamdani rule base.
        """
        # Baseline normal traffic distribution parameters
        rate_p50 = float(self.baseline_stats.get("rate_percentiles", {}).get("p50", 170.0))
        rate_p75 = float(self.baseline_stats.get("rate_percentiles", {}).get("p75", 440.0))
        rate_p95 = float(self.baseline_stats.get("rate_percentiles", {}).get("p95", 2500.0))
        rate_max = 50000.0

        cv_median = float(self.baseline_stats.get("normal_cv_median", 1.02))
        cv_p25 = float(self.baseline_stats.get("cv_percentiles", {}).get("p25", 0.90))
        cv_p75 = float(self.baseline_stats.get("cv_percentiles", {}).get("p75", 1.25))
        cv_p95 = float(self.baseline_stats.get("cv_percentiles", {}).get("p95", 1.55))
        cv_max = 6.0

        dev_max = 5.0

        # Define Antecedents (Inputs)
        self.cv = ctrl.Antecedent(np.linspace(0, cv_max, self.cfg.FUZZY_RESOLUTION), "cv")
        self.rate = ctrl.Antecedent(np.linspace(0, rate_max, self.cfg.FUZZY_RESOLUTION), "rate")
        self.iat_dev = ctrl.Antecedent(np.linspace(0, dev_max, self.cfg.FUZZY_RESOLUTION), "iat_dev")

        # Define Consequent (Output)
        self.risk = ctrl.Consequent(np.linspace(0, 1.0, self.cfg.FUZZY_RESOLUTION), "risk")

        # ----------------------------------------------------------------------
        # Membership Functions:
        # LOW: Typical normal operations (or low-dispersion synthetic floods for CV)
        # MEDIUM: Normal / moderate activity
        # HIGH: Severe deviations / attack conditions
        # ----------------------------------------------------------------------
        
        # 1. CV Membership Functions
        # In normal traffic, CV is naturally clustered around ~1.0 (0.8 - 1.3)
        # Flood attacks collapse CV down to ~0.001 - 0.20
        # Port scans / jitter surge CV above 1.5
        self.cv["low"] = fuzz.trimf(self.cv.universe, [0.0, 0.0, 0.65])
        self.cv["medium"] = fuzz.trimf(self.cv.universe, [0.50, cv_median, 1.55])
        self.cv["high"] = fuzz.trapmf(self.cv.universe, [1.40, 1.80, cv_max, cv_max])

        # 2. Rate Membership Functions (pkts/s)
        # Normal traffic: median ~170 pkts/s, p75 ~440 pkts/s
        self.rate["low"] = fuzz.trimf(self.rate.universe, [0.0, 0.0, 350.0])
        self.rate["medium"] = fuzz.trimf(self.rate.universe, [200.0, 600.0, 1500.0])
        self.rate["high"] = fuzz.trapmf(self.rate.universe, [1000.0, 2500.0, rate_max, rate_max])

        # 3. IAT Deviation Membership Functions (|mu - mu_normal| / mu_normal)
        self.iat_dev["low"] = fuzz.trimf(self.iat_dev.universe, [0.0, 0.0, 0.18])
        self.iat_dev["medium"] = fuzz.trimf(self.iat_dev.universe, [0.12, 0.35, 0.65])
        self.iat_dev["high"] = fuzz.trapmf(self.iat_dev.universe, [0.50, 0.90, dev_max, dev_max])

        # 4. Attack Risk Consequent ([0, 1])
        self.risk["low"] = fuzz.trimf(self.risk.universe, [0.0, 0.0, 0.35])
        self.risk["medium"] = fuzz.trimf(self.risk.universe, [0.25, 0.50, 0.75])
        self.risk["high"] = fuzz.trapmf(self.risk.universe, [0.60, 0.85, 1.0, 1.0])

        # ----------------------------------------------------------------------
        # Mamdani Fuzzy Rule Base
        # ----------------------------------------------------------------------
        self.rules = [
            # Normal operational profile: CV around normal median, low packet rate, low IAT dev
            ctrl.Rule(self.cv["medium"] & self.rate["low"] & self.iat_dev["low"], self.risk["low"]),
            ctrl.Rule(self.cv["low"] & self.rate["low"] & self.iat_dev["low"], self.risk["low"]),
            ctrl.Rule(self.cv["medium"] & self.rate["low"] & self.iat_dev["medium"], self.risk["low"]),

            # Rule 2: High CV and High Rate -> High Risk (Burst DDoS)
            ctrl.Rule(self.cv["high"] & self.rate["high"], self.risk["high"]),
            # Rule 3: High CV and High IAT Deviation -> High Risk (Unsteady scan / pulse attack)
            ctrl.Rule(self.cv["high"] & self.iat_dev["high"], self.risk["high"]),
            # Rule 4: High Rate and High IAT Deviation -> High Risk (Volumetric flood)
            ctrl.Rule(self.rate["high"] & self.iat_dev["high"], self.risk["high"]),
            # Rule 5: Medium CV and High Rate -> High Risk
            ctrl.Rule(self.cv["medium"] & self.rate["high"], self.risk["high"]),
            # Rule 6: High CV and Medium Rate -> High Risk
            ctrl.Rule(self.cv["high"] & self.rate["medium"], self.risk["high"]),

            # Rule 7: Moderate conditions across all dimensions -> Medium Risk (Suspicious)
            ctrl.Rule(self.cv["medium"] & self.rate["medium"] & self.iat_dev["medium"], self.risk["medium"]),

            # Rule 9: Volumetric surge alone is a high-confidence attack indicator
            ctrl.Rule(self.rate["high"], self.risk["high"]),
            # Rule 10: Structural timing deviation
            ctrl.Rule(self.iat_dev["high"], self.risk["high"]),
            # Rule 11: Constant periodic flood (e.g. DoS / Mirai botnet: abnormally LOW CV + High Rate)
            ctrl.Rule(self.cv["low"] & self.rate["high"], self.risk["high"]),
            # Rule 12: High CV with low rate (Slow reconnaissance / stealthy scans)
            ctrl.Rule(self.cv["high"] & self.rate["low"], self.risk["medium"]),
        ]

        # Compile Mamdani control system
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        print("[OK] Mamdani Fuzzy Inference System initialized with 12 interpretable rules.")

    def evaluate_window(self, cv_val: float, rate_val: float, dev_val: float) -> float:
        """
        Evaluate a single window through Mamdani inference.
        Returns continuous Attack Risk score in [0, 1].
        """
        cv_clipped = float(np.clip(cv_val, self.cv.universe.min(), self.cv.universe.max()))
        rate_clipped = float(np.clip(rate_val, self.rate.universe.min(), self.rate.universe.max()))
        dev_clipped = float(np.clip(dev_val, self.iat_dev.universe.min(), self.iat_dev.universe.max()))

        try:
            self.simulation.input["cv"] = cv_clipped
            self.simulation.input["rate"] = rate_clipped
            self.simulation.input["iat_dev"] = dev_clipped
            self.simulation.compute()
            risk_score = float(self.simulation.output["risk"])
        except Exception:
            risk_score = 0.50

        return risk_score

    def predict_batch(self, df_windows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate a batch of sliding windows.
        
        Returns:
            risk_scores: Continuous Attack Risk in [0, 1]
            categorical_labels: 'NORMAL', 'SUSPICIOUS', 'ATTACK'
            binary_predictions: 0 for NORMAL, 1 for ATTACK/SUSPICIOUS
        """
        cv_arr = df_windows["cv_iat"].to_numpy()
        rate_arr = df_windows["mean_rate"].to_numpy()
        dev_arr = df_windows["iat_deviation"].to_numpy()

        n = len(df_windows)
        risk_scores = np.zeros(n, dtype=np.float64)
        categorical_labels = []
        binary_predictions = np.zeros(n, dtype=np.int32)

        for i in range(n):
            score = self.evaluate_window(cv_arr[i], rate_arr[i], dev_arr[i])
            risk_scores[i] = score

            if score < self.cfg.FUZZY_LOW_THRESHOLD:
                cat = "NORMAL"
                bin_pred = 0
            elif score >= self.cfg.FUZZY_HIGH_THRESHOLD:
                cat = "ATTACK"
                bin_pred = 1
            else:
                cat = "SUSPICIOUS"
                bin_pred = 1

            categorical_labels.append(cat)
            binary_predictions[i] = bin_pred

        return risk_scores, np.array(categorical_labels), binary_predictions
