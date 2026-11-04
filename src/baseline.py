"""
Normal Baseline and Adaptive Median + MAD Thresholding module.

Constructs normal traffic statistical baselines, prevents attack poisoning,
and implements Simple CV Threshold (Method 1) and Adaptive Median + MAD (Method 2).
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import ProjectConfig, config


class NormalBaselineBuilder:
    """Computes unpolluted normal traffic baseline statistics."""

    def __init__(self, cfg: ProjectConfig = config):
        self.cfg = cfg
        self.stats: Dict[str, Any] = {}

    def fit(self, normal_window_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit normal baseline using pure normal traffic windows.
        
        Args:
            normal_window_features: DataFrame of window features extracted from
                                    NORMAL (BenignTraffic) flows only.
        """
        if len(normal_window_features) == 0:
            raise ValueError("Cannot fit baseline with empty normal window features.")

        cv_vals = normal_window_features["cv_iat"].to_numpy()
        rate_vals = normal_window_features["mean_rate"].to_numpy()
        iat_vals = normal_window_features["mean_iat"].to_numpy()
        dur_vals = normal_window_features["mean_flow_duration"].to_numpy()

        # Median and MAD for CV
        cv_median = float(np.median(cv_vals))
        cv_mad = float(np.median(np.abs(cv_vals - cv_median)))
        if cv_mad < 1e-6:
            cv_mad = float(np.std(cv_vals)) if np.std(cv_vals) > 1e-6 else 1e-3

        # Rate statistics
        rate_median = float(np.median(rate_vals))
        rate_mean = float(np.mean(rate_vals))
        rate_std = float(np.std(rate_vals))

        # IAT statistics
        iat_mean = float(np.mean(iat_vals))
        iat_std = float(np.std(iat_vals))

        # Store baseline dictionary
        self.stats = {
            "normal_mean_iat": iat_mean,
            "normal_std_iat": iat_std,
            "normal_mean_rate": rate_mean,
            "normal_std_rate": rate_std,
            "normal_median_rate": rate_median,
            "normal_cv_median": cv_median,
            "normal_cv_mad": cv_mad,
            "normal_cv_mean": float(np.mean(cv_vals)),
            "normal_cv_std": float(np.std(cv_vals)),
            "cv_percentiles": {
                "p25": float(np.percentile(cv_vals, 25)),
                "p50": float(np.percentile(cv_vals, 50)),
                "p75": float(np.percentile(cv_vals, 75)),
                "p90": float(np.percentile(cv_vals, 90)),
                "p95": float(np.percentile(cv_vals, 95)),
                "p99": float(np.percentile(cv_vals, 99)),
                "min": float(np.min(cv_vals)),
                "max": float(np.max(cv_vals)),
            },
            "rate_percentiles": {
                "p25": float(np.percentile(rate_vals, 25)),
                "p50": float(np.percentile(rate_vals, 50)),
                "p75": float(np.percentile(rate_vals, 75)),
                "p90": float(np.percentile(rate_vals, 90)),
                "p95": float(np.percentile(rate_vals, 95)),
                "p99": float(np.percentile(rate_vals, 99)),
                "min": float(np.min(rate_vals)),
                "max": float(np.max(rate_vals)),
            },
            "num_normal_windows": len(normal_window_features),
        }

        print("\n" + "=" * 60)
        print("STEP 7: NORMAL TRAFFIC BASELINE ESTABLISHED")
        print("=" * 60)
        print(f"Windows analyzed: {self.stats['num_normal_windows']:,} (100% Benign Traffic)")
        print(f"Normal Mean IAT: {self.stats['normal_mean_iat']:.4e} s")
        print(f"Normal Mean Rate: {self.stats['normal_mean_rate']:.2f} pkts/s (p95: {self.stats['rate_percentiles']['p95']:.2f})")
        print(f"Normal CV Median: {self.stats['normal_cv_median']:.4f}, MAD: {self.stats['normal_cv_mad']:.4f}")
        print(f"Normal CV 95th Percentile: {self.stats['cv_percentiles']['p95']:.4f}")
        print("=" * 60 + "\n")

        return self.stats


class AdaptiveThresholdDetector:
    """
    Implements:
    1. Method 1: Simple Static CV Threshold
    2. Method 2: Dynamic Adaptive Median + MAD Threshold with Anti-Poisoning Protection
    """

    def __init__(self, baseline_stats: Dict[str, Any], cfg: ProjectConfig = config):
        self.cfg = cfg
        self.baseline_stats = baseline_stats
        
        # Simple threshold (e.g. 95th percentile of normal CV)
        self.simple_threshold = baseline_stats["cv_percentiles"]["p95"]
        
        # Initialize adaptive history buffer with initial normal CV values
        self.k = cfg.MAD_K
        self.initial_median = baseline_stats["normal_cv_median"]
        self.initial_mad = baseline_stats["normal_cv_mad"]
        
        # History queue of confirmed normal windows
        self.history_cv: List[float] = [self.initial_median] * 30
        self.buffer_size = cfg.ADAPTATION_BUFFER_SIZE

    def predict_simple_cv(self, df_windows: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method 1: Simple CV Threshold
        Anomaly condition: CV > simple_threshold
        
        Returns:
            binary_predictions: 0 for Normal, 1 for Attack
            threshold_values: Array of static threshold
        """
        cvs = df_windows["cv_iat"].to_numpy()
        preds = (cvs > self.simple_threshold).astype(int)
        thresholds = np.full_like(cvs, self.simple_threshold)
        return preds, thresholds

    def run_adaptive_detection(
        self,
        df_windows: pd.DataFrame,
        k: Optional[float] = None,
        confidence_risk_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Method 2: Adaptive Median + MAD Thresholding
        
        Equation:
            T_t = Median(CV_history) + k * MAD(CV_history)
            
        Anti-Poisoning Rule:
            Only update CV_history if the window is verified normal
            (e.g., CV <= T_t AND fuzzy risk < confidence threshold if available).
            
        Returns:
            Dictionary containing predictions, threshold history, and ratio values.
        """
        k_val = k if k is not None else self.k
        cvs = df_windows["cv_iat"].to_numpy()
        n = len(cvs)

        adaptive_thresholds = np.zeros(n, dtype=np.float64)
        threshold_ratios = np.zeros(n, dtype=np.float64)
        predictions = np.zeros(n, dtype=np.int32)
        history_buffer = list(self.history_cv)

        for i in range(n):
            current_cv = cvs[i]
            
            # 1. Compute current adaptive threshold from history
            med = float(np.median(history_buffer))
            mad = float(np.median(np.abs(np.array(history_buffer) - med)))
            if mad < 1e-4:
                mad = self.initial_mad

            t_t = med + k_val * mad
            adaptive_thresholds[i] = t_t
            
            # Threshold ratio = current_CV / adaptive_threshold
            ratio = current_cv / t_t if t_t > 1e-6 else 0.0
            threshold_ratios[i] = ratio

            # Anomaly decision
            is_anomaly = current_cv > t_t
            predictions[i] = 1 if is_anomaly else 0

            # 2. Poisoning prevention mechanism:
            # Only update normal history if window is classified as normal
            # and if external fuzzy confidence score confirms it (if provided)
            is_safe_normal = not is_anomaly
            if confidence_risk_scores is not None:
                is_safe_normal = is_safe_normal and (confidence_risk_scores[i] < self.cfg.HISTORY_UPDATE_CONFIDENCE_THRESHOLD)

            if is_safe_normal:
                history_buffer.append(current_cv)
                if len(history_buffer) > self.buffer_size:
                    history_buffer.pop(0)

        return {
            "predictions": predictions,
            "adaptive_thresholds": adaptive_thresholds,
            "threshold_ratios": threshold_ratios,
            "k_used": k_val,
        }
