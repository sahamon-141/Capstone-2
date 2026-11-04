"""
Statistical Feature Engineering for Sliding Windows.

Computes Mean IAT, Std IAT, Coefficient of Variation (CV), Rate,
Flow Duration, Number, and IAT Deviation from normal baseline.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import ProjectConfig, config
from src.windowing import SlidingWindowGenerator


class WindowFeatureExtractor:
    """Computes statistical and temporal features across sliding windows."""

    def __init__(self, normal_mean_iat: Optional[float] = None, cfg: ProjectConfig = config):
        self.cfg = cfg
        self.normal_mean_iat = normal_mean_iat

    def set_normal_baseline(self, normal_mean_iat: float) -> None:
        """Update the normal mean IAT learned from baseline normal traffic."""
        self.normal_mean_iat = float(normal_mean_iat)

    def extract_features(
        self,
        df: pd.DataFrame,
        window_size: Optional[int] = None,
        overlap: Optional[float] = None,
        save_csv: bool = False,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Processes dataframe into window statistical features.
        
        Args:
            df: Input cleaned flow records.
            window_size: Window size in observations.
            overlap: Overlap fraction.
            save_csv: Whether to save to results/window_features.csv.
            output_path: Optional destination path.
        """
        w_size = window_size or self.cfg.WINDOW_SIZE
        w_overlap = overlap if overlap is not None else self.cfg.OVERLAP
        
        generator = SlidingWindowGenerator(
            window_size=w_size,
            overlap=w_overlap,
            attack_threshold=self.cfg.WINDOW_ATTACK_THRESHOLD
        )

        rows = []
        # Fallback if normal_mean_iat is not yet set: use median/mean of initial benign records
        ref_mean_iat = self.normal_mean_iat
        if ref_mean_iat is None or ref_mean_iat <= 0:
            benign_rows = df[df["binary_label"] == 0]
            if len(benign_rows) > 0:
                ref_mean_iat = float(benign_rows["IAT"].mean())
            else:
                ref_mean_iat = float(df["IAT"].mean())

        for w in generator.generate_windows(df):
            iat_data = w["iat"]
            rate_data = w["rate"]
            num_data = w["number"]
            dur_data = w["flow_duration"]

            # 1. Mean IAT
            m_iat = float(np.mean(iat_data))
            
            # 2. Std IAT (population std)
            s_iat = float(np.std(iat_data, ddof=0))
            
            # 3. Coefficient of Variation (CV = sigma / mu)
            cv_iat = float(s_iat / m_iat) if m_iat > 1e-12 else 0.0

            # 4. Rate statistics
            m_rate = float(np.mean(rate_data))
            s_rate = float(np.std(rate_data, ddof=0))

            # 5. Number and Flow Duration
            m_number = float(np.mean(num_data))
            m_dur = float(np.mean(dur_data))

            # 6. IAT Deviation from normal baseline: D = |mu_current - mu_normal| / mu_normal
            iat_dev = float(abs(m_iat - ref_mean_iat) / ref_mean_iat) if ref_mean_iat > 1e-12 else 0.0

            rows.append({
                "window_id": w["window_id"],
                "start_index": w["start_index"],
                "end_index": w["end_index"],
                "mean_iat": m_iat,
                "std_iat": s_iat,
                "cv_iat": cv_iat,
                "mean_rate": m_rate,
                "std_rate": s_rate,
                "mean_number": m_number,
                "mean_flow_duration": m_dur,
                "iat_deviation": iat_dev,
                "original_label": w["original_label"],
                "binary_label": w["binary_label"],
                "attack_fraction": w["attack_fraction"]
            })

        features_df = pd.DataFrame(rows)

        if save_csv:
            csv_path = output_path or self.cfg.WINDOW_FEATURES_PATH
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            features_df.to_csv(csv_path, index=False)
            print(f"Saved {len(features_df):,} window feature records to {csv_path}")

        return features_df
