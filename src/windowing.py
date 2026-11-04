"""
Sliding Window Generator for IoT Network Flow Observations.

Constructs ordered sliding windows over consecutive dataset flow records.
Does NOT fabricate or invent timestamps. Window index represents sequential arrival order.
"""

from typing import List, Dict, Any, Generator, Tuple
import pandas as pd
import numpy as np

from src.config import ProjectConfig, config


class SlidingWindowGenerator:
    """Partitions sequential flow records into overlapping windows."""

    def __init__(
        self,
        window_size: int = 20,
        overlap: float = 0.50,
        attack_threshold: float = 0.50
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = max(1, int(window_size * (1.0 - overlap)))
        self.attack_threshold = attack_threshold

    def generate_windows(self, df: pd.DataFrame) -> Generator[Dict[str, Any], None, None]:
        """
        Yields window slices containing consecutive flow observations.
        
        Args:
            df: DataFrame containing sequential flow records with CORE_RAW_FEATURES,
                'binary_label', and 'original_label'.
        """
        n_rows = len(df)
        if n_rows < self.window_size:
            print(f"[Warning] Dataset length ({n_rows}) is smaller than window size ({self.window_size}).")
            return

        # Pre-extract numpy arrays for fast slicing
        iat_arr = df["IAT"].to_numpy(dtype=np.float64)
        rate_arr = df["Rate"].to_numpy(dtype=np.float64)
        number_arr = df["Number"].to_numpy(dtype=np.float64) if "Number" in df.columns else np.zeros(n_rows)
        duration_arr = df["flow_duration"].to_numpy(dtype=np.float64) if "flow_duration" in df.columns else np.zeros(n_rows)
        bin_label_arr = df["binary_label"].to_numpy(dtype=np.int32)
        orig_label_arr = df["original_label"].to_numpy()

        window_id = 0
        for start_idx in range(0, n_rows - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            
            # Slices
            w_bin_labels = bin_label_arr[start_idx:end_idx]
            w_orig_labels = orig_label_arr[start_idx:end_idx]
            
            attack_ratio = float(np.mean(w_bin_labels))
            # Window is attack if attack records meet or exceed threshold
            w_binary = 1 if attack_ratio >= self.attack_threshold else 0
            
            # Dominant multiclass label
            unique, counts = np.unique(w_orig_labels, return_counts=True)
            top_label = unique[np.argmax(counts)]

            yield {
                "window_id": window_id,
                "start_index": start_idx,
                "end_index": end_idx - 1,
                "iat": iat_arr[start_idx:end_idx],
                "rate": rate_arr[start_idx:end_idx],
                "number": number_arr[start_idx:end_idx],
                "flow_duration": duration_arr[start_idx:end_idx],
                "binary_label": w_binary,
                "attack_fraction": attack_ratio,
                "original_label": top_label,
            }
            window_id += 1
