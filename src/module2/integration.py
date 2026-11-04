"""
Integration Pipeline: Module 1 (Fuzzy Anomaly Detection) -> Module 2 (LSTM Classification).

Connects Module 1's window-level anomaly alerts to Module 2's sequential deep neural network.
End-to-End Workflow:
1. Module 1 processes incoming observation windows.
2. If Attack_Risk < 0.35 (NORMAL):
       Traffic passes freely; Module 2 remains idle (conserving IoT compute & battery).
3. If Attack_Risk >= 0.35 (SUSPICIOUS or ATTACK):
       Retrieve the associated ordered observation sequence [start_idx .. end_idx].
       Apply Module 2 preprocessing.
       Invoke FP32 or INT8 LSTM model.
       Output predicted attack class, confidence, latency, and unknown/zero-day status.
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.module2.config import Module2Config, module2_config
from src.module2.preprocessor import Module2Preprocessor
from src.module2.inference import Module2InferenceEngine


class PipelineIntegrator:
    """Orchestrates end-to-end handoff from Module 1 alerts to Module 2 classification."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_quantized: bool = True,
        rejection_threshold: float = 0.50,
        cfg: Module2Config = module2_config,
    ):
        self.cfg = cfg
        self.model_dir = model_dir or self.cfg.MODELS_DIR
        self.use_quantized = use_quantized
        self.rejection_threshold = rejection_threshold

        self.engine = Module2InferenceEngine(
            model_dir=self.model_dir,
            use_quantized=self.use_quantized,
            rejection_threshold=self.rejection_threshold,
            cfg=self.cfg,
        )

    def process_module1_window(
        self,
        module1_window_result: Dict[str, Any],
        raw_flows_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Receives a single window decision from Module 1 and, if anomalous,
        retrieves associated flow records and runs Module 2 classification.
        
        Args:
            module1_window_result: Dict containing:
                - window_id: int
                - start_index: int
                - end_index: int
                - attack_risk: float
                - is_anomalous: bool (or fuzzy_pred == 1)
                - category: 'NORMAL', 'SUSPICIOUS', or 'ATTACK'
            raw_flows_df: Original unscaled DataFrame containing raw flow records.
        """
        t_start = time.perf_counter()

        window_id = module1_window_result.get("window_id", 0)
        risk_score = float(module1_window_result.get("attack_risk", 0.0))
        is_anomalous = bool(module1_window_result.get("is_anomalous", risk_score >= 0.35))
        start_idx = int(module1_window_result.get("start_index", 0))
        end_idx = int(module1_window_result.get("end_index", start_idx + 19))

        # 1. Normal traffic bypasses Module 2 (saves IoT computational budget)
        if not is_anomalous:
            return {
                "window_id": window_id,
                "module1_verdict": "NORMAL",
                "attack_risk": round(risk_score, 4),
                "module2_invoked": False,
                "predicted_attack": "None (Traffic is Normal)",
                "confidence": 1.0 - round(risk_score, 4),
                "is_zero_day": False,
                "total_pipeline_latency_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
            }

        # 2. Anomalous traffic: Retrieve associated sequence
        # Target length is Module 2's sequence length (e.g. 50 observations ending at end_idx)
        seq_len = self.cfg.DEFAULT_SEQUENCE_LENGTH
        actual_start = max(0, end_idx - seq_len + 1)
        seq_df = raw_flows_df.iloc[actual_start : end_idx + 1]

        # Pad with earliest available observation if window is near stream start
        if len(seq_df) < seq_len:
            pad_count = seq_len - len(seq_df)
            first_row = seq_df.iloc[[0] * pad_count]
            seq_df = pd.concat([first_row, seq_df], ignore_index=True)

        # 3. Apply Module 2 fitted preprocessing
        seq_clean = self.engine.preprocessor.transform(seq_df)
        seq_features = seq_clean[self.engine.preprocessor.feature_cols].to_numpy(dtype=np.float32)

        # 4. Invoke Module 2 Classifier (FP32 or INT8)
        m2_res = self.engine.predict_single_sequence(seq_features)
        total_latency = (time.perf_counter() - t_start) * 1000.0

        return {
            "window_id": window_id,
            "module1_verdict": module1_window_result.get("category", "ATTACK"),
            "attack_risk": round(risk_score, 4),
            "module2_invoked": True,
            "predicted_attack": m2_res["predicted_class"],
            "raw_class": m2_res["raw_class"],
            "class_id": m2_res["class_id"],
            "confidence": m2_res["confidence"],
            "is_zero_day": m2_res["is_uncertain_rejection"],
            "module2_runtime": m2_res["runtime"],
            "module2_inference_latency_ms": m2_res["latency_ms"],
            "total_pipeline_latency_ms": round(total_latency, 3),
        }

    def run_batch_integration(
        self,
        module1_windows_df: pd.DataFrame,
        raw_flows_df: pd.DataFrame,
        max_windows: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Executes end-to-end integration over a batch of Module 1 window records.
        """
        n_windows = len(module1_windows_df)
        limit = min(n_windows, max_windows) if max_windows else n_windows

        results = []
        invocations = 0

        print(f"\n--- Running End-to-End Pipeline Integration over {limit:,} windows ---")
        for i in range(limit):
            row = module1_windows_df.iloc[i]
            w_res = {
                "window_id": int(row["window_id"]),
                "start_index": int(row["start_index"]),
                "end_index": int(row["end_index"]),
                "attack_risk": float(row.get("attack_risk", 0.0)),
                "category": str(row.get("fuzzy_category", "ATTACK")),
                "is_anomalous": bool(row.get("fuzzy_pred", row.get("binary_label", 1)) == 1),
            }
            integrated = self.process_module1_window(w_res, raw_flows_df)
            if integrated["module2_invoked"]:
                invocations += 1
            results.append(integrated)

        df_out = pd.DataFrame(results)
        print(f"Integration complete: {limit:,} windows processed, {invocations:,} anomalous windows analyzed by Module 2.")
        return df_out
