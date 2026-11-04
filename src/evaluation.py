"""
Evaluation and Metrics module.

Computes Accuracy, Precision, Recall, F1-Score, FPR, TPR, Confusion Matrices,
and sensitivity analyses for window sizes and adaptive threshold parameters.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from src.config import ProjectConfig, config


class PerformanceEvaluator:
    """Evaluates anomaly detection pipelines and generates structured reports."""

    def __init__(self, cfg: ProjectConfig = config):
        self.cfg = cfg
        self.cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    def calculate_binary_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        method_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Compute standard binary classification and anomaly metrics.
        """
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tpr = rec  # TPR is recall
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        return {
            "method": method_name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall_tpr": round(rec, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "confusion_matrix": cm.tolist()
        }

    def compare_methods(
        self,
        y_true: np.ndarray,
        preds_dict: Dict[str, np.ndarray],
        cohort_results: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Compare Method 1 (Simple CV), Method 2 (Adaptive MAD), and Method 3 (Fuzzy Logic).
        """
        records = []
        for name, y_pred in preds_dict.items():
            metrics = self.calculate_binary_metrics(y_true, y_pred, method_name=name)
            
            # Incorporate zero-day cohort specific recall if available
            unseen_data = {}
            if cohort_results:
                for k in ["unseen", "held_out_unseen", "held_out", "zero_day", "held_out_zero_day"]:
                    if k in cohort_results:
                        unseen_data = cohort_results[k]
                        break

            if name == "Simple CV":
                metrics["zero_day_detection_rate"] = round(unseen_data.get("detection_rate_simple", 0.0), 4)
            elif name == "Adaptive MAD":
                metrics["zero_day_detection_rate"] = round(unseen_data.get("detection_rate_adaptive", 0.0), 4)
            elif name == "Fuzzy Logic":
                metrics["zero_day_detection_rate"] = round(unseen_data.get("detection_rate_fuzzy", 0.0), 4)
            else:
                metrics["zero_day_detection_rate"] = 0.0

            records.append(metrics)

        df_comparison = pd.DataFrame(records)
        csv_path = self.cfg.METRICS_DIR / "method_comparison.csv"
        df_comparison.to_csv(csv_path, index=False)
        print(f"\nSaved method comparison metrics to {csv_path}")

        return df_comparison

    def save_summary_report(self, report_dict: Dict[str, Any], filename: str = "summary_metrics.json") -> Path:
        """Save complete dictionary report as formatted JSON."""
        dest_path = self.cfg.METRICS_DIR / filename
        
        # Helper to convert numpy objects for JSON serialization
        def sanitize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64, np.floating)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64, np.integer)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        clean_report = sanitize(report_dict)
        with open(dest_path, "w") as f:
            json.dump(clean_report, f, indent=4)

        print(f"Summary metrics report successfully saved to {dest_path}")
        return dest_path
