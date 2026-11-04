"""
Comprehensive Evaluation and Benchmarking Module for Module 2.

Computes:
- Per-class Precision, Recall, F1-Score
- Macro-F1 and Weighted-F1
- Confusion Matrix visualization
- Model comparison tables (Random Forest vs FP32 LSTM vs INT8 LSTM)
- Prediction agreement between FP32 and INT8 models
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from src.module2.config import Module2Config, module2_config


class Module2Evaluator:
    """Evaluates multiclass predictions, generates diagnostic plots, and serializes reports."""

    def __init__(self, cfg: Module2Config = module2_config):
        self.cfg = cfg
        self.cfg.ensure_directories()

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """Calculates standard multiclass metrics and per-class reports."""
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

        report_dict = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))) if class_names else None,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

        return {
            "model_name": model_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "confusion_matrix": cm.tolist(),
            "per_class_report": report_dict,
        }

    def compute_agreement(self, y_fp32: np.ndarray, y_int8: np.ndarray) -> float:
        """Computes proportion of identical predictions between FP32 and INT8 models."""
        if len(y_fp32) == 0:
            return 1.0
        return float(np.mean(y_fp32 == y_int8))

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix",
        filename: str = "module2_confusion_matrix.png",
    ) -> Path:
        """Plots publication-grade normalized multiclass confusion matrix."""
        dest = self.cfg.PLOTS_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Normalize by true row sum
        with np.errstate(all="ignore"):
            cm_norm = cm.astype(np.float32) / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Normalized Recall"},
        )
        plt.title(title, fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("Predicted Class", fontsize=12)
        plt.ylabel("True Class", fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(dest, dpi=300)
        plt.close()
        print(f"Confusion matrix plot saved to {dest}")
        return dest

    def plot_model_comparison(
        self,
        comparison_df: pd.DataFrame,
        filename: str = "module2_model_comparison.png",
    ) -> Path:
        """Plots bar chart comparing RF vs FP32 LSTM vs INT8 LSTM."""
        dest = self.cfg.PLOTS_DIR / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        models = comparison_df["Model"].tolist()
        accs = comparison_df["Accuracy"].tolist()
        macro_f1s = comparison_df["Macro-F1"].tolist()
        latencies = comparison_df["Latency (ms)"].tolist()

        x = np.arange(len(models))
        width = 0.35

        # Plot 1: Accuracy & Macro-F1
        ax1.bar(x - width/2, accs, width, label="Accuracy", color="#2b5c8f")
        ax1.bar(x + width/2, macro_f1s, width, label="Macro-F1", color="#e28743")
        ax1.set_title("Classification Performance Comparison", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=11)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=10)
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        ax1.legend(loc="lower right")

        # Plot 2: Inference Latency
        colors = ["#2ca02c", "#1f77b4", "#ff7f0e"]
        ax2.bar(models, latencies, color=colors[:len(models)], width=0.5)
        ax2.set_title("Inference Latency per Sample (CPU)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Latency (ms)", fontsize=11)
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(dest, dpi=300)
        plt.close(fig)
        print(f"Model comparison chart saved to {dest}")
        return dest

    def save_metrics_report(self, report_dict: Dict[str, Any], filepath: Optional[Path] = None) -> Path:
        """Serializes complete benchmark results as formatted JSON."""
        dest = filepath or self.cfg.METRICS_PATH
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types
        def sanitize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj

        clean_report = sanitize(report_dict)
        with open(dest, "w") as f:
            json.dump(clean_report, f, indent=4)

        print(f"Module 2 benchmark metrics saved to {dest}")
        return dest
