"""
Visualization module for IoT Network Anomaly Detection.

Generates 10+ publication-grade plots saved directly to results/plots/:
- IAT distributions (overall & normal vs attack)
- CV & Rate distributions
- Sliding-window CV timelines & Adaptive Threshold tracking
- Fuzzy Membership Functions
- Attack Risk Score timeline with Normal/Suspicious/Attack decision zones
- Confusion Matrices across all methods
- Comprehensive method performance comparison bar charts
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import ProjectConfig, config


class ResearchVisualizer:
    """Produces publication-ready charts and diagnostic figures."""

    def __init__(self, cfg: ProjectConfig = config):
        self.cfg = cfg
        self.cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        # Apply clean research styling
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300
        })

    def plot_01_iat_distribution(self, df: pd.DataFrame) -> Path:
        """Plot 1: Overall raw IAT distribution (log scale)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        iat_vals = df["IAT"].dropna()
        # Filter for positive values to allow log scale representation
        pos_iat = iat_vals[iat_vals > 0]
        
        sns.histplot(pos_iat, bins=50, log_scale=True, kde=True, color="#2b5c8f", ax=ax)
        ax.set_title("1. Distribution of Flow Inter-Arrival Time (IAT) in CICIoT2023")
        ax.set_xlabel("IAT (log scale)")
        ax.set_ylabel("Frequency (Flow Count)")
        
        out_path = self.cfg.PLOTS_DIR / "01_iat_distribution.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_02_iat_normal_vs_attack(self, df: pd.DataFrame) -> Path:
        """Plot 2: IAT distribution for Normal vs Attack traffic."""
        fig, ax = plt.subplots(figsize=(9, 5))
        
        normal_iat = df[df["binary_label"] == 0]["IAT"]
        attack_iat = df[df["binary_label"] == 1]["IAT"]

        normal_pos = normal_iat[normal_iat > 0]
        attack_pos = attack_iat[attack_iat > 0]

        sns.kdeplot(normal_pos, log_scale=True, label=f"Normal ({self.cfg.NORMAL_LABEL})",
                    color="#2ca02c", linewidth=2.5, ax=ax)
        sns.kdeplot(attack_pos, log_scale=True, label="Attack Traffic",
                    color="#d62728", linewidth=2.0, linestyle="--", ax=ax)

        ax.set_title("2. Flow IAT Density: Normal vs Attack Traffic")
        ax.set_xlabel("IAT (log scale)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper left")

        out_path = self.cfg.PLOTS_DIR / "02_iat_normal_vs_attack.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_03_cv_distribution(self, df_windows: pd.DataFrame) -> Path:
        """Plot 3: Distribution of Coefficient of Variation (CV) across windows."""
        fig, ax = plt.subplots(figsize=(9, 5))
        
        normal_cv = df_windows[df_windows["binary_label"] == 0]["cv_iat"]
        attack_cv = df_windows[df_windows["binary_label"] == 1]["cv_iat"]

        sns.histplot(normal_cv, bins=40, color="#2ca02c", alpha=0.6, label="Normal Windows", kde=True, ax=ax)
        sns.histplot(attack_cv, bins=40, color="#d62728", alpha=0.5, label="Attack Windows", kde=True, ax=ax)

        ax.set_title("3. Distribution of Windowed Coefficient of Variation (CV)")
        ax.set_xlabel("CV (std_iat / mean_iat)")
        ax.set_ylabel("Window Count")
        ax.legend(loc="upper right")

        out_path = self.cfg.PLOTS_DIR / "03_cv_distribution.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_04_packet_rate_distribution(self, df_windows: pd.DataFrame) -> Path:
        """Plot 4: Distribution of Windowed Mean Packet Rate."""
        fig, ax = plt.subplots(figsize=(9, 5))

        normal_rate = df_windows[df_windows["binary_label"] == 0]["mean_rate"]
        attack_rate = df_windows[df_windows["binary_label"] == 1]["mean_rate"]

        sns.kdeplot(normal_rate[normal_rate > 0], log_scale=True, label="Normal Windows",
                    color="#2ca02c", linewidth=2.5, ax=ax)
        sns.kdeplot(attack_rate[attack_rate > 0], log_scale=True, label="Attack Windows",
                    color="#d62728", linewidth=2.0, linestyle="--", ax=ax)

        ax.set_title("4. Window Mean Packet Rate Density (Normal vs Attack)")
        ax.set_xlabel("Mean Packet Rate (log scale, pkts/s)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")

        out_path = self.cfg.PLOTS_DIR / "04_packet_rate_distribution.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_05_sliding_window_cv_timeline(self, df_windows: pd.DataFrame, max_windows: int = 500) -> Path:
        """Plot 5: Windowed CV over sequential window IDs."""
        sample_w = df_windows.iloc[:max_windows]
        fig, ax = plt.subplots(figsize=(12, 5))

        ax.plot(sample_w["window_id"], sample_w["cv_iat"], color="#3b528b", linewidth=1.5, label="Window CV")
        
        # Shade attack windows
        attack_mask = sample_w["binary_label"] == 1
        ax.fill_between(
            sample_w["window_id"], 0, sample_w["cv_iat"].max() * 1.05,
            where=attack_mask, color="#ff9896", alpha=0.35, label="Attack Window Ground Truth"
        )

        ax.set_title(f"5. Sliding Window CV Sequence (First {len(sample_w)} Windows)")
        ax.set_xlabel("Window ID (Sequential Arrival Order)")
        ax.set_ylabel("Coefficient of Variation (CV)")
        ax.legend(loc="upper right")

        out_path = self.cfg.PLOTS_DIR / "05_sliding_window_cv_timeline.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_06_adaptive_threshold_vs_cv(
        self,
        df_windows: pd.DataFrame,
        adaptive_thresholds: np.ndarray,
        max_windows: int = 500
    ) -> Path:
        """Plot 6: Adaptive Threshold T_t vs CV over sequence."""
        n = min(len(df_windows), max_windows)
        sample_w = df_windows.iloc[:n]
        sub_t = adaptive_thresholds[:n]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(sample_w["window_id"], sample_w["cv_iat"], color="#1f77b4", label="Observed Window CV", alpha=0.85)
        ax.plot(sample_w["window_id"], sub_t, color="#ff7f0e", linestyle="--", linewidth=2.0, label="Adaptive Threshold (Median + k*MAD)")

        ax.set_title(f"6. Adaptive Threshold Dynamic Tracking vs Window CV (n={n})")
        ax.set_xlabel("Window ID")
        ax.set_ylabel("CV Metric")
        ax.legend(loc="upper right")

        out_path = self.cfg.PLOTS_DIR / "06_adaptive_threshold_vs_cv.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_07_fuzzy_membership_functions(self, fuzzy_system: Any) -> Path:
        """Plot 7: Membership functions for CV, Rate, IAT Deviation, and Risk."""
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))

        # CV
        ax = axes[0, 0]
        u_cv = fuzzy_system.cv.universe
        for term, col in [("low", "#2ca02c"), ("medium", "#ff7f0e"), ("high", "#d62728")]:
            ax.plot(u_cv, fuzzy_system.cv[term].mf, label=term.upper(), color=col, linewidth=2)
        ax.set_title("Input 1: Coefficient of Variation (CV)")
        ax.set_xlabel("CV Value")
        ax.set_ylabel("Membership Degree")
        ax.legend()

        # Rate
        ax = axes[0, 1]
        u_rate = fuzzy_system.rate.universe
        for term, col in [("low", "#2ca02c"), ("medium", "#ff7f0e"), ("high", "#d62728")]:
            ax.plot(u_rate, fuzzy_system.rate[term].mf, label=term.upper(), color=col, linewidth=2)
        ax.set_title("Input 2: Packet Rate (pkts/s)")
        ax.set_xlabel("Rate")
        ax.set_ylabel("Membership Degree")
        ax.legend()

        # IAT Deviation
        ax = axes[1, 0]
        u_dev = fuzzy_system.iat_dev.universe
        for term, col in [("low", "#2ca02c"), ("medium", "#ff7f0e"), ("high", "#d62728")]:
            ax.plot(u_dev, fuzzy_system.iat_dev[term].mf, label=term.upper(), color=col, linewidth=2)
        ax.set_title("Input 3: IAT Deviation (|mu - mu_norm| / mu_norm)")
        ax.set_xlabel("Relative Deviation")
        ax.set_ylabel("Membership Degree")
        ax.legend()

        # Attack Risk Consequent
        ax = axes[1, 1]
        u_risk = fuzzy_system.risk.universe
        for term, col in [("low", "#2ca02c"), ("medium", "#ff7f0e"), ("high", "#d62728")]:
            ax.plot(u_risk, fuzzy_system.risk[term].mf, label=term.upper(), color=col, linewidth=2)
        ax.set_title("Output: Attack Risk Score [0, 1]")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Membership Degree")
        ax.legend()

        fig.suptitle("7. Mamdani Fuzzy System Membership Functions", fontsize=15, weight="bold")
        out_path = self.cfg.PLOTS_DIR / "07_fuzzy_membership_functions.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_08_attack_risk_timeline(
        self,
        df_windows: pd.DataFrame,
        risk_scores: np.ndarray,
        max_windows: int = 400
    ) -> Path:
        """Plot 8: Continuous Attack Risk timeline with Normal, Suspicious, Attack zones."""
        n = min(len(df_windows), max_windows)
        sub_df = df_windows.iloc[:n]
        sub_scores = risk_scores[:n]

        fig, ax = plt.subplots(figsize=(13, 6))

        # Decision zone boundaries
        ax.axhspan(0.0, self.cfg.FUZZY_LOW_THRESHOLD, color="#d4edda", alpha=0.5, label="Normal Zone")
        ax.axhspan(self.cfg.FUZZY_LOW_THRESHOLD, self.cfg.FUZZY_HIGH_THRESHOLD, color="#fff3cd", alpha=0.5, label="Suspicious Zone")
        ax.axhspan(self.cfg.FUZZY_HIGH_THRESHOLD, 1.0, color="#f8d7da", alpha=0.5, label="Attack Zone")

        # Attack Risk trajectory
        ax.plot(sub_df["window_id"], sub_scores, color="#343a40", linewidth=1.8, label="Fuzzy Attack Risk Score", zorder=3)
        
        # Ground truth markers
        attack_idx = sub_df[sub_df["binary_label"] == 1]["window_id"]
        attack_sc = sub_scores[sub_df["binary_label"] == 1]
        ax.scatter(attack_idx, attack_sc, color="#dc3545", s=18, alpha=0.6, label="Ground Truth Attack Window", zorder=4)

        ax.set_title(f"8. Mamdani Fuzzy Attack Risk Score over Time (n={n} windows)")
        ax.set_xlabel("Window ID")
        ax.set_ylabel("Attack Risk [0, 1]")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", framealpha=0.9)

        out_path = self.cfg.PLOTS_DIR / "08_attack_risk_timeline.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_09_confusion_matrices(
        self,
        cm_simple: List[List[int]],
        cm_adaptive: List[List[int]],
        cm_fuzzy: List[List[int]]
    ) -> Path:
        """Plot 9: Confusion matrices for all three methods side by side."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        labels = ["Normal", "Attack"]

        cms = [
            ("Method 1: Simple CV", cm_simple, axes[0]),
            ("Method 2: Adaptive MAD", cm_adaptive, axes[1]),
            ("Method 3: Mamdani Fuzzy", cm_fuzzy, axes[2]),
        ]

        for title, cm, ax in cms:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

        fig.suptitle("9. Confusion Matrix Comparison Across Detection Methods", fontsize=14, weight="bold")
        out_path = self.cfg.PLOTS_DIR / "09_confusion_matrices.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_10_performance_comparison(self, df_comparison: pd.DataFrame) -> Path:
        """Plot 10: Comparative bar chart of Accuracy, Precision, Recall, F1, and Zero-Day Detection."""
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_to_plot = ["accuracy", "precision", "recall_tpr", "f1_score"]
        if "zero_day_detection_rate" in df_comparison.columns:
            metrics_to_plot.append("zero_day_detection_rate")

        plot_data = pd.melt(
            df_comparison,
            id_vars=["method"],
            value_vars=metrics_to_plot,
            var_name="Metric",
            value_name="Score"
        )
        
        # Clean metric labels
        metric_map = {
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall_tpr": "Recall (TPR)",
            "f1_score": "F1-Score",
            "zero_day_detection_rate": "Zero-Day Detection"
        }
        plot_data["Metric"] = plot_data["Metric"].map(metric_map)

        sns.barplot(data=plot_data, x="Metric", y="Score", hue="method", palette="tab10", ax=ax)
        ax.set_title("10. Performance Comparison: Baseline Methods vs Fuzzy Logic", fontweight="bold")
        ax.set_ylabel("Score (0.0 to 1.0)")
        ax.set_ylim(0.0, 1.1)
        ax.legend(title="Method", loc="lower right")

        for p in ax.patches:
            height = p.get_height()
            if not np.isnan(height) and height > 0:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=8,
                            xytext=(0, 3), textcoords="offset points")

        out_path = self.cfg.PLOTS_DIR / "10_performance_comparison.png"
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    def plot_sensitivity_analysis(
        self,
        window_results: Dict[int, float],
        k_results: Dict[float, float]
    ) -> Tuple[Path, Path]:
        """Plot 11 & 12: Sensitivity analysis for Window Size and Adaptive k parameter."""
        # 11: Window size sensitivity
        fig1, ax1 = plt.subplots(figsize=(7, 4.5))
        w_sizes = list(window_results.keys())
        w_scores = list(window_results.values())
        ax1.plot(w_sizes, w_scores, marker="o", color="#2b5c8f", linewidth=2.0)
        ax1.set_title("11. Sensitivity: Window Size vs Detection F1-Score")
        ax1.set_xlabel("Window Size (Observations)")
        ax1.set_ylabel("F1-Score")
        ax1.set_ylim(0.0, 1.05)
        path1 = self.cfg.PLOTS_DIR / "11_window_size_sensitivity.png"
        fig1.tight_layout()
        fig1.savefig(path1)
        plt.close(fig1)

        # 12: k parameter sensitivity
        fig2, ax2 = plt.subplots(figsize=(7, 4.5))
        k_vals = list(k_results.keys())
        k_scores = list(k_results.values())
        ax2.plot(k_vals, k_scores, marker="s", color="#e28743", linewidth=2.0)
        ax2.set_title("12. Sensitivity: Adaptive Parameter (k) vs Alarm Rate")
        ax2.set_xlabel("MAD Multiplier (k)")
        ax2.set_ylabel("Detection Alarm Rate")
        ax2.set_ylim(0.0, 1.05)
        path2 = self.cfg.PLOTS_DIR / "12_mad_k_sensitivity.png"
        fig2.tight_layout()
        fig2.savefig(path2)
        plt.close(fig2)

        return path1, path2
