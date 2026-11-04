"""
Preprocessing and Label Analysis module.

Performs data cleaning, validation of IAT and numerical columns,
missing/infinite value handling, and binary label generation.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import ProjectConfig, config


class DataPreprocessor:
    """Preprocesses raw CICIoT2023 tabular flow data."""

    def __init__(self, cfg: ProjectConfig = config):
        self.cfg = cfg
        self.report: Dict[str, Any] = {}

    def clean_and_validate(self, df: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
        """
        Execute full data cleaning and validation pipeline.
        
        Steps:
        1. Numerical type enforcement
        2. Detection and handling of NaNs and infinite values
        3. Inspection of IAT (zero values, negative values)
        4. Preservation of stream integrity (no unnecessary row drops)
        5. Label formatting and binary indicator creation
        """
        print(f"\n--- Preprocessing & Data Cleaning: {dataset_name} ---")
        initial_rows = len(df)
        self.report[f"{dataset_name}_initial_rows"] = initial_rows

        df_clean = df.copy()

        # 1. Enforce numeric types on core features
        for col in self.cfg.CORE_RAW_FEATURES:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # 2. Check and report NaNs
        nan_counts = df_clean[self.cfg.CORE_RAW_FEATURES].isna().sum().to_dict()
        total_nans = sum(nan_counts.values())
        self.report[f"{dataset_name}_nan_counts"] = nan_counts

        if total_nans > 0:
            print(f"[Warning] Found {total_nans} NaN values across core features. Imputing with median.")
            for col in self.cfg.CORE_RAW_FEATURES:
                if df_clean[col].isna().sum() > 0:
                    median_val = df_clean[col].median()
                    df_clean[col].fillna(median_val, inplace=True)
        else:
            print("[OK] Missing values check: 0 NaNs detected.")

        # 3. Check and report Infinite values
        num_cols = [c for c in self.cfg.CORE_RAW_FEATURES if c in df_clean.columns]
        inf_mask = np.isinf(df_clean[num_cols].to_numpy())
        inf_count = int(np.sum(inf_mask))
        self.report[f"{dataset_name}_inf_count"] = inf_count

        if inf_count > 0:
            print(f"[Warning] Found {inf_count} infinite values. Clipping to feature max.")
            for col in num_cols:
                is_inf = np.isinf(df_clean[col])
                if is_inf.any():
                    finite_max = df_clean.loc[~is_inf, col].max()
                    df_clean.loc[is_inf, col] = finite_max
        else:
            print("[OK] Infinite values check: 0 infinite values detected.")

        # 4. Rigorous IAT Inspection
        iat = df_clean["IAT"]
        neg_iat_count = int((iat < 0).sum())
        zero_iat_count = int((iat == 0).sum())
        self.report[f"{dataset_name}_neg_iat"] = neg_iat_count
        self.report[f"{dataset_name}_zero_iat"] = zero_iat_count

        print(f"[OK] IAT Range: Min={iat.min():.6e}, Max={iat.max():.6e}, Median={iat.median():.6e}")
        print(f"  - Negative IAT count: {neg_iat_count}")
        print(f"  - Zero IAT count: {zero_iat_count}")

        if neg_iat_count > 0:
            # Negative IAT is physically invalid in flow arrival; clip to 0
            print(f"[Cleaning Note] {neg_iat_count} negative IAT records clamped to 0.")
            df_clean.loc[df_clean["IAT"] < 0, "IAT"] = 0.0

        if zero_iat_count > 0:
            # Zero IAT occurs when consecutive packets arrive within the measurement precision
            # Replace 0 with a tiny epsilon (1e-6) to ensure coefficient of variation (sigma/mu)
            # does not trigger divide-by-zero errors in window calculations.
            print(f"[Cleaning Note] {zero_iat_count} zero-value IAT records safely regularized with 1e-6 epsilon.")
            df_clean.loc[df_clean["IAT"] == 0, "IAT"] = 1e-6

        # 5. Label formatting
        df_clean, label_summary = self.analyze_labels(df_clean, dataset_name)
        self.report[f"{dataset_name}_label_summary"] = label_summary

        final_rows = len(df_clean)
        self.report[f"{dataset_name}_final_rows"] = final_rows
        print(f"Cleaning complete: {final_rows:,} records ready.\n")

        return df_clean

    def analyze_labels(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyze multi-class labels, identify normal vs attack distributions,
        and generate a clean binary_label column (0=Normal, 1=Attack).
        """
        if "label" not in df.columns:
            raise KeyError("Dataset does not contain a 'label' column.")

        # Strip whitespace from label names
        df["original_label"] = df["label"].astype(str).str.strip()

        # Class counts
        val_counts = df["original_label"].value_counts()
        total = len(df)
        
        # Verify normal label presence
        normal_count = int((df["original_label"] == self.cfg.NORMAL_LABEL).sum())
        attack_count = total - normal_count
        
        normal_pct = (normal_count / total) * 100 if total > 0 else 0
        attack_pct = (attack_count / total) * 100 if total > 0 else 0

        # Create binary label: 0 for BenignTraffic, 1 for Attack
        df["binary_label"] = (df["original_label"] != self.cfg.NORMAL_LABEL).astype(int)

        summary = {
            "total_records": total,
            "unique_labels": int(len(val_counts)),
            "normal_label": self.cfg.NORMAL_LABEL,
            "normal_count": normal_count,
            "normal_percent": round(normal_pct, 2),
            "attack_count": attack_count,
            "attack_percent": round(attack_pct, 2),
            "top_10_classes": val_counts.head(10).to_dict()
        }

        print("--- Label Analysis Summary ---")
        print(f"  - Total records: {total:,}")
        print(f"  - Unique classes: {len(val_counts)}")
        print(f"  - Normal ('{self.cfg.NORMAL_LABEL}'): {normal_count:,} ({normal_pct:.2f}%)")
        print(f"  - Attack records: {attack_count:,} ({attack_pct:.2f}%)")
        print(f"  - Top 5 classes: {dict(val_counts.head(5))}")

        return df, summary
