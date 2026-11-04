"""
Data Loader module for CICIoT2023 dataset.

Handles dataset inspection, verification of file structure, column validation,
and memory-efficient loading while preserving sequential stream ordering.
Supports realistic evaluation stream scenarios with distinct operational phases.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List
import pandas as pd
import numpy as np

from src.config import ProjectConfig, config


class CICIoTDataLoader:
    """Manages reading and inspecting CICIoT2023 CSV files."""

    def __init__(self, cfg: Optional[ProjectConfig] = None):
        self.cfg = cfg or config
        self.cfg.ensure_directories()

    def inspect_dataset_files(self) -> Dict[str, Any]:
        """
        Inspect the dataset directory, file existence, sizes, and headers.
        Returns a dictionary summary.
        """
        train_path = self.cfg.DATASET_DIR / self.cfg.TRAIN_FILENAME
        test_path = self.cfg.DATASET_DIR / self.cfg.TEST_FILENAME

        report = {
            "dataset_dir": str(self.cfg.DATASET_DIR),
            "train_exists": train_path.exists(),
            "test_exists": test_path.exists(),
            "files": {}
        }

        print("=" * 60)
        print("STEP 1: DATASET INSPECTION & STRUCTURE VERIFICATION")
        print("=" * 60)
        print(f"Dataset Directory: {self.cfg.DATASET_DIR}")

        for name, path in [("train", train_path), ("test", test_path)]:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                header_df = pd.read_csv(path, nrows=5)
                cols = list(header_df.columns)
                has_ts = any("time" in c.lower() or c.lower() == "ts" for c in cols)
                has_iat = "IAT" in cols

                report["files"][name] = {
                    "path": str(path),
                    "size_mb": round(size_mb, 2),
                    "num_columns": len(cols),
                    "columns": cols,
                    "has_timestamp_col": has_ts,
                    "has_iat_col": has_iat,
                }

                print(f"\n[File: {path.name}]")
                print(f"  - Size: {size_mb:.2f} MB")
                print(f"  - Total Columns: {len(cols)}")
                print(f"  - Timestamp column present? {has_ts} (Constraint verified)")
                print(f"  - 'IAT' column present? {has_iat} (Primary feature verified)")
            else:
                print(f"\n[WARNING] File not found: {path}")

        return report

    def load_data(
        self,
        split: str = "test",
        sample_mode: Optional[bool] = None,
        sample_size: Optional[int] = None,
        use_core_features_only: bool = True,
    ) -> pd.DataFrame:
        """
        Load dataset split ('train' or 'test') sequentially.
        """
        filename = self.cfg.TRAIN_FILENAME if split == "train" else self.cfg.TEST_FILENAME
        filepath = self.cfg.DATASET_DIR / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at {filepath}.")

        is_sample = self.cfg.SAMPLE_MODE if sample_mode is None else sample_mode
        n_rows = self.cfg.SAMPLE_SIZE if sample_size is None else sample_size
        nrows_to_read = n_rows if is_sample else None

        cols_to_use = self.cfg.CORE_RAW_FEATURES + ["label"] if use_core_features_only else None

        print(f"Loading {split} split from {filepath.name} "
              f"({'sample ' + str(nrows_to_read) + ' rows' if is_sample else 'FULL dataset'})...")

        df = pd.read_csv(filepath, nrows=nrows_to_read, usecols=cols_to_use, low_memory=False)

        print(f"Loaded {len(df):,} rows and {df.shape[1]} columns. "
              f"Memory usage: {df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

        return df

    def get_normal_training_data(self, max_samples: int = 50000) -> pd.DataFrame:
        """
        Extract purely normal traffic (BenignTraffic) from training split
        to construct the unpolluted normal baseline.
        """
        train_path = self.cfg.DATASET_DIR / self.cfg.TRAIN_FILENAME
        if not train_path.exists():
            train_path = self.cfg.DATASET_DIR / self.cfg.TEST_FILENAME

        cols = self.cfg.CORE_RAW_FEATURES + ["label"]
        benign_dfs = []
        collected = 0

        print(f"Extracting up to {max_samples:,} NORMAL ('{self.cfg.NORMAL_LABEL}') samples for baseline...")

        for chunk in pd.read_csv(train_path, usecols=cols, chunksize=self.cfg.CHUNK_SIZE, low_memory=False):
            normal_chunk = chunk[chunk["label"] == self.cfg.NORMAL_LABEL]
            if len(normal_chunk) > 0:
                benign_dfs.append(normal_chunk)
                collected += len(normal_chunk)
                if collected >= max_samples:
                    break

        if not benign_dfs:
            raise ValueError(f"No samples with label '{self.cfg.NORMAL_LABEL}' found in {train_path}!")

        df_normal = pd.concat(benign_dfs, ignore_index=True)
        if len(df_normal) > max_samples:
            df_normal = df_normal.iloc[:max_samples]

        print(f"Successfully extracted {len(df_normal):,} pure normal training records for baseline.")
        return df_normal

    def load_evaluation_stream(
        self,
        total_records: int = 50000,
        normal_ratio: float = 0.30,
        zero_day_ratio: float = 0.20,
    ) -> pd.DataFrame:
        """
        Construct a continuous, sequential evaluation scenario from test.csv containing:
        Phase 1: Normal operational IoT traffic (Baseline verification)
        Phase 2: Known volumetric / DoS attacks
        Phase 3: Operational recovery (Normal IoT traffic)
        Phase 4: Held-out Zero-Day novel attacks (e.g. Mirai botnet)
        Phase 5: Mixed dynamic background traffic

        Preserves chronological ordering within each phase while providing
        a balanced, realistic anomaly detection timeline.
        """
        test_path = self.cfg.DATASET_DIR / self.cfg.TEST_FILENAME
        cols = self.cfg.CORE_RAW_FEATURES + ["label"]
        
        target_normal = int(total_records * normal_ratio)
        target_zero_day = int(total_records * zero_day_ratio)
        target_known = total_records - target_normal - target_zero_day

        benign_list = []
        known_list = []
        unseen_list = []

        print(f"Constructing realistic continuous evaluation stream ({total_records:,} flows)...")
        print(f"  Target: ~{target_normal:,} Normal, ~{target_known:,} Known Attacks, ~{target_zero_day:,} Unseen Attacks")

        for chunk in pd.read_csv(test_path, usecols=cols, chunksize=self.cfg.CHUNK_SIZE, low_memory=False):
            # 1. Benign
            b_chunk = chunk[chunk["label"] == self.cfg.NORMAL_LABEL]
            if len(b_chunk) > 0 and sum(len(x) for x in benign_list) < target_normal:
                benign_list.append(b_chunk)

            # 2. Unseen / Zero-day
            u_chunk = chunk[chunk["label"].isin(self.cfg.HELD_OUT_ATTACKS)]
            if len(u_chunk) > 0 and sum(len(x) for x in unseen_list) < target_zero_day:
                unseen_list.append(u_chunk)

            # 3. Known attacks
            k_chunk = chunk[(chunk["label"] != self.cfg.NORMAL_LABEL) & (~chunk["label"].isin(self.cfg.HELD_OUT_ATTACKS))]
            if len(k_chunk) > 0 and sum(len(x) for x in known_list) < target_known:
                known_list.append(k_chunk)

            if (sum(len(x) for x in benign_list) >= target_normal and
                sum(len(x) for x in unseen_list) >= target_zero_day and
                sum(len(x) for x in known_list) >= target_known):
                break

        df_b = pd.concat(benign_list, ignore_index=True).iloc[:target_normal]
        df_u = pd.concat(unseen_list, ignore_index=True).iloc[:target_zero_day]
        df_k = pd.concat(known_list, ignore_index=True).iloc[:target_known]

        # Structure realistic sequential timeline:
        # Phase 1: Normal (half of normal)
        # Phase 2: Known Attack (half of known)
        # Phase 3: Recovery Normal (second half of normal)
        # Phase 4: Zero-Day Attack (all unseen)
        # Phase 5: Known Attack burst (second half of known)
        n_half = len(df_b) // 2
        k_half = len(df_k) // 2

        phase1_norm = df_b.iloc[:n_half]
        phase2_known = df_k.iloc[:k_half]
        phase3_norm = df_b.iloc[n_half:]
        phase4_zero = df_u
        phase5_known = df_k.iloc[k_half:]

        eval_scenario = pd.concat(
            [phase1_norm, phase2_known, phase3_norm, phase4_zero, phase5_known],
            ignore_index=True
        )

        print(f"Continuous scenario assembled: {len(eval_scenario):,} flows.")
        print(f"  - Phase 1 (Normal Operations): {len(phase1_norm):,} flows")
        print(f"  - Phase 2 (Known Attacks):     {len(phase2_known):,} flows")
        print(f"  - Phase 3 (Network Recovery):  {len(phase3_norm):,} flows")
        print(f"  - Phase 4 (Unseen Zero-Day):   {len(phase4_zero):,} flows")
        print(f"  - Phase 5 (Secondary Attacks): {len(phase5_known):,} flows")

        return eval_scenario
