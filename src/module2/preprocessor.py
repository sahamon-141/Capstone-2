"""
Preprocessing and Data Partitioning Module for Module 2.

Handles:
- Missing value imputation (fitted strictly on training split)
- Infinite value clamping
- Numerical type safety
- Feature scaling using StandardScaler (fit on train only, applied to val/test)
- Label encoding with bidirectional mapping (str <-> int)
- Artifact serialization for reproducible inference (FP32, INT8, Pi deployment)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.module2.config import Module2Config, module2_config


class Module2Preprocessor:
    """Manages feature cleaning, scaling, and multiclass label encoding."""

    def __init__(self, cfg: Module2Config = module2_config):
        self.cfg = cfg
        self.feature_cols: List[str] = list(self.cfg.SELECTED_FEATURES)
        self.scaler: Optional[StandardScaler] = None
        self.feature_medians: Dict[str, float] = {}
        self.feature_maxs: Dict[str, float] = {}
        self.label_to_id: Dict[str, int] = {}
        self.id_to_label: Dict[int, str] = {}
        self.is_fitted: bool = False

    def fit_labels(self, labels: List[str]) -> Dict[str, int]:
        """
        Constructs deterministic integer mapping for multiclass target labels.
        Always sets BenignTraffic to ID 0 if present.
        """
        unique_labels = sorted(list(set(labels)))
        if self.cfg.NORMAL_LABEL in unique_labels:
            unique_labels.remove(self.cfg.NORMAL_LABEL)
            unique_labels = [self.cfg.NORMAL_LABEL] + unique_labels

        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        return self.label_to_id

    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """Encodes string labels into integer class IDs."""
        if not self.label_to_id:
            raise ValueError("Label mapping not fitted. Call fit_labels first.")
        return labels.map(self.label_to_id).fillna(-1).to_numpy(dtype=np.int32)

    def decode_labels(self, ids: np.ndarray) -> List[str]:
        """Decodes integer class IDs back into human-readable attack names."""
        return [self.id_to_label.get(int(i), "Unknown") for i in ids]

    def fit(self, df_train: pd.DataFrame) -> "Module2Preprocessor":
        """
        Fits imputers, clippers, scaler, and label encoder on training records only.
        Guarantees zero information leakage from validation or test sets.
        """
        print(f"Fitting Module 2 Preprocessor on {len(df_train):,} training records...")

        # 1. Fit Label Encoding
        if "label" in df_train.columns:
            raw_labels = df_train["label"].astype(str).str.strip().tolist()
            self.fit_labels(raw_labels)

        # 2. Compute training medians and max values for feature imputation & clipping
        df_clean = df_train.copy()
        for col in self.feature_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                # Replace inf with nan temporarily to compute true finite median & max
                finite_series = df_clean[col].replace([np.inf, -np.inf], np.nan)
                med = float(finite_series.median()) if not np.isnan(finite_series.median()) else 0.0
                f_max = float(finite_series.max()) if not np.isnan(finite_series.max()) else 1.0

                self.feature_medians[col] = med
                self.feature_maxs[col] = f_max

                df_clean[col] = df_clean[col].fillna(med)
                df_clean[col] = df_clean[col].replace([np.inf, -np.inf], f_max)

        # 3. Fit StandardScaler strictly on cleaned training features
        X_train_raw = df_clean[self.feature_cols].to_numpy(dtype=np.float32)
        self.scaler = StandardScaler()
        self.scaler.fit(X_train_raw)

        self.is_fitted = True
        print(f"Preprocessor fit complete for {len(self.feature_cols)} features and {len(self.label_to_id)} classes.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned imputation, clipping, and scaling to any dataset partition.
        """
        if not self.is_fitted or self.scaler is None:
            raise ValueError("Preprocessor is not fitted. Call fit() first.")

        df_out = df.copy()

        # Clean features using saved training parameters
        for col in self.feature_cols:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
                med = self.feature_medians.get(col, 0.0)
                f_max = self.feature_maxs.get(col, 1.0)

                df_out[col] = df_out[col].fillna(med)
                df_out[col] = df_out[col].replace([np.inf, -np.inf], f_max)
            else:
                df_out[col] = self.feature_medians.get(col, 0.0)

        # Scale features
        X_raw = df_out[self.feature_cols].to_numpy(dtype=np.float32)
        X_scaled = self.scaler.transform(X_raw)

        for idx, col in enumerate(self.feature_cols):
            df_out[col] = X_scaled[:, idx]

        # Encode label if present
        if "label" in df_out.columns:
            clean_labels = df_out["label"].astype(str).str.strip()
            df_out["label_encoded"] = self.encode_labels(clean_labels)

        return df_out

    def transform_sequence_array(self, seq_array: np.ndarray) -> np.ndarray:
        """
        Transforms a 3D sequence array of shape (N, T, F) using the fitted scaler.
        """
        if not self.is_fitted or self.scaler is None:
            raise ValueError("Preprocessor is not fitted.")

        N, T, F = seq_array.shape
        flat = seq_array.reshape(-1, F)
        scaled_flat = self.scaler.transform(flat)
        return scaled_flat.reshape(N, T, F).astype(np.float32)

    def save_artifacts(self, save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """Saves scaler, label map, and feature list for reproducible deployment."""
        target_dir = save_dir or self.cfg.MODELS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        scaler_path = target_dir / "module2_scaler.pkl"
        label_path = target_dir / "module2_label_map.json"
        features_path = target_dir / "module2_features.json"

        # 1. Save Scaler
        with open(scaler_path, "wb") as f:
            pickle.dump({
                "scaler": self.scaler,
                "medians": self.feature_medians,
                "maxs": self.feature_maxs,
            }, f)

        # 2. Save Label Map
        with open(label_path, "w") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": {str(k): v for k, v in self.id_to_label.items()},
                "num_classes": len(self.label_to_id),
            }, f, indent=4)

        # 3. Save Features Metadata
        with open(features_path, "w") as f:
            json.dump({
                "features": self.feature_cols,
                "num_features": len(self.feature_cols),
                "sequence_length": self.cfg.DEFAULT_SEQUENCE_LENGTH,
            }, f, indent=4)

        print(f"Preprocessor artifacts saved to {target_dir}")
        return {
            "scaler": scaler_path,
            "label_map": label_path,
            "features": features_path,
        }

    def load_artifacts(self, model_dir: Optional[Path] = None) -> "Module2Preprocessor":
        """Loads fitted scaler, label mapping, and feature configuration."""
        target_dir = model_dir or self.cfg.MODELS_DIR
        scaler_path = target_dir / "module2_scaler.pkl"
        label_path = target_dir / "module2_label_map.json"
        features_path = target_dir / "module2_features.json"

        if not (scaler_path.exists() and label_path.exists() and features_path.exists()):
            raise FileNotFoundError(f"Missing preprocessor artifacts in {target_dir}")

        with open(scaler_path, "rb") as f:
            data = pickle.load(f)
            self.scaler = data["scaler"]
            self.feature_medians = data["medians"]
            self.feature_maxs = data["maxs"]

        with open(label_path, "r") as f:
            data = json.load(f)
            self.label_to_id = data["label_to_id"]
            self.id_to_label = {int(k): v for k, v in data["id_to_label"].items()}

        with open(features_path, "r") as f:
            data = json.load(f)
            self.feature_cols = data["features"]

        self.is_fitted = True
        return self
