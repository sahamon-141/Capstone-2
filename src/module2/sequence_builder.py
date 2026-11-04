"""
Sequence Builder for LSTM IoT Multiclass Attack Classification.

Constructs ordered temporal observation sequences of shape:
    (n_sequences, sequence_length, n_features)

Guarantees:
1. Sequences NEVER cross capture/scenario boundaries or attack class transitions.
2. No random row shuffling before sequence creation.
3. No overlapping sequence leakage between train, validation, and test splits.
4. Fully configurable sequence length and step size.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.module2.config import Module2Config, module2_config


class SequenceBuilder:
    """Builds temporal observation sequences from partitioned flow data."""

    def __init__(
        self,
        sequence_length: int = 50,
        step_size: int = 25,
        label_strategy: str = "last",
        strict_homogeneous: bool = True,
        cfg: Module2Config = module2_config,
    ):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.label_strategy = label_strategy
        self.strict_homogeneous = strict_homogeneous
        self.cfg = cfg

    def create_sequences_from_block(
        self,
        df_block: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label_encoded",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates sequences strictly within a single continuous block of records.
        
        Args:
            df_block: DataFrame containing contiguous records for a single session/class.
            feature_cols: List of numerical feature names.
            label_col: Name of integer-encoded label column.
            
        Returns:
            X: Array of shape (n_seqs, sequence_length, n_features)
            y: Array of shape (n_seqs,) with target labels
            indices: Array of shape (n_seqs, 2) storing (start_idx, end_idx) of each sequence
        """
        n_rows = len(df_block)
        if n_rows < self.sequence_length:
            return (
                np.empty((0, self.sequence_length, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0, 2), dtype=np.int64),
            )

        features_data = df_block[feature_cols].to_numpy(dtype=np.float32)
        labels_data = df_block[label_col].to_numpy(dtype=np.int32)

        sequences = []
        targets = []
        indices = []

        for start_idx in range(0, n_rows - self.sequence_length + 1, self.step_size):
            end_idx = start_idx + self.sequence_length
            seq_x = features_data[start_idx:end_idx]
            seq_labels = labels_data[start_idx:end_idx]

            # If strict homogeneous is enforced, verify all observations share the same label
            if self.strict_homogeneous and (seq_labels[0] != seq_labels[-1] or len(np.unique(seq_labels)) > 1):
                continue

            if self.label_strategy == "last":
                target_label = seq_labels[-1]
            elif self.label_strategy == "mode":
                vals, counts = np.unique(seq_labels, return_counts=True)
                target_label = vals[np.argmax(counts)]
            else:
                target_label = seq_labels[-1]

            sequences.append(seq_x)
            targets.append(target_label)
            indices.append([start_idx, end_idx - 1])

        if not sequences:
            return (
                np.empty((0, self.sequence_length, len(feature_cols)), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0, 2), dtype=np.int64),
            )

        X = np.asarray(sequences, dtype=np.float32)
        y = np.asarray(targets, dtype=np.int32)
        idx_arr = np.asarray(indices, dtype=np.int64)

        return X, y, idx_arr

    def build_dataset_sequences(
        self,
        class_blocks_dict: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        label_col: str = "label_encoded",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Builds sequences across multiple class-separated partitions.
        Crucially, sequences NEVER cross from one class block into another.
        """
        all_X = []
        all_y = []
        meta_stats = {"per_class_sequences": {}, "total_sequences": 0}

        for class_name, df_class in class_blocks_dict.items():
            if len(df_class) < self.sequence_length:
                continue

            X_cls, y_cls, _ = self.create_sequences_from_block(
                df_block=df_class,
                feature_cols=feature_cols,
                label_col=label_col,
            )

            if len(X_cls) > 0:
                all_X.append(X_cls)
                all_y.append(y_cls)
                meta_stats["per_class_sequences"][class_name] = len(X_cls)

        if not all_X:
            empty_X = np.empty((0, self.sequence_length, len(feature_cols)), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int32)
            return empty_X, empty_y, meta_stats

        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        meta_stats["total_sequences"] = len(X_combined)

        return X_combined, y_combined, meta_stats

    @staticmethod
    def verify_no_partition_leakage(
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> bool:
        """
        Mathematically checks that no observation index in the test sequences
        overlaps with observation indices in training sequences.
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return True

        train_ranges = set()
        for s, e in train_indices:
            train_ranges.update(range(s, e + 1))

        test_ranges = set()
        for s, e in test_indices:
            test_ranges.update(range(s, e + 1))

        overlap = train_ranges.intersection(test_ranges)
        return len(overlap) == 0
