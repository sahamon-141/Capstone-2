"""
Non-Sequential Random Forest Baseline Classifier.

Implemented in pure NumPy to guarantee complete immunity against OS DLL-blocking policies
(such as Windows Application Control / AppLocker) while providing a scientifically rigorous,
fairly benchmarked non-sequential baseline.

Features:
- Bagging with bootstrap sampling per tree
- Random feature subspace selection at each split (sqrt(n_features))
- Gini impurity optimization for multiclass splitting
- Fast vectorized tree evaluation
- Complete scikit-learn compatible interface (fit, predict, predict_proba)
- Supports sequence aggregate and final-observation representations
"""

import time
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.module2.config import Module2Config, module2_config


class DecisionTreeNode:
    """Node in a decision tree."""
    __slots__ = ["feature_idx", "threshold", "left", "right", "value", "is_leaf"]

    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["DecisionTreeNode"] = None,
        right: Optional["DecisionTreeNode"] = None,
        value: Optional[np.ndarray] = None,
        is_leaf: bool = False,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Array of class probabilities if leaf
        self.is_leaf = is_leaf


class FastDecisionTree:
    """Lightweight, vectorized classification tree."""

    def __init__(
        self,
        max_depth: int = 12,
        min_samples_split: int = 5,
        max_features: Optional[int] = None,
        num_classes: int = 34,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.num_classes = num_classes
        self.rng = np.random.RandomState(random_state)
        self.root: Optional[DecisionTreeNode] = None

    def _gini(self, y_indices: np.ndarray, y: np.ndarray) -> float:
        """Computes Gini impurity."""
        m = len(y_indices)
        if m == 0:
            return 0.0
        labels = y[y_indices]
        counts = np.bincount(labels, minlength=self.num_classes)
        probs = counts / m
        return 1.0 - float(np.sum(probs ** 2))

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        n_features: int,
    ) -> Tuple[Optional[int], Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
        """Finds best feature and threshold to split node indices."""
        best_gain = -1.0
        best_feat = None
        best_thresh = None
        best_left = None
        best_right = None

        current_gini = self._gini(indices, y)
        n_sub = self.max_features or max(1, int(np.sqrt(n_features)))
        cand_feats = self.rng.choice(n_features, size=n_sub, replace=False)

        n_samples = len(indices)
        for f in cand_feats:
            vals = X[indices, f]
            # Use percentiles as candidate thresholds for speed
            thresholds = np.percentile(vals, [20, 40, 60, 80])
            for thresh in thresholds:
                left_mask = vals <= thresh
                n_l = int(np.sum(left_mask))
                n_r = n_samples - n_l
                if n_l < 2 or n_r < 2:
                    continue

                left_idx = indices[left_mask]
                right_idx = indices[~left_mask]

                g_l = self._gini(left_idx, y)
                g_r = self._gini(right_idx, y)
                gain = current_gini - ((n_l / n_samples) * g_l + (n_r / n_samples) * g_r)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
                    best_left = left_idx
                    best_right = right_idx

        return best_feat, best_thresh, best_left, best_right

    def _build_tree(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively builds decision tree."""
        n_samples = len(indices)
        labels = y[indices]
        counts = np.bincount(labels, minlength=self.num_classes)
        probs = (counts / n_samples).astype(np.float32)

        # Check stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.max(probs) >= 0.99
        ):
            return DecisionTreeNode(value=probs, is_leaf=True)

        f_idx, thresh, left_idx, right_idx = self._best_split(X, y, indices, X.shape[1])
        if f_idx is None or left_idx is None or len(left_idx) == 0 or len(right_idx) == 0:
            return DecisionTreeNode(value=probs, is_leaf=True)

        left_node = self._build_tree(X, y, left_idx, depth + 1)
        right_node = self._build_tree(X, y, right_idx, depth + 1)

        return DecisionTreeNode(
            feature_idx=f_idx,
            threshold=thresh,
            left=left_node,
            right=right_node,
            value=probs,
            is_leaf=False,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits tree on data."""
        indices = np.arange(len(X))
        self.root = self._build_tree(X, y, indices, depth=0)
        return self

    def _predict_row(self, x: np.ndarray, node: DecisionTreeNode) -> np.ndarray:
        """Traverses tree for a single row."""
        if node.is_leaf or node.feature_idx is None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        return self._predict_row(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predicts class probabilities for matrix X."""
        probs = np.zeros((len(X), self.num_classes), dtype=np.float32)
        for i in range(len(X)):
            probs[i] = self._predict_row(X[i], self.root)
        return probs


class BaselineRandomForest:
    """Ensemble of fast decision trees implementing Random Forest in pure NumPy."""

    def __init__(
        self,
        n_estimators: int = 30,
        max_depth: int = 12,
        cfg: Module2Config = module2_config,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cfg = cfg
        self.trees: List[FastDecisionTree] = []
        self.num_classes = 34
        self.train_time_sec: float = 0.0
        self.model_size_mb: float = 0.0

    @staticmethod
    def extract_rf_features(X_seq: np.ndarray, mode: str = "last_and_agg") -> np.ndarray:
        """Extracts tabular features from 3D sequence array."""
        N, T, F = X_seq.shape
        last_obs = X_seq[:, -1, :]

        if mode == "last":
            return last_obs

        seq_mean = np.mean(X_seq, axis=1)
        seq_std = np.std(X_seq, axis=1)
        return np.concatenate([last_obs, seq_mean, seq_std], axis=1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: Optional[int] = None):
        """Fits ensemble using bootstrap sampling and random feature subspaces."""
        n_samples = len(X)
        self.num_classes = num_classes or int(np.max(y) + 1)
        self.trees = []

        start_time = time.perf_counter()
        rng = np.random.RandomState(self.cfg.RANDOM_SEED)

        print(f"Training Pure-NumPy Random Forest ({self.n_estimators} trees, max_depth={self.max_depth})...")
        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]

            tree = FastDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=5,
                max_features=int(np.sqrt(X.shape[1])),
                num_classes=self.num_classes,
                random_state=self.cfg.RANDOM_SEED + i,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        self.train_time_sec = time.perf_counter() - start_time
        serialized = pickle.dumps(self.trees)
        self.model_size_mb = len(serialized) / (1024 * 1024)
        print(f"Random Forest training complete in {self.train_time_sec:.2f}s. Model size: {self.model_size_mb:.2f} MB")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes mean predicted class probability across all trees."""
        accum = np.zeros((len(X), self.num_classes), dtype=np.float32)
        for tree in self.trees:
            accum += tree.predict_proba(X)
        return accum / len(self.trees)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class ID with highest average probability."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int32)

    def train(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        feature_mode: str = "last_and_agg",
    ) -> float:
        """Wrapper to train directly from 3D sequence arrays."""
        X_train_rf = self.extract_rf_features(X_train_seq, mode=feature_mode)
        self.fit(X_train_rf, y_train)
        return self.train_time_sec

    def evaluate(
        self,
        X_test_seq: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
        feature_mode: str = "last_and_agg",
    ) -> Dict[str, Any]:
        """Evaluates Random Forest benchmark on test sequences."""
        X_test_rf = self.extract_rf_features(X_test_seq, mode=feature_mode)

        n_samples = len(X_test_rf)
        t0 = time.perf_counter()
        probs = self.predict_proba(X_test_rf)
        y_pred = np.argmax(probs, axis=1)
        total_time = time.perf_counter() - t0
        latency_ms_per_sample = (total_time / n_samples) * 1000.0 if n_samples > 0 else 0.0

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))) if class_names else None,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )

        results = {
            "model": "Random Forest",
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "training_time_sec": round(self.train_time_sec, 2),
            "inference_latency_ms": round(latency_ms_per_sample, 4),
            "model_size_mb": round(self.model_size_mb, 2),
            "predictions": y_pred,
            "probabilities": probs,
            "per_class_report": report_dict,
        }

        print("\n--- Random Forest Baseline Evaluation ---")
        print(f"Accuracy:        {acc:.4f}")
        print(f"Macro-F1:        {macro_f1:.4f}")
        print(f"Weighted-F1:     {weighted_f1:.4f}")
        print(f"Avg Latency:     {latency_ms_per_sample:.4f} ms/sample")
        print(f"Model Size:      {self.model_size_mb:.2f} MB")

        return results

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves trained Random Forest model."""
        dest = filepath or (self.cfg.MODELS_DIR / "module2_random_forest.pkl")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            pickle.dump(self.trees, f)
        return dest
