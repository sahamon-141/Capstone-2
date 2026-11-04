"""
Zero-Day / Unseen Attack Evaluation Module for Module 2.

Evaluates supervised multiclass LSTM when encountering unseen/held-out exploits.
Distinguishes between:
1. Classification of known classes
2. Detection/rejection of unknown attacks via Maximum Softmax Probability (MSP) policy

Methodology:
- Held-out attack family (Mirai botnet) is completely excluded from training.
- Rejection threshold tau is calibrated strictly on validation data:
      tau = percentile_p(max_softmax_prob of correct validation predictions)
- Test samples with max_softmax_prob < tau are flagged as "UNKNOWN / ZERO-DAY".
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.module2.config import Module2Config, module2_config


class ZeroDayEvaluator:
    """Evaluates novel exploit detection and uncertainty rejection policies."""

    def __init__(
        self,
        held_out_classes: Optional[List[str]] = None,
        cfg: Module2Config = module2_config,
    ):
        self.cfg = cfg
        self.held_out_classes = held_out_classes or list(self.cfg.HELD_OUT_ATTACKS)
        self.rejection_threshold: float = 0.50

    def calibrate_rejection_threshold(
        self,
        val_probs: np.ndarray,
        y_val_true: np.ndarray,
        rejection_percentile: float = 0.10,
    ) -> float:
        """
        Calibrates uncertainty threshold strictly on validation data.
        Finds the lower percentile of max softmax probabilities on correctly classified validation samples.
        """
        val_preds = np.argmax(val_probs, axis=1)
        max_probs = np.max(val_probs, axis=1)

        # Look at correctly classified validation samples
        correct_mask = val_preds == y_val_true
        if np.sum(correct_mask) > 0:
            correct_max_probs = max_probs[correct_mask]
            # Threshold at specified percentile (e.g. 10th percentile gives ~10% false reject rate on known traffic)
            self.rejection_threshold = float(np.percentile(correct_max_probs, rejection_percentile * 100))
        else:
            self.rejection_threshold = 0.50

        print(f"[Calibration] Zero-Day Softmax Rejection Threshold set to {self.rejection_threshold:.4f} "
              f"(calibrated on {len(val_probs):,} validation samples)")
        return self.rejection_threshold

    def evaluate_unseen_detection(
        self,
        known_test_probs: np.ndarray,
        y_known_test: np.ndarray,
        unseen_test_probs: np.ndarray,
        known_class_names: List[str],
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates known vs unseen traffic separation under the uncertainty rejection policy.
        
        Args:
            known_test_probs: Model softmax outputs for test samples from known classes.
            y_known_test: Ground truth integer class IDs for known test samples.
            unseen_test_probs: Model softmax outputs for held-out attack samples.
            known_class_names: List of known class names.
            custom_threshold: Optional override for rejection threshold.
        """
        tau = custom_threshold if custom_threshold is not None else self.rejection_threshold

        # 1. Performance on Known Traffic
        known_max_probs = np.max(known_test_probs, axis=1)
        known_raw_preds = np.argmax(known_test_probs, axis=1)

        # Flagged as unknown if max_prob < tau
        known_flagged_unknown = known_max_probs < tau
        false_unknown_rate = float(np.mean(known_flagged_unknown))

        # Known accuracy on unrejected samples
        accepted_mask = ~known_flagged_unknown
        accepted_acc = float(accuracy_score(y_known_test[accepted_mask], known_raw_preds[accepted_mask])) if np.sum(accepted_mask) > 0 else 0.0
        overall_known_f1 = float(f1_score(y_known_test, known_raw_preds, average="macro", zero_division=0))

        # 2. Performance on Unseen / Held-Out Traffic
        unseen_max_probs = np.max(unseen_test_probs, axis=1)
        unseen_raw_preds = np.argmax(unseen_test_probs, axis=1)

        # Correct detection of zero-day occurs when max_prob < tau (rejected as unknown)
        unseen_detected = unseen_max_probs < tau
        zero_day_detection_rate = float(np.mean(unseen_detected))

        # What classes do unseen attacks get misattributed to if NOT rejected?
        unseen_misattributions = {}
        for p in unseen_raw_preds[~unseen_detected]:
            cls_name = known_class_names[p] if p < len(known_class_names) else f"Class_{p}"
            unseen_misattributions[cls_name] = unseen_misattributions.get(cls_name, 0) + 1

        results = {
            "rejection_threshold": round(tau, 4),
            "known_test_samples": len(known_test_probs),
            "unseen_test_samples": len(unseen_test_probs),
            "known_macro_f1": round(overall_known_f1, 4),
            "known_accepted_accuracy": round(accepted_acc, 4),
            "false_unknown_rate": round(false_unknown_rate, 4),
            "zero_day_detection_rate": round(zero_day_detection_rate, 4),
            "mean_confidence_known": round(float(np.mean(known_max_probs)), 4),
            "mean_confidence_unseen": round(float(np.mean(unseen_max_probs)), 4),
            "unseen_misattributions_when_accepted": unseen_misattributions,
        }

        print("\n" + "=" * 60)
        print("ZERO-DAY / UNSEEN ATTACK EXPERIMENT RESULTS")
        print("=" * 60)
        print(f"Held-Out Attack Family:          {self.held_out_classes}")
        print(f"Rejection Threshold (tau):       {tau:.4f}")
        print(f"Known Traffic Macro-F1:          {overall_known_f1:.4f}")
        print(f"False Unknown Rate (Known FP):   {false_unknown_rate * 100:.2f}%")
        print(f"Zero-Day Detection Rate (Recall):{zero_day_detection_rate * 100:.2f}%")
        print(f"Mean Confidence on Known:        {results['mean_confidence_known']:.4f}")
        print(f"Mean Confidence on Unseen:       {results['mean_confidence_unseen']:.4f}")
        print("=" * 60)

        return results
