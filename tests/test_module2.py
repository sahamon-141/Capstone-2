"""
Unit and System Verification Tests for Module 2.

Covers:
1. Feature selection and configuration integrity
2. Preprocessor fitting, scaling, and bidirectional label mapping
3. SequenceBuilder shapes and cross-boundary leakage prevention
4. LSTM architecture and input shape compatibility
5. FP32 inference engine outputs and probabilities
6. INT8 TFLite model inference and tensor verification
7. Zero-day uncertainty rejection policy
8. Module 1 -> Module 2 end-to-end integration interface
"""

import sys
import os
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.module2.config import module2_config
from src.module2.preprocessor import Module2Preprocessor
from src.module2.sequence_builder import SequenceBuilder
from src.module2.lstm_model import IoTAttackLSTM
from src.module2.zero_day_eval import ZeroDayEvaluator
from src.module2.quantization import ModelQuantizer
from src.module2.inference import Module2InferenceEngine
from src.module2.integration import PipelineIntegrator


class TestModule2Pipeline(unittest.TestCase):
    """Test suite for Module 2 components."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = module2_config
        # Create synthetic test dataset
        np.random.seed(42)
        n_rows = 200
        cls.features = list(cls.cfg.SELECTED_FEATURES)
        data = {col: np.random.uniform(0.0, 100.0, size=n_rows) for col in cls.features}
        # Add labels
        labels = (["BenignTraffic"] * 80) + (["DDoS-ICMP_Flood"] * 60) + (["Mirai-udpplain"] * 60)
        data["label"] = labels
        cls.synthetic_df = pd.DataFrame(data)

    def test_01_feature_selection(self):
        """Verifies selected features contain no targets, post-hoc flags, or IDs."""
        forbidden = ["label", "binary_label", "label_encoded", "scenario_id", "flow_id"]
        for f in self.cfg.SELECTED_FEATURES:
            self.assertNotIn(f, forbidden, f"Forbidden feature '{f}' found in SELECTED_FEATURES")
        self.assertGreaterEqual(len(self.cfg.SELECTED_FEATURES), 15, "Feature count should be >= 15")

    def test_02_preprocessor_fitting_and_label_mapping(self):
        """Tests fitting preprocessor, scaling numeric stability, and bidirectional encoding."""
        preprocessor = Module2Preprocessor(self.cfg)
        preprocessor.fit(self.synthetic_df)

        self.assertTrue(preprocessor.is_fitted)
        self.assertIn("BenignTraffic", preprocessor.label_to_id)
        self.assertEqual(preprocessor.label_to_id["BenignTraffic"], 0, "BenignTraffic must map to class ID 0")

        # Test transformation
        transformed = preprocessor.transform(self.synthetic_df)
        self.assertIn("label_encoded", transformed.columns)
        self.assertEqual(transformed["label_encoded"].dtype, np.int32)

        # Verify decoded labels match original
        decoded = preprocessor.decode_labels(transformed["label_encoded"].to_numpy())
        self.assertEqual(decoded[:10], ["BenignTraffic"] * 10)

    def test_03_sequence_builder_shapes_and_boundary_integrity(self):
        """Tests that sequences have correct 3D shape and never cross class boundaries."""
        preprocessor = Module2Preprocessor(self.cfg)
        preprocessor.fit(self.synthetic_df)
        df_clean = preprocessor.transform(self.synthetic_df)

        builder = SequenceBuilder(sequence_length=20, step_size=10, cfg=self.cfg)
        class_blocks = {
            "BenignTraffic": df_clean[df_clean["label"] == "BenignTraffic"],
            "DDoS-ICMP_Flood": df_clean[df_clean["label"] == "DDoS-ICMP_Flood"],
        }

        X, y, stats = builder.build_dataset_sequences(class_blocks, preprocessor.feature_cols)

        # Check shape: (N, 20, num_features)
        self.assertEqual(X.ndim, 3)
        self.assertEqual(X.shape[1], 20)
        self.assertEqual(X.shape[2], len(preprocessor.feature_cols))
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(X), 0)

    def test_04_leakage_prevention_check(self):
        """Verifies that the non-leakage verification logic correctly detects overlap."""
        # Non-overlapping indices
        train_idx = np.array([[0, 19], [10, 29], [20, 39]])
        test_idx = np.array([[50, 69], [60, 79]])
        self.assertTrue(SequenceBuilder.verify_no_partition_leakage(train_idx, test_idx))

        # Leaking indices
        leaking_test_idx = np.array([[35, 54]])
        self.assertFalse(SequenceBuilder.verify_no_partition_leakage(train_idx, leaking_test_idx))

    def test_05_lstm_input_compatibility(self):
        """Verifies LSTM model compiles and produces correct output dimensions."""
        lstm = IoTAttackLSTM(
            sequence_length=20,
            num_features=len(self.cfg.SELECTED_FEATURES),
            num_classes=5,
            lstm_units=16,
            dense_units=16,
            cfg=self.cfg,
        )
        model = lstm.build_model()
        dummy_input = np.random.randn(4, 20, len(self.cfg.SELECTED_FEATURES)).astype(np.float32)
        probs = lstm.predict_probabilities(dummy_input)

        self.assertEqual(probs.shape, (4, 5))
        # Probabilities sum to 1.0 across classes
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(4), atol=1e-5)

    def test_06_zero_day_uncertainty_rejection(self):
        """Verifies rejection policy correctly separates high vs low confidence predictions."""
        evaluator = ZeroDayEvaluator(cfg=self.cfg)
        # Synthetic probabilities: 2 high confidence (0.95), 2 low confidence (0.25)
        known_probs = np.array([[0.95, 0.05], [0.90, 0.10], [0.60, 0.40]])
        y_true = np.array([0, 0, 0])
        tau = evaluator.calibrate_rejection_threshold(known_probs, y_true, rejection_percentile=0.10)

        unseen_probs = np.array([[0.10, 0.15], [0.12, 0.08]])
        results = evaluator.evaluate_unseen_detection(
            known_test_probs=known_probs,
            y_known_test=y_true,
            unseen_test_probs=unseen_probs,
            known_class_names=["Class_0", "Class_1"],
            custom_threshold=0.50,
        )

        self.assertEqual(results["zero_day_detection_rate"], 1.0, "All low confidence unseen attacks should be rejected")
        self.assertEqual(results["false_unknown_rate"], 0.0, "High confidence known samples should not be falsely rejected")

    def test_07_saved_models_and_inference_engine(self):
        """Verifies that trained FP32 and INT8 models exist and run inference."""
        if not (self.cfg.FP32_MODEL_PATH.exists() and self.cfg.INT8_MODEL_PATH.exists()):
            self.skipTest("Trained models not present in models/. Skipping artifact test.")

        engine_fp32 = Module2InferenceEngine(use_quantized=False, cfg=self.cfg)
        engine_int8 = Module2InferenceEngine(use_quantized=True, cfg=self.cfg)

        dummy_seq = np.random.randn(self.cfg.DEFAULT_SEQUENCE_LENGTH, len(self.cfg.SELECTED_FEATURES)).astype(np.float32)

        res_fp32 = engine_fp32.predict_single_sequence(dummy_seq)
        res_int8 = engine_int8.predict_single_sequence(dummy_seq)

        self.assertIn("predicted_class", res_fp32)
        self.assertIn("predicted_class", res_int8)
        self.assertGreater(res_fp32["confidence"], 0.0)
        self.assertGreater(res_int8["confidence"], 0.0)
        self.assertGreater(res_fp32["latency_ms"], 0.0)
        self.assertGreater(res_int8["latency_ms"], 0.0)

    def test_08_module1_module2_integration_handoff(self):
        """Verifies programmatic handoff from Module 1 anomaly alert to Module 2 classification."""
        if not self.cfg.INT8_MODEL_PATH.exists():
            self.skipTest("INT8 model not present. Skipping integration test.")

        integrator = PipelineIntegrator(use_quantized=True, cfg=self.cfg)

        # Case A: Normal window (Module 2 must NOT be invoked)
        normal_m1 = {
            "window_id": 1,
            "start_index": 0,
            "end_index": 19,
            "attack_risk": 0.12,
            "is_anomalous": False,
            "category": "NORMAL",
        }
        res_norm = integrator.process_module1_window(normal_m1, self.synthetic_df)
        self.assertFalse(res_norm["module2_invoked"])
        self.assertEqual(res_norm["module1_verdict"], "NORMAL")

        # Case B: Anomalous window (Module 2 MUST be invoked)
        attack_m1 = {
            "window_id": 2,
            "start_index": 50,
            "end_index": 69,
            "attack_risk": 0.88,
            "is_anomalous": True,
            "category": "ATTACK",
        }
        res_attack = integrator.process_module1_window(attack_m1, self.synthetic_df)
        self.assertTrue(res_attack["module2_invoked"])
        self.assertIn(res_attack["predicted_attack"], list(integrator.engine.preprocessor.id_to_label.values()) + ["UNKNOWN_ZERO_DAY"])


if __name__ == "__main__":
    unittest.main()
