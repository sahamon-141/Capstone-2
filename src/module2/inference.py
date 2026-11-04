"""
Inference Engine and Deployment Benchmark for Module 2.

Supports:
1. High-throughput FP32 Keras model inference
2. Lightweight, edge-optimized INT8 TFLite model inference (ideal for Raspberry Pi)
3. Precision agreement, average latency, and P95 latency benchmarking
4. Zero-Day uncertainty rejection flagging
"""

import time
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np

from src.module2.config import Module2Config, module2_config
from src.module2.preprocessor import Module2Preprocessor


class Module2InferenceEngine:
    """Unified inference engine for FP32 and INT8 IoT attack classification."""

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_quantized: bool = False,
        rejection_threshold: float = 0.50,
        cfg: Module2Config = module2_config,
    ):
        self.cfg = cfg
        self.model_dir = model_dir or self.cfg.MODELS_DIR
        self.use_quantized = use_quantized
        self.rejection_threshold = rejection_threshold

        self.preprocessor = Module2Preprocessor(self.cfg)
        self.preprocessor.load_artifacts(self.model_dir)

        self.keras_model = None
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None

        self._load_model()

    def _load_model(self) -> None:
        """Loads appropriate model runtime based on quantization flag."""
        if self.use_quantized:
            tflite_path = self.model_dir / "module2_lstm_int8.tflite"
            if not tflite_path.exists():
                raise FileNotFoundError(f"INT8 model not found at {tflite_path}")

            # Prefer lightweight LiteRT runtime to avoid heavy/fragile TF DLLs
            try:
                import ai_edge_litert.interpreter as litert
                self.tflite_interpreter = litert.Interpreter(model_path=str(tflite_path))
            except ImportError:
                import tensorflow as tf
                self.tflite_interpreter = tf.lite.Interpreter(model_path=str(tflite_path))

            self.tflite_interpreter.allocate_tensors()
            self.tflite_input_details = self.tflite_interpreter.get_input_details()
            self.tflite_output_details = self.tflite_interpreter.get_output_details()
            print(f"[InferenceEngine] Loaded INT8 TFLite model from {tflite_path.name}")
        else:
            keras_path = self.model_dir / "module2_lstm_fp32.keras"
            if not keras_path.exists():
                raise FileNotFoundError(f"FP32 model not found at {keras_path}")

            try:
                import tensorflow as tf
                self.keras_model = tf.keras.models.load_model(str(keras_path))
                print(f"[InferenceEngine] Loaded FP32 Keras model from {keras_path.name}")
            except Exception as e:
                print(f"[InferenceEngine] Keras FP32 runtime unavailable on this host: {e}")
                self.keras_model = None


    def predict_single_sequence(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on a single sequence of shape (sequence_length, num_features).
        Returns predicted class, confidence, latency, and rejection status.
        """
        if sequence.ndim == 2:
            batch = np.expand_dims(sequence, axis=0).astype(np.float32)
        else:
            batch = sequence.astype(np.float32)

        start_time = time.perf_counter()

        if self.use_quantized:
            inp_detail = self.tflite_input_details[0]
            out_detail = self.tflite_output_details[0]

            # Check if model expects quantized int8 input or float32
            if inp_detail["dtype"] == np.int8:
                scale, zero_point = inp_detail["quantization"]
                quant_input = (batch / scale + zero_point).astype(np.int8)
                self.tflite_interpreter.set_tensor(inp_detail["index"], quant_input)
            else:
                self.tflite_interpreter.set_tensor(inp_detail["index"], batch)

            self.tflite_interpreter.invoke()

            raw_out = self.tflite_interpreter.get_tensor(out_detail["index"])
            if out_detail["dtype"] == np.int8:
                scale, zero_point = out_detail["quantization"]
                probs = (raw_out.astype(np.float32) - zero_point) * scale
            else:
                probs = raw_out.astype(np.float32)
        else:
            probs = self.keras_model.predict(batch, verbose=0)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        p = probs[0]
        # Softmax normalization if needed
        p_sum = np.sum(p)
        if p_sum > 0:
            p = p / p_sum

        pred_id = int(np.argmax(p))
        confidence = float(p[pred_id])
        class_name = self.preprocessor.id_to_label.get(pred_id, f"Class_{pred_id}")

        # Check uncertainty rejection policy
        is_unknown = confidence < self.rejection_threshold

        return {
            "predicted_class": "UNKNOWN_ZERO_DAY" if is_unknown else class_name,
            "raw_class": class_name,
            "class_id": pred_id,
            "confidence": round(confidence, 4),
            "is_uncertain_rejection": is_unknown,
            "probabilities": p.tolist(),
            "latency_ms": round(latency_ms, 3),
            "runtime": "INT8_TFLite" if self.use_quantized else "FP32_Keras",
        }

    def benchmark_batch(
        self,
        X_test_seq: np.ndarray,
        y_test: Optional[np.ndarray] = None,
        warmup_runs: int = 10,
    ) -> Dict[str, Any]:
        """
        Runs detailed latency, accuracy, and P95 benchmark over a test batch.
        """
        n_samples = len(X_test_seq)
        if n_samples == 0:
            return {"num_samples": 0}

        # Warmup
        warmup_n = min(warmup_runs, n_samples)
        for i in range(warmup_n):
            _ = self.predict_single_sequence(X_test_seq[i])

        latencies = []
        predictions = []
        confidences = []

        for i in range(n_samples):
            t0 = time.perf_counter()
            res = self.predict_single_sequence(X_test_seq[i])
            latencies.append((time.perf_counter() - t0) * 1000.0)
            predictions.append(res["class_id"])
            confidences.append(res["confidence"])

        lat_arr = np.array(latencies)
        preds_arr = np.array(predictions)

        results = {
            "runtime": "INT8_TFLite" if self.use_quantized else "FP32_Keras",
            "num_samples": n_samples,
            "mean_latency_ms": round(float(np.mean(lat_arr)), 3),
            "median_latency_ms": round(float(np.median(lat_arr)), 3),
            "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 3),
            "min_latency_ms": round(float(np.min(lat_arr)), 3),
            "max_latency_ms": round(float(np.max(lat_arr)), 3),
            "mean_confidence": round(float(np.mean(confidences)), 4),
            "predictions": preds_arr,
        }

        if y_test is not None:
            acc = float(np.mean(preds_arr == y_test))
            results["accuracy"] = round(acc, 4)

        return results
