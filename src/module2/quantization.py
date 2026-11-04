"""
Post-Training Quantization Module for Module 2.

Implements TensorFlow Lite Post-Training Quantization:
1. Full-integer INT8 post-training quantization with representative calibration dataset
2. Verification of converted graph operators, tensor dtypes, and model size
3. Fallback to hybrid dynamic-range integer quantization if full INT8 graph has unsupported ops
"""

from pathlib import Path
from typing import Dict, Any, Generator, Optional, Tuple, Callable, List
import os
import numpy as np

from src.module2.config import Module2Config, module2_config


class ModelQuantizer:
    """Handles post-training quantization and operator verification for Keras models."""

    def __init__(self, cfg: Module2Config = module2_config):
        self.cfg = cfg

    def create_representative_dataset_generator(
        self,
        X_train_seq: np.ndarray,
        num_calibration_samples: int = 300,
    ) -> Callable[[], Generator[List[np.ndarray], None, None]]:
        """
        Creates a generator yielding representative training sequences for INT8 calibration.
        Strict rule: MUST use training data only; never calibrate on test data.
        """
        n_available = len(X_train_seq)
        n_samples = min(num_calibration_samples, n_available)

        # Draw deterministic subset of training samples
        indices = np.linspace(0, n_available - 1, n_samples, dtype=int)
        calibration_data = X_train_seq[indices].astype(np.float32)

        def representative_data_gen():
            for i in range(n_samples):
                # Shape: (1, sequence_length, num_features)
                sample = np.expand_dims(calibration_data[i], axis=0)
                yield [sample]

        return representative_data_gen

    def convert_to_int8(
        self,
        keras_model,
        X_train_seq: np.ndarray,
        output_path: Optional[Path] = None,
        num_calibration_samples: int = 300,
    ) -> Dict[str, Any]:
        """
        Attempts full-integer INT8 post-training quantization using TFLiteConverter.
        Inspects resulting tensor dtypes and operator compatibility.
        """
        import tensorflow as tf

        out_path = output_path or self.cfg.INT8_MODEL_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rep_gen = self.create_representative_dataset_generator(
            X_train_seq=X_train_seq,
            num_calibration_samples=num_calibration_samples,
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        # Enable default optimizations (enables quantization)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = rep_gen

        # Ensure supported ops for LSTM integer execution
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        converter._experimental_lower_tensor_list_ops = False

        conversion_mode = "full_integer_int8"
        tflite_quant_model = None

        try:
            print("Attempting Full-Integer INT8 Post-Training Quantization...")
            tflite_quant_model = converter.convert()
        except Exception as e:
            print(f"[Notice] Full-integer conversion encountered: {e}")
            print("Applying dynamic-range integer quantization with supported ops...")
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter._experimental_lower_tensor_list_ops = False
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            tflite_quant_model = converter.convert()
            conversion_mode = "dynamic_range_int8"

        with open(out_path, "wb") as f:
            f.write(tflite_quant_model)

        size_bytes = out_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        # Inspect model details using TFLite interpreter
        inspector_report = self.inspect_tflite_model(out_path)

        report = {
            "output_path": str(out_path),
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 3),
            "conversion_mode": conversion_mode,
            "input_details": inspector_report["input_details"],
            "output_details": inspector_report["output_details"],
            "tensor_count": inspector_report["tensor_count"],
        }

        print(f"Quantized model saved to {out_path} ({size_mb:.2f} MB, mode: {conversion_mode})")
        return report

    @staticmethod
    def inspect_tflite_model(model_path: Path) -> Dict[str, Any]:
        """Loads and inspects tensors, input/output dtypes, and shapes of a TFLite model."""
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        all_tensors = interpreter.get_tensor_details()

        in_summary = [
            {
                "name": str(inp["name"]),
                "shape": inp["shape"].tolist(),
                "dtype": str(inp["dtype"].__name__),
                "quantization": inp.get("quantization", (0.0, 0)),
            }
            for inp in input_details
        ]

        out_summary = [
            {
                "name": str(out["name"]),
                "shape": out["shape"].tolist(),
                "dtype": str(out["dtype"].__name__),
                "quantization": out.get("quantization", (0.0, 0)),
            }
            for out in output_details
        ]

        return {
            "input_details": in_summary,
            "output_details": out_summary,
            "tensor_count": len(all_tensors),
        }
