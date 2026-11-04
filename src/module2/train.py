"""
Module 2 End-to-End Training and Quantization Orchestration Script.

Executes the complete pipeline:
1. Feature extraction and class-stratified sampling
2. Held-out Zero-Day attack family quarantine (Mirai botnet)
3. Leakage-safe preprocessor fitting (StandardScaler + LabelEncoder)
4. Non-crossing sequence generation (L=50)
5. FP32 LSTM model training with early stopping and checkpointing
6. Non-sequential Random Forest baseline training and benchmarking
7. Zero-day held-out exploit evaluation & uncertainty rejection
8. Full-integer INT8 post-training quantization (TFLite)
9. FP32 vs INT8 inference latency and prediction agreement evaluation
10. Metric serialization and diagnostic figure generation
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.module2.config import Module2Config, module2_config
from src.module2.preprocessor import Module2Preprocessor
from src.module2.sequence_builder import SequenceBuilder
from src.module2.lstm_model import IoTAttackLSTM
from src.module2.baseline_rf import BaselineRandomForest
from src.module2.zero_day_eval import ZeroDayEvaluator
from src.module2.quantization import ModelQuantizer
from src.module2.inference import Module2InferenceEngine
from src.module2.evaluate import Module2Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Module 2 LSTM Training & Quantization Pipeline")
    parser.add_argument("--samples-per-class", type=int, default=1500, help="Max flows per class for training (default: 1500)")
    parser.add_argument("--seq-len", type=int, default=50, help="Sequence length in observations (default: 50)")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--held-out", type=str, default="Mirai-greeth_flood,Mirai-udpplain,Mirai-greip_flood", help="Held-out attack classes")
    return parser.parse_args()


def load_stratified_scenario(
    cfg: Module2Config,
    samples_per_class: int,
    held_out_classes: list,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Loads records from train.csv and test.csv in a structured, class-separated manner.
    Guarantees that held-out attacks are strictly quarantined into a dedicated evaluation bucket.
    """
    train_path = cfg.DATASET_DIR / cfg.TRAIN_FILENAME
    test_path = cfg.DATASET_DIR / cfg.TEST_FILENAME
    cols = cfg.SELECTED_FEATURES + ["label"]

    print(f"Loading class-separated flow records from {train_path.name} (target: {samples_per_class:,} per class)...")

    # Accumulate rows per class from train.csv
    train_class_buckets: Dict[str, List[pd.DataFrame]] = {}
    total_needed = samples_per_class

    for chunk in pd.read_csv(train_path, usecols=cols, chunksize=100000, low_memory=False):
        chunk["label_clean"] = chunk["label"].astype(str).str.strip()
        for c_name, c_df in chunk.groupby("label_clean"):
            if c_name not in train_class_buckets:
                train_class_buckets[c_name] = []
            current_count = sum(len(x) for x in train_class_buckets[c_name])
            if current_count < total_needed:
                train_class_buckets[c_name].append(c_df.drop(columns=["label_clean"]))

        # Check if all classes have enough records
        all_full = all(sum(len(x) for x in train_class_buckets[c]) >= total_needed for c in train_class_buckets)
        if len(train_class_buckets) >= 30 and all_full:
            break

    # Assemble concatenated DataFrames per class
    train_dict = {}
    val_dict = {}
    test_dict = {}
    unseen_dict = {}

    for c_name, df_list in train_class_buckets.items():
        combined = pd.concat(df_list, ignore_index=True)
        if len(combined) > total_needed:
            combined = combined.iloc[:total_needed]

        # If class is held out for zero-day evaluation, place strictly into unseen_dict
        if c_name in held_out_classes:
            unseen_dict[c_name] = combined
            continue

        # Split into train, val, test guaranteeing sufficient records for sequence creation
        n = len(combined)
        min_tail = cfg.DEFAULT_SEQUENCE_LENGTH + 15
        if n >= min_tail * 2 + 50:
            n_val = max(min_tail, int(n * 0.15))
            n_test = max(min_tail, int(n * 0.15))
            n_train = n - n_val - n_test
        else:
            n_train = max(cfg.DEFAULT_SEQUENCE_LENGTH, int(n * 0.60))
            n_val = max(cfg.DEFAULT_SEQUENCE_LENGTH, int(n * 0.20))
            n_test = max(0, n - n_train - n_val)

        train_dict[c_name] = combined.iloc[:n_train].copy()
        val_dict[c_name] = combined.iloc[n_train : n_train + n_val].copy()
        test_dict[c_name] = combined.iloc[n_train + n_val :].copy()

    print(f"Loaded {len(train_dict)} development classes ({sum(len(d) for d in train_dict.values()):,} train flows)")
    print(f"Quarantined {len(unseen_dict)} held-out classes ({sum(len(d) for d in unseen_dict.values()):,} unseen flows)")

    return train_dict, val_dict, test_dict, unseen_dict


def main():
    args = parse_args()
    cfg = module2_config
    cfg.DEFAULT_SEQUENCE_LENGTH = args.seq_len
    cfg.MAX_EPOCHS = args.epochs
    cfg.BATCH_SIZE = args.batch_size
    cfg.HELD_OUT_ATTACKS = [a.strip() for a in args.held_out.split(",") if a.strip()]
    cfg.ensure_directories()

    print("\n" + "=" * 80)
    print("      MODULE 2: LSTM-BASED IOT MULTICLASS ATTACK CLASSIFICATION")
    print("                 WITH INT8 QUANTIZATION BENCHMARK")
    print("=" * 80)
    print(f"Sequence Length:    {cfg.DEFAULT_SEQUENCE_LENGTH} observations")
    print(f"Features Count:     {len(cfg.SELECTED_FEATURES)}")
    print(f"Max Epochs:         {cfg.MAX_EPOCHS}")
    print(f"Batch Size:         {cfg.BATCH_SIZE}")
    print(f"Held-Out Attacks:   {cfg.HELD_OUT_ATTACKS}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # STEP 1: Load Data & Quarantined Partitions
    # --------------------------------------------------------------------------
    train_dict, val_dict, test_dict, unseen_dict = load_stratified_scenario(
        cfg=cfg,
        samples_per_class=args.samples_per_class,
        held_out_classes=cfg.HELD_OUT_ATTACKS,
    )

    # --------------------------------------------------------------------------
    # STEP 2: Fit Preprocessor (Strictly on Training Data)
    # --------------------------------------------------------------------------
    preprocessor = Module2Preprocessor(cfg)
    df_all_train = pd.concat(list(train_dict.values()), ignore_index=True)
    preprocessor.fit(df_all_train)
    preprocessor.save_artifacts()

    class_names = [preprocessor.id_to_label[i] for i in range(len(preprocessor.id_to_label))]
    num_classes = len(class_names)
    print(f"Target classes to classify ({num_classes}): {class_names}")

    # Transform all partitions
    print("Applying learned scaling and encoding across partitions...")
    for d in [train_dict, val_dict, test_dict, unseen_dict]:
        for k in d:
            d[k] = preprocessor.transform(d[k])

    # --------------------------------------------------------------------------
    # STEP 3: Construct Boundary-Safe Sequences
    # --------------------------------------------------------------------------
    seq_builder = SequenceBuilder(
        sequence_length=cfg.DEFAULT_SEQUENCE_LENGTH,
        step_size=cfg.SEQUENCE_STEP,
        label_strategy="last",
        strict_homogeneous=True,
        cfg=cfg,
    )

    print("Generating sequences within class blocks (no cross-boundary leakage)...")
    val_seq_builder = SequenceBuilder(
        sequence_length=cfg.DEFAULT_SEQUENCE_LENGTH,
        step_size=15,
        label_strategy="last",
        strict_homogeneous=True,
        cfg=cfg,
    )
    X_train, y_train, stats_train = seq_builder.build_dataset_sequences(train_dict, preprocessor.feature_cols)
    X_val, y_val, stats_val = val_seq_builder.build_dataset_sequences(val_dict, preprocessor.feature_cols)
    X_test, y_test, stats_test = val_seq_builder.build_dataset_sequences(test_dict, preprocessor.feature_cols)
    X_unseen, y_unseen, stats_unseen = val_seq_builder.build_dataset_sequences(unseen_dict, preprocessor.feature_cols)

    print(f"Constructed Sequences:")
    print(f"  - Train:   X={X_train.shape}, y={y_train.shape}")
    print(f"  - Val:     X={X_val.shape}, y={y_val.shape}")
    print(f"  - Test:    X={X_test.shape}, y={y_test.shape}")
    print(f"  - Unseen:  X={X_unseen.shape}, y={y_unseen.shape}")

    # --------------------------------------------------------------------------
    # STEP 4: Build & Train FP32 LSTM
    # --------------------------------------------------------------------------
    lstm = IoTAttackLSTM(
        sequence_length=cfg.DEFAULT_SEQUENCE_LENGTH,
        num_features=len(preprocessor.feature_cols),
        num_classes=num_classes,
        lstm_units=cfg.LSTM_UNITS,
        dense_units=cfg.DENSE_UNITS,
        dropout_rate=cfg.DROPOUT_RATE,
        learning_rate=cfg.LEARNING_RATE,
        cfg=cfg,
    )
    print("\nModel Architecture Summary:")
    print(lstm.get_summary_string())

    t_train_start = time.perf_counter()
    train_history = lstm.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=cfg.BATCH_SIZE,
        epochs=cfg.MAX_EPOCHS,
        save_best=True,
    )
    lstm_train_time = time.perf_counter() - t_train_start
    lstm.plot_training_curves()

    # --------------------------------------------------------------------------
    # STEP 5: Train Random Forest Baseline
    # --------------------------------------------------------------------------
    print("\n--- Training Non-Sequential Random Forest Baseline ---")
    rf_baseline = BaselineRandomForest(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        cfg=cfg,
    )
    rf_baseline.train(X_train, y_train, feature_mode="last_and_agg")
    rf_baseline.save()
    rf_results = rf_baseline.evaluate(X_test, y_test, class_names=class_names)

    # --------------------------------------------------------------------------
    # STEP 6: Zero-Day / Held-Out Attack Evaluation
    # --------------------------------------------------------------------------
    zero_day_eval = ZeroDayEvaluator(held_out_classes=cfg.HELD_OUT_ATTACKS, cfg=cfg)

    # Softmax probabilities on validation data to calibrate rejection threshold
    val_probs = lstm.predict_probabilities(X_val)
    rejection_tau = zero_day_eval.calibrate_rejection_threshold(val_probs, y_val, rejection_percentile=0.10)

    # Evaluate on known test vs unseen test sequences
    known_test_probs = lstm.predict_probabilities(X_test)
    unseen_test_probs = lstm.predict_probabilities(X_unseen) if len(X_unseen) > 0 else np.empty((0, num_classes))

    zero_day_results = zero_day_eval.evaluate_unseen_detection(
        known_test_probs=known_test_probs,
        y_known_test=y_test,
        unseen_test_probs=unseen_test_probs,
        known_class_names=class_names,
    )

    # --------------------------------------------------------------------------
    # STEP 7: Post-Training INT8 Quantization
    # --------------------------------------------------------------------------
    print("\n--- Executing Post-Training Quantization (TFLite INT8) ---")
    quantizer = ModelQuantizer(cfg=cfg)
    quant_report = quantizer.convert_to_int8(
        keras_model=lstm.model,
        X_train_seq=X_train,
        output_path=cfg.INT8_MODEL_PATH,
        num_calibration_samples=300,
    )

    # --------------------------------------------------------------------------
    # STEP 8: Benchmark FP32 vs INT8 Inference
    # --------------------------------------------------------------------------
    print("\n--- Running Latency & Agreement Benchmark (FP32 vs INT8) ---")
    fp32_engine = Module2InferenceEngine(use_quantized=False, rejection_threshold=rejection_tau, cfg=cfg)
    int8_engine = Module2InferenceEngine(use_quantized=True, rejection_threshold=rejection_tau, cfg=cfg)

    fp32_bench = fp32_engine.benchmark_batch(X_test, y_test=y_test)
    int8_bench = int8_engine.benchmark_batch(X_test, y_test=y_test)

    evaluator = Module2Evaluator(cfg=cfg)
    agreement = evaluator.compute_agreement(fp32_bench["predictions"], int8_bench["predictions"])

    fp32_eval = evaluator.evaluate_predictions(y_test, fp32_bench["predictions"], "FP32 LSTM", class_names)
    int8_eval = evaluator.evaluate_predictions(y_test, int8_bench["predictions"], "INT8 LSTM", class_names)

    # Plot Confusion Matrices
    evaluator.plot_confusion_matrix(
        np.array(fp32_eval["confusion_matrix"]),
        class_names,
        title="Module 2: FP32 LSTM Confusion Matrix",
        filename="module2_confusion_matrix_fp32.png",
    )
    evaluator.plot_confusion_matrix(
        np.array(int8_eval["confusion_matrix"]),
        class_names,
        title="Module 2: INT8 LSTM Confusion Matrix",
        filename="module2_confusion_matrix_int8.png",
    )

    # Model File Sizes in MB
    fp32_size_mb = round(cfg.FP32_MODEL_PATH.stat().st_size / (1024 * 1024), 2)
    int8_size_mb = round(cfg.INT8_MODEL_PATH.stat().st_size / (1024 * 1024), 2)
    size_reduction_pct = round(((fp32_size_mb - int8_size_mb) / fp32_size_mb) * 100, 1)

    # --------------------------------------------------------------------------
    # STEP 9: Generate Comparative Results Table
    # --------------------------------------------------------------------------
    comparison_records = [
        {
            "Model": "Random Forest",
            "Accuracy": rf_results["accuracy"],
            "Macro-F1": rf_results["macro_f1"],
            "Weighted-F1": rf_results["weighted_f1"],
            "Latency (ms)": rf_results["inference_latency_ms"],
            "Model Size (MB)": rf_results["model_size_mb"],
        },
        {
            "Model": "FP32 LSTM",
            "Accuracy": fp32_eval["accuracy"],
            "Macro-F1": fp32_eval["macro_f1"],
            "Weighted-F1": fp32_eval["weighted_f1"],
            "Latency (ms)": fp32_bench["mean_latency_ms"],
            "Model Size (MB)": fp32_size_mb,
        },
        {
            "Model": "INT8 LSTM",
            "Accuracy": int8_eval["accuracy"],
            "Macro-F1": int8_eval["macro_f1"],
            "Weighted-F1": int8_eval["weighted_f1"],
            "Latency (ms)": int8_bench["mean_latency_ms"],
            "Model Size (MB)": int8_size_mb,
        },
    ]

    comparison_df = pd.DataFrame(comparison_records)
    evaluator.plot_model_comparison(comparison_df)

    print("\n" + "=" * 80)
    print("                    FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print("=" * 80)
    print(f"FP32 vs INT8 Prediction Agreement:  {agreement * 100:.2f}%")
    print(f"INT8 Model Compression:             {size_reduction_pct}% size reduction")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # STEP 10: Serialize Comprehensive Metrics JSON
    # --------------------------------------------------------------------------
    final_metrics = {
        "pipeline": "Module 2: Multiclass Attack Classification with INT8 Quantization",
        "dataset": "CICIoT2023",
        "configuration": {
            "sequence_length": cfg.DEFAULT_SEQUENCE_LENGTH,
            "features": preprocessor.feature_cols,
            "classes": class_names,
            "held_out_classes": cfg.HELD_OUT_ATTACKS,
            "rejection_threshold": rejection_tau,
        },
        "dataset_statistics": {
            "num_train_sequences": len(X_train),
            "num_val_sequences": len(X_val),
            "num_test_sequences": len(X_test),
            "num_unseen_sequences": len(X_unseen),
        },
        "model_comparison": comparison_records,
        "quantization_summary": {
            "fp32_size_mb": fp32_size_mb,
            "int8_size_mb": int8_size_mb,
            "size_reduction_pct": size_reduction_pct,
            "prediction_agreement": round(agreement, 4),
            "fp32_p95_latency_ms": fp32_bench["p95_latency_ms"],
            "int8_p95_latency_ms": int8_bench["p95_latency_ms"],
            "conversion_mode": quant_report["conversion_mode"],
        },
        "zero_day_evaluation": zero_day_results,
        "random_forest_baseline": {
            "training_time_sec": rf_results["training_time_sec"],
            "model_size_mb": rf_results["model_size_mb"],
        },
    }

    metrics_path = evaluator.save_metrics_report(final_metrics)
    print(f"Execution successfully completed. Final report saved to: {metrics_path}")


if __name__ == "__main__":
    main()
