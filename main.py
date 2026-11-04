"""
Main Orchestration Script for IAT-Based Fuzzy Logic IoT Network Anomaly Detection.

Executes the complete end-to-end research pipeline:
1. Dataset inspection & structure verification
2. Preprocessing & label analysis
3. Pure normal baseline establishment (anti-leakage)
4. Sliding window generation & statistical feature engineering
5. Mamdani Fuzzy Inference System (centroid defuzzification)
6. Adaptive Median + MAD thresholding (anti-poisoning)
7. Zero-day / unseen attack experiment
8. Baseline comparisons (Simple CV vs Adaptive MAD vs Fuzzy Logic)
9. Sensitivity experiments (window sizes & k parameter)
10. Publication-ready visualizations
"""

import sys
import argparse
from pathlib import Path

# Ensure UTF-8 console output encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.config import ProjectConfig, config
from src.data_loader import CICIoTDataLoader
from src.preprocessing import DataPreprocessor
from src.windowing import SlidingWindowGenerator
from src.feature_engineering import WindowFeatureExtractor
from src.baseline import NormalBaselineBuilder, AdaptiveThresholdDetector
from src.fuzzy_system import IoTFuzzyAnomalyDetector
from src.zero_day_experiment import ZeroDayExperiment
from src.evaluation import PerformanceEvaluator
from src.visualization import ResearchVisualizer


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="IAT-Based Fuzzy Logic IoT Anomaly Detection Pipeline"
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        default=True,
        help="Run in fast sample mode (default: True)"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Run on full dataset (disables sample mode)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of records to load in evaluation scenario (default: 50,000)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=20,
        help="Sliding window size in observations (default: 20)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.50,
        help="Sliding window overlap fraction (default: 0.50)"
    )
    parser.add_argument(
        "--k",
        type=float,
        default=3.0,
        help="Adaptive MAD multiplier k (default: 3.0)"
    )
    parser.add_argument(
        "--held-out",
        type=str,
        default="Mirai-greeth_flood,Mirai-udpplain,Mirai-greip_flood",
        help="Comma-separated list of held-out zero-day attack classes"
    )
    parser.add_argument(
        "--raw-stream",
        action="store_true",
        help="Use raw sequential slice of test.csv instead of structured multi-phase scenario"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # --------------------------------------------------------------------------
    # Initialize Configuration
    # --------------------------------------------------------------------------
    cfg = config
    cfg.SAMPLE_MODE = not args.full_dataset
    cfg.SAMPLE_SIZE = args.sample_size
    cfg.WINDOW_SIZE = args.window_size
    cfg.OVERLAP = args.overlap
    cfg.MAD_K = args.k
    cfg.HELD_OUT_ATTACKS = [a.strip() for a in args.held_out.split(",") if a.strip()]
    cfg.ensure_directories()

    print("\n" + "=" * 80)
    print("      IAT-BASED FUZZY LOGIC ANOMALY DETECTION FOR IOT NETWORKS")
    print("                    CICIoT2023 Research Pipeline")
    print("=" * 80)
    print(f"Sample Mode:       {cfg.SAMPLE_MODE} (Evaluation Flows: {cfg.SAMPLE_SIZE:,})")
    print(f"Window Size:       {cfg.WINDOW_SIZE} consecutive observations (Overlap: {cfg.OVERLAP * 100:.0f}%)")
    print(f"Adaptive MAD k:    {cfg.MAD_K}")
    print(f"Held-Out Attacks:  {cfg.HELD_OUT_ATTACKS}")
    print(f"Evaluation Mode:   {'Raw sequential slice' if args.raw_stream else 'Multi-phase Continuous Scenario'}")
    print(f"Random Seed:       {cfg.RANDOM_SEED}")
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # STEP 1: Dataset Inspection
    # --------------------------------------------------------------------------
    loader = CICIoTDataLoader(cfg)
    file_report = loader.inspect_dataset_files()

    # --------------------------------------------------------------------------
    # STEP 2 & 3: Data Cleaning, Preprocessing & Label Analysis
    # --------------------------------------------------------------------------
    preprocessor = DataPreprocessor(cfg)

    # 1. Load pure normal training traffic to establish uncontaminated baseline
    raw_normal_train = loader.get_normal_training_data(max_samples=50000)
    clean_normal_train = preprocessor.clean_and_validate(raw_normal_train, dataset_name="normal_baseline_train")

    # 2. Load evaluation traffic (multi-phase scenario or raw slice)
    if args.raw_stream:
        raw_eval_df = loader.load_data(split="test", sample_size=cfg.SAMPLE_SIZE)
    else:
        raw_eval_df = loader.load_evaluation_stream(total_records=cfg.SAMPLE_SIZE)

    clean_eval_df = preprocessor.clean_and_validate(raw_eval_df, dataset_name="evaluation_stream")

    # --------------------------------------------------------------------------
    # STEP 7: Normal Baseline Construction (Strictly isolated from attacks)
    # --------------------------------------------------------------------------
    feature_extractor = WindowFeatureExtractor(cfg=cfg)
    normal_windows_df = feature_extractor.extract_features(clean_normal_train)

    baseline_builder = NormalBaselineBuilder(cfg)
    baseline_stats = baseline_builder.fit(normal_windows_df)
    feature_extractor.set_normal_baseline(baseline_stats["normal_mean_iat"])

    # --------------------------------------------------------------------------
    # STEP 5 & 6: Sliding Window & Feature Engineering on Evaluation Stream
    # --------------------------------------------------------------------------
    print("\n--- Generating Sliding Windows & Statistical Features on Evaluation Stream ---")
    eval_windows_df = feature_extractor.extract_features(clean_eval_df, save_csv=True)
    print(f"Generated {len(eval_windows_df):,} sliding windows.")

    # --------------------------------------------------------------------------
    # STEP 9: Mamdani Fuzzy Inference System
    # --------------------------------------------------------------------------
    print("\n--- Initializing Mamdani Fuzzy Inference System ---")
    fuzzy_detector = IoTFuzzyAnomalyDetector(baseline_stats=baseline_stats, cfg=cfg)

    # Compute continuous fuzzy risk scores across all evaluation windows
    print("Executing fuzzy inference across evaluation windows...")
    risk_scores, categories, fuzzy_preds = fuzzy_detector.predict_batch(eval_windows_df)
    eval_windows_df["attack_risk"] = risk_scores
    eval_windows_df["fuzzy_category"] = categories
    eval_windows_df["fuzzy_pred"] = fuzzy_preds

    # --------------------------------------------------------------------------
    # STEP 8: Adaptive Threshold Detector (Median + MAD)
    # --------------------------------------------------------------------------
    print("\n--- Executing Adaptive Median + MAD Detection ---")
    adaptive_detector = AdaptiveThresholdDetector(baseline_stats=baseline_stats, cfg=cfg)
    simple_detector = AdaptiveThresholdDetector(baseline_stats=baseline_stats, cfg=cfg)

    # Method 1: Simple CV
    simple_preds, _ = simple_detector.predict_simple_cv(eval_windows_df)
    eval_windows_df["simple_pred"] = simple_preds

    # Method 2: Adaptive MAD
    adaptive_res = adaptive_detector.run_adaptive_detection(
        eval_windows_df,
        confidence_risk_scores=risk_scores
    )
    eval_windows_df["adaptive_pred"] = adaptive_res["predictions"]
    eval_windows_df["adaptive_threshold"] = adaptive_res["adaptive_thresholds"]

    # --------------------------------------------------------------------------
    # STEP 10: Zero-Day / Unseen Attack Experiment
    # --------------------------------------------------------------------------
    zero_day_exp = ZeroDayExperiment(held_out_attacks=cfg.HELD_OUT_ATTACKS, cfg=cfg)
    df_norm, df_known, df_held_out = zero_day_exp.partition_splits(clean_eval_df)

    cohort_results = {}
    for name, sub_df in [("Normal", df_norm), ("Known_Attacks", df_known), ("Held_Out_Unseen", df_held_out)]:
        if len(sub_df) >= cfg.WINDOW_SIZE:
            w_sub = feature_extractor.extract_features(sub_df)
            cohort_results[name.lower()] = zero_day_exp.evaluate_cohort(
                cohort_name=name,
                df_windows=w_sub,
                simple_detector=simple_detector,
                adaptive_detector=adaptive_detector,
                fuzzy_detector=fuzzy_detector,
            )

    # --------------------------------------------------------------------------
    # STEP 11 & 12: Comprehensive Evaluation & Method Comparison
    # --------------------------------------------------------------------------
    evaluator = PerformanceEvaluator(cfg)
    y_true = eval_windows_df["binary_label"].to_numpy()

    preds_dict = {
        "Simple CV": eval_windows_df["simple_pred"].to_numpy(),
        "Adaptive MAD": eval_windows_df["adaptive_pred"].to_numpy(),
        "Fuzzy Logic": eval_windows_df["fuzzy_pred"].to_numpy(),
    }

    df_comparison = evaluator.compare_methods(
        y_true=y_true,
        preds_dict=preds_dict,
        cohort_results=cohort_results
    )

    print("\n" + "=" * 80)
    print("                OVERALL DETECTION METHOD COMPARISON")
    print("=" * 80)
    print(df_comparison.to_string(index=False))
    print("=" * 80 + "\n")

    # --------------------------------------------------------------------------
    # Sensitivity Analyses: Window Sizes and MAD k Parameter
    # --------------------------------------------------------------------------
    print("--- Running Parameter Sensitivity Analysis ---")
    # 1. Window size sensitivity
    window_sens = {}
    sample_sub = clean_eval_df.iloc[:15000]
    for w_s in cfg.WINDOW_EXPERIMENT_SIZES:
        w_tmp = feature_extractor.extract_features(sample_sub, window_size=w_s)
        _, _, p_fuzz = fuzzy_detector.predict_batch(w_tmp)
        m = evaluator.calculate_binary_metrics(w_tmp["binary_label"].to_numpy(), p_fuzz)
        window_sens[w_s] = m["f1_score"]

    # 2. k sensitivity
    k_sens = {}
    for k_val in cfg.K_EXPERIMENT_LIST:
        res_k = adaptive_detector.run_adaptive_detection(eval_windows_df.iloc[:2000], k=k_val)
        alarm_rate = float(np.mean(res_k["predictions"]))
        k_sens[k_val] = round(alarm_rate, 4)

    # --------------------------------------------------------------------------
    # STEP 13: Generate Research Visualizations
    # --------------------------------------------------------------------------
    print("\n--- Generating Publication-Grade Visualizations in results/plots/ ---")
    visualizer = ResearchVisualizer(cfg)
    visualizer.plot_01_iat_distribution(clean_eval_df)
    visualizer.plot_02_iat_normal_vs_attack(clean_eval_df)
    visualizer.plot_03_cv_distribution(eval_windows_df)
    visualizer.plot_04_packet_rate_distribution(eval_windows_df)
    visualizer.plot_05_sliding_window_cv_timeline(eval_windows_df)
    visualizer.plot_06_adaptive_threshold_vs_cv(eval_windows_df, eval_windows_df["adaptive_threshold"].to_numpy())
    visualizer.plot_07_fuzzy_membership_functions(fuzzy_detector)
    visualizer.plot_08_attack_risk_timeline(eval_windows_df, eval_windows_df["attack_risk"].to_numpy())

    cm_s = evaluator.calculate_binary_metrics(y_true, eval_windows_df["simple_pred"].to_numpy())["confusion_matrix"]
    cm_a = evaluator.calculate_binary_metrics(y_true, eval_windows_df["adaptive_pred"].to_numpy())["confusion_matrix"]
    cm_f = evaluator.calculate_binary_metrics(y_true, eval_windows_df["fuzzy_pred"].to_numpy())["confusion_matrix"]
    visualizer.plot_09_confusion_matrices(cm_s, cm_a, cm_f)
    visualizer.plot_10_performance_comparison(df_comparison)
    visualizer.plot_sensitivity_analysis(window_sens, k_sens)

    # --------------------------------------------------------------------------
    # Save Final Experiment Report
    # --------------------------------------------------------------------------
    summary_data = {
        "config": {
            "window_size": cfg.WINDOW_SIZE,
            "overlap": cfg.OVERLAP,
            "step_size": cfg.STEP_SIZE,
            "mad_k": cfg.MAD_K,
            "sample_size": len(clean_eval_df),
            "held_out_attacks": cfg.HELD_OUT_ATTACKS,
            "fuzzy_thresholds": {
                "low": cfg.FUZZY_LOW_THRESHOLD,
                "high": cfg.FUZZY_HIGH_THRESHOLD
            }
        },
        "baseline_statistics": baseline_stats,
        "method_comparison": df_comparison.to_dict(orient="records"),
        "cohort_results": {k: {m: v for m, v in val.items() if m not in ["ground_truth", "pred_simple", "pred_adaptive", "pred_fuzzy", "fuzzy_risk_scores", "fuzzy_categories", "adaptive_thresholds"]}
                          for k, val in cohort_results.items()},
        "window_size_sensitivity": window_sens,
        "mad_k_sensitivity": k_sens
    }
    report_path = evaluator.save_summary_report(summary_data)

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Results saved:")
    print(f"  - Window features:   {cfg.WINDOW_FEATURES_PATH}")
    print(f"  - Summary report:    {report_path}")
    print(f"  - Comparison CSV:    {cfg.METRICS_DIR / 'method_comparison.csv'}")
    print(f"  - Diagnostic plots:  {cfg.PLOTS_DIR} (12 figures generated)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
