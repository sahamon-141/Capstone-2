"""
Runs Module 3 on its own, without re-running the whole pipeline.

``main.py`` needs the 1.5 GB CICIoT2023 CSVs. This entry point does not: it
reuses the window features and the normal baseline that a previous run already
wrote to ``results/``, rebuilds the Mamdani detector from that baseline, scores
the windows and then applies the mitigation stage to every one of them.

    python run_module3.py

Writes ``results/metrics/module3_metrics.json`` and
``results/metrics/module3_per_window.csv``.
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config
from src.fuzzy_system import IoTFuzzyAnomalyDetector
from module3_mitigation.pipeline_evaluation import run_module3_evaluation


def main() -> int:
    features_path = config.WINDOW_FEATURES_PATH
    summary_path = config.METRICS_DIR / "summary_metrics.json"

    for path, what in ((features_path, "window features"), (summary_path, "summary metrics")):
        if not path.exists():
            print(f"[ERROR] Missing {what}: {path}")
            print("        Run main.py once to generate it, then re-run this script.")
            return 1

    windows_df = pd.read_csv(features_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_stats = summary["baseline_statistics"]
    held_out = summary.get("config", {}).get("held_out_attacks", config.HELD_OUT_ATTACKS)

    print(f"Loaded {len(windows_df):,} windows from {features_path.name}")

    # Score the windows with the same Mamdani system Module 1 uses.
    print("Scoring windows with the Mamdani fuzzy detector...")
    fuzzy = IoTFuzzyAnomalyDetector(baseline_stats=baseline_stats)
    risk, categories, preds = fuzzy.predict_batch(windows_df)
    windows_df["attack_risk"] = risk
    windows_df["fuzzy_category"] = categories
    windows_df["fuzzy_pred"] = preds

    print("Applying Module 3 to every flagged window...")
    result = run_module3_evaluation(
        windows_df=windows_df,
        fuzzy_detector=fuzzy,
        baseline_stats=baseline_stats,
        held_out_classes=held_out,
        metrics_dir=config.METRICS_DIR,
        risk_threshold=config.FUZZY_LOW_THRESHOLD,
    )

    cov, con, zd = result["policy_coverage"], result["containment"], result["zero_day"]
    col = result["collateral_on_benign_windows"]

    print("\n" + "=" * 68)
    print("MODULE 3 RESULTS")
    print("=" * 68)
    print(f"Total windows                 : {result['total_windows']:,}")
    print(f"Bypassed by Module 1          : {result['windows_bypassed_by_module1']:,}")
    print(f"Sent to mitigation            : {result['windows_sent_to_mitigation']:,} "
          f"({result['windows_sent_to_mitigation_pct']:.2f}%)")
    print("-" * 68)
    print(f"Matched a class policy        : {cov['matched_class_policy']:,} "
          f"({cov['matched_class_policy_pct']:.2f}%)")
    print(f"Zero-day behaviour route      : {cov['zero_day_behaviour_route']:,} "
          f"({cov['zero_day_behaviour_route_pct']:.2f}%)")
    sup = result["volume_suppression_by_first_action"]
    print("-" * 68)
    print("Volume removed by the least disruptive action (measured):")
    print(f"  windows filtered            : {sup.get('windows', 0):,}")
    print(f"  mean packet rate            : {sup.get('mean_rate_before', 0):,.0f} -> "
          f"{sup.get('mean_rate_after', 0):,.0f} pkt/s")
    print(f"  mean volume suppressed      : {sup.get('mean_pct', 0):.2f}% "
          f"(median {sup.get('median_pct', 0):.2f}%)")
    print("-" * 68)
    print("Risk containment (modelled, conservative lower bound):")
    print(f"  actionable windows          : {con['actionable_windows']:,}")
    print(f"  cleared by the filter alone : {con['contained_by_filter_alone']:,} "
          f"({con['contained_by_filter_alone_pct']:.2f}%)")
    print(f"  escalated to a block        : {con['needed_disruptive_action']:,} "
          f"({con['needed_disruptive_action_pct']:.2f}%)")
    print(f"  mean fuzzy risk drop        : {con['mean_risk_drop']}")
    print("-" * 68)
    print(f"Benign windows flagged        : {col['benign_windows_flagged_by_module1']:,}")
    print(f"  best case, blocked          : {col['best_case_receiving_a_blocking_action']:,} "
          "(Module 2 labels them benign)")
    print(f"  worst case, blocked         : {col['worst_case_receiving_a_blocking_action']:,} "
          f"({col['worst_case_blocked_pct_of_flagged_benign']:.2f}% if Module 2 calls them an attack)")
    print("-" * 68)
    print(f"Held-out windows mitigated    : {zd['held_out_windows_reaching_mitigation']:,}")
    print(f"  mean volume suppressed      : {zd['mean_volume_suppressed_pct']:.2f}%")
    print(f"  contained after escalation  : {zd['contained_overall']:,} "
          f"({zd['contained_overall_pct']:.2f}%)")
    print(f"  behaviour split             : {zd['behaviour_distribution']}")
    print("-" * 68)
    print("Action distribution:")
    for action, count in result["action_distribution"].items():
        print(f"  {action:<32} {count:,}")
    print("-" * 68)
    print(f"Families exercised by this stream: {', '.join(result['families_exercised_by_this_stream'])}")
    print(f"Families defined in policies.json: {len(result['families_defined_in_policy_file'])}")
    print("=" * 68)
    print("Volume suppression is measured. Risk containment is MODELLED,")
    print("not measured on a live network.")
    print(f"Written to {config.METRICS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
