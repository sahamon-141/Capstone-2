"""
Numbers and figures for the paper, all computed on the current pipeline (ids/).

    python -m ids.paper_figures      (after prepare_data, train_module2, evaluate_module3)

Writes results/pipeline/paper_metrics.json and results/pipeline/fig_*.png:
    fig_anomaly_detection_comparison.png   static CV, adaptive MAD, timing-only fuzzy, final fuzzy
    fig_anomaly_confusion_matrices.png     confusion matrices of the four detectors
    fig_cohort_detection_rates.png         normal / known attack / held-out Mirai flag rates
    fig_multiclass_comparison.png          RF vs FP32 MLP vs INT8 MLP: quality, latency, size
    fig_parameter_sensitivity.png          window size W (final fuzzy F1), MAD k (alarm rate)
    fig_zero_day_analysis.png              Mirai detection per stage + zero-day score distribution
    fig_module2_invocation.png             Module 1 decisions on the mixed-day stream
    fig_mitigation_by_family.png           Module 3 resolved / automatic per policy family
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ids.config import FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, NORMAL_LABEL, TRAIN_SAMPLE
from ids.module1_detection import FuzzyAnomalyDetector
from ids.pipeline import IDSPipeline

rng = np.random.default_rng(42)
COL = {"Static CV": "#94a3b8", "Adaptive MAD": "#f59e0b", "Fuzzy (timing only)": "#60a5fa",
       "Fuzzy (timing + B)": "#dc2626"}


def scores(flag, y):
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    tp, fp = int((flag & y).sum()), int((flag & ~y).sum())
    fn, tn = int((~flag & y).sum()), int((~flag & ~y).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"accuracy": (tp + tn) / len(y), "balanced_accuracy": 0.5 * (r + tn / max(tn + fp, 1)),
            "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12), "fpr": fp / max(fp + tn, 1),
            "confusion": [[tn, fp], [fn, tp]]}


def make_windows(df, W, per_class):
    out = []
    for label, g in df.groupby("label"):
        for _ in range(per_class):
            out.append((label, g.sample(W, replace=len(g) < W, random_state=int(rng.integers(1e9)))))
    order = rng.permutation(len(out))          # interleave classes so the adaptive threshold sees a mixed stream
    return [out[i] for i in order]


def main():
    P = IDSPipeline()
    train = pd.read_parquet(TRAIN_SAMPLE)
    benign_tr = train[train.label == NORMAL_LABEL]
    full, timing = P.m1, P.m1_timing
    M = {}

    # ---------------- Module 1 vs statistical baselines (same 1,360 windows) --------------------
    ref_cv = np.array([full.window_features(benign_tr.sample(20, random_state=i))["cv"] for i in range(2000)])
    p95 = float(np.percentile(ref_cv, 95))
    wins = make_windows(P.test, 20, 40)
    rows = []
    for label, w in wins:
        r_full = full.analyse_window(w)
        r_tim = timing.infer(r_full.features["cv"], r_full.features["rate"], r_full.features["dev"])
        rows.append({"label": label, "cv": r_full.features["cv"], "risk_full": r_full.risk,
                     "risk_timing": r_tim.risk})
    df = pd.DataFrame(rows)
    y = (df.label != NORMAL_LABEL).to_numpy()

    def adaptive(k):
        hist = list(np.random.default_rng(0).choice(ref_cv, 200, replace=False))
        flags = []
        for cv, risk in zip(df.cv, df.risk_full):
            h = np.array(hist[-500:]); med = np.median(h); mad = np.median(np.abs(h - med))
            t = med + k * mad
            flags.append(cv > t)
            if cv <= t and risk < 0.35:          # anti-poisoning: only confident normal windows
                hist.append(cv)
        return np.array(flags)

    flags = {"Static CV": (df.cv > p95).to_numpy(), "Adaptive MAD": adaptive(3.0),
             "Fuzzy (timing only)": (df.risk_timing >= 0.35).to_numpy(),
             "Fuzzy (timing + B)": (df.risk_full >= 0.35).to_numpy()}
    M["detectors"] = {k: scores(v, y) for k, v in flags.items()}
    held = df.label.isin(HELD_OUT_CLASSES).to_numpy()
    norm = (df.label == NORMAL_LABEL).to_numpy()
    known = ~held & ~norm
    M["cohorts"] = {k: {"normal": float(v[norm].mean()), "known": float(v[known].mean()), "held_out": float(v[held].mean())}
                    for k, v in flags.items()}
    M["windows"] = {"total": len(df), "normal": int(norm.sum()), "known": int(known.sum()), "held_out": int(held.sum())}

    # timing of Module 1
    t0 = time.perf_counter()
    for _, w in wins[:200]:
        full.analyse_window(w)
    M["module1_ms_per_window"] = (time.perf_counter() - t0) / 200 * 1000

    metrics_names = ["accuracy", "precision", "recall", "f1", "fpr"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(metrics_names)); wdt = 0.2
    for i, (k, s) in enumerate(M["detectors"].items()):
        ax.bar(x + (i - 1.5) * wdt, [s[m] * 100 for m in metrics_names], wdt, label=k, color=COL[k])
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1-score", "FPR"]); ax.set_ylabel("%")
    ax.set_ylim(0, 105); ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_anomaly_detection_comparison.png", dpi=150); plt.close()

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (k, s) in zip(axes, M["detectors"].items()):
        cm = np.array(s["confusion"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["normal", "attack"]); ax.set_yticks([0, 1], ["normal", "attack"])
        ax.set_title(k, fontsize=9); ax.set_xlabel("predicted")
    axes[0].set_ylabel("actual"); plt.tight_layout()
    plt.savefig(METRICS_DIR / "fig_anomaly_confusion_matrices.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(8, 3.8))
    coh = ["normal", "known", "held_out"]; x = np.arange(3)
    for i, (k, v) in enumerate(M["cohorts"].items()):
        ax.bar(x + (i - 1.5) * wdt, [v[c] * 100 for c in coh], wdt, label=k, color=COL[k])
    ax.set_xticks(x, ["Normal (false alarms)", "Known attacks", "Held-out Mirai"]); ax.set_ylabel("% of windows flagged")
    ax.set_ylim(0, 105); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_cohort_detection_rates.png", dpi=150); plt.close()

    # ---------------- Parameter sensitivity -------------------------------------------------------
    sens_w = {}
    for W in (10, 20, 50, 100):
        ws = make_windows(P.test, W, 15)
        fl = [full.analyse_window(w).risk >= 0.35 for _, w in ws]
        sens_w[W] = scores(fl, [l != NORMAL_LABEL for l, _ in ws])["f1"]
    sens_k = {k: float(adaptive(k).mean()) for k in (2.0, 2.5, 3.0, 3.5, 4.0)}
    M["sensitivity"] = {"window_size_f1": sens_w, "mad_k_alarm_rate": sens_k}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(list(sens_w), [v * 100 for v in sens_w.values()], "o-", color="#dc2626")
    a1.axvline(20, ls=":", color="grey"); a1.set_xlabel("window size W"); a1.set_ylabel("F1-score (%)")
    a1.set_title("(a) Module 1 F1 vs window size", fontsize=9)
    a2.plot(list(sens_k), [v * 100 for v in sens_k.values()], "o-", color="#f59e0b")
    a2.axvline(3.0, ls=":", color="grey"); a2.set_xlabel("k"); a2.set_ylabel("alarm rate (%)")
    a2.set_title("(b) Adaptive MAD alarm rate vs k", fontsize=9)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_parameter_sensitivity.png", dpi=150); plt.close()

    # ---------------- Module 2 comparison (latency measured per 20-flow window) -------------------
    m2 = P.metrics["module2"]
    lat = {"rf": [], "int8": []}
    for label, w in make_windows(P.test[P.test.label != NORMAL_LABEL], 20, 3)[:90]:
        for key in lat:
            lat[key].append(P.m2.classify_window(w, model=key).latency_ms)
    M["module2_latency_ms_per_window"] = {k: float(np.median(v)) for k, v in lat.items()}
    rf, mlp = m2["random_forest"], m2["mlp"]
    names = ["Random Forest", "MLP FP32", "MLP INT8"]
    acc = [rf["accuracy"], mlp["fp32"]["accuracy"], mlp["int8"]["accuracy"]]
    mf1 = [rf["macro_f1"], mlp["fp32"]["macro_f1"], mlp["int8"]["macro_f1"]]
    fam = [rf["family_accuracy"], mlp["fp32"]["family_accuracy"], mlp["int8"]["family_accuracy"]]
    size_kb = [rf["size_mb"] * 1024, mlp["fp32"]["size_kb"], mlp["int8"]["size_kb"]]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(3)
    a1.bar(x - .25, np.array(acc) * 100, .25, label="accuracy"); a1.bar(x, np.array(mf1) * 100, .25, label="macro-F1")
    a1.bar(x + .25, np.array(fam) * 100, .25, label="family acc."); a1.set_xticks(x, names, fontsize=8)
    a1.set_ylim(0, 105); a1.legend(fontsize=7); a1.set_title("(a) classification (%)", fontsize=9)
    a2.bar(["Random Forest", "MLP INT8"], [M["module2_latency_ms_per_window"]["rf"], M["module2_latency_ms_per_window"]["int8"]],
           color=["#3b82f6", "#16a34a"]); a2.set_ylabel("ms per window (20 flows)"); a2.set_title("(b) latency", fontsize=9)
    a3.bar(names, size_kb, color=["#3b82f6", "#94a3b8", "#16a34a"]); a3.set_yscale("log"); a3.set_ylabel("KB (log)")
    a3.set_title("(c) model size", fontsize=9); a3.tick_params(axis="x", labelsize=8)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_multiclass_comparison.png", dpi=150); plt.close()

    # ---------------- Zero-day analysis -----------------------------------------------------------
    zd_known, zd_mirai = [], []
    for label, w in make_windows(P.test[P.test.label != NORMAL_LABEL], 20, 6):
        s = P.m2.classify_window(w).zero_day_score
        (zd_mirai if label in HELD_OUT_CLASSES else zd_known).append(s)
    we = m2["window_eval"]
    stages = {"Module 1 (timing only)": M["cohorts"]["Fuzzy (timing only)"]["held_out"],
              "Module 1 (timing + B)": M["cohorts"]["Fuzzy (timing + B)"]["held_out"],
              "Module 2 open-set check": we["mirai_flagged_unknown"],
              "Earlier softmax rejection": 0.0}
    M["zero_day_scores"] = {"known_median": float(np.median(zd_known)), "mirai_median": float(np.median(zd_mirai))}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.barh(list(stages), [v * 100 for v in stages.values()], color=["#60a5fa", "#dc2626", "#16a34a", "#94a3b8"])
    a1.set_xlim(0, 105); a1.set_xlabel("% of held-out Mirai windows detected"); a1.set_title("(a) unseen-attack detection", fontsize=9)
    bins = np.logspace(-2, 1.5, 40)
    a2.hist(np.clip(zd_known, 1e-2, None), bins=bins, alpha=.6, label="known attacks")
    a2.hist(np.clip(zd_mirai, 1e-2, None), bins=bins, alpha=.6, label="held-out Mirai")
    a2.axvline(1.0, color="k", ls="--", label="threshold"); a2.set_xscale("log")
    a2.set_xlabel("zero-day score Z (log)"); a2.legend(fontsize=8); a2.set_title("(b) Module 2 zero-day score", fontsize=9)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_zero_day_analysis.png", dpi=150); plt.close()

    # ---------------- Module 1 routing + Module 3 per family -------------------------------------
    m3 = json.loads((METRICS_DIR / "module3_metrics.json").read_text())
    pw = pd.read_csv(METRICS_DIR / "module3_per_window.csv")
    md = pw[pw.scenario == "mixed_day"]
    verdict = pd.cut(md.m1_risk, [-1, 0.35, 0.65, 2], right=False, labels=["Normal", "Suspicious", "Attack"])
    counts = verdict.value_counts().reindex(["Normal", "Suspicious", "Attack"])
    M["mixed_day_routing"] = counts.to_dict()
    plt.figure(figsize=(6, 3.4))
    plt.bar(counts.index, counts.values, color=["#22c55e", "#f59e0b", "#ef4444"])
    for i, v in enumerate(counts.values):
        plt.text(i, v + 3, f"{v} ({v / len(md):.1%})", ha="center", fontsize=9)
    plt.ylabel("windows"); plt.title("Module 1 decisions (390 windows); only Suspicious + Attack reach Module 2", fontsize=9)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_module2_invocation.png", dpi=150); plt.close()

    fam_df = pd.DataFrame(m3["coverage_per_policy_family"]).sort_values("resolved_pct")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(fam_df))
    ax.barh(yy - .2, fam_df.resolved_pct * 100, .4, label="resolved (risk < 0.35)", color="#16a34a")
    ax.barh(yy + .2, fam_df.auto_pct * 100, .4, label="started automatically", color="#3b82f6")
    ax.set_yticks(yy, fam_df.policy_family.str.replace("_", " ")); ax.set_xlim(0, 105); ax.set_xlabel("%")
    ax.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(METRICS_DIR / "fig_mitigation_by_family.png", dpi=150); plt.close()

    (METRICS_DIR / "paper_metrics.json").write_text(json.dumps(M, indent=2, default=float))
    print(json.dumps(M, indent=1, default=float))


if __name__ == "__main__":
    main()
