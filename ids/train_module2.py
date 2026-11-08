"""
Train and evaluate everything the pipeline needs, and write the numbers the app shows.

    python -m ids.prepare_data      # once: builds data/processed/*.parquet
    python -m ids.train_module2     # trains Module 2, calibrates Module 1, evaluates all

Outputs
    models/pipeline/module1_baseline.json   benign baseline used by Module 1
    models/pipeline/module2_bundle.pkl      RF + INT8 MLP + zero-day detector
    results/pipeline/metrics.json           every number reported in the app / guide
    results/pipeline/*.png                  figures
"""

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ids.config import (FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, MODEL_DIR, NORMAL_LABEL,
                        RANDOM_SEED, TEST_SAMPLE, TRAIN_SAMPLE, WINDOW_SIZE)
from ids.module1_detection import FuzzyAnomalyDetector
from ids.module2_classification import (FEATURES, AttackClassifier, Int8MLP, Int8Quantizer,
                                        log_transform)

rng = np.random.default_rng(RANDOM_SEED)


def windows_by_class(df: pd.DataFrame, per_class: int, background: float = 0.0, benign_pool=None):
    """Homogeneous windows of WINDOW_SIZE flows per class (optionally mixed with benign flows)."""
    out = []
    for label, grp in df.groupby("label"):
        n_att = WINDOW_SIZE - int(round(background * WINDOW_SIZE)) if label != NORMAL_LABEL else WINDOW_SIZE
        for _ in range(per_class):
            w = grp.sample(n_att, replace=len(grp) < n_att, random_state=int(rng.integers(1e9)))
            if n_att < WINDOW_SIZE:
                w = pd.concat([w, benign_pool.sample(WINDOW_SIZE - n_att, random_state=int(rng.integers(1e9)))])
                w = w.sample(frac=1.0, random_state=int(rng.integers(1e9)))
            out.append((label, w))
    return out


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(TRAIN_SAMPLE)
    test = pd.read_parquet(TEST_SAMPLE)
    metrics: dict = {"dataset": {"train_rows": len(train), "test_rows": len(test),
                                 "classes": int(train["label"].nunique()),
                                 "held_out_zero_day": HELD_OUT_CLASSES}}

    known_tr = train[~train["label"].isin(HELD_OUT_CLASSES)]
    known_te = test[~test["label"].isin(HELD_OUT_CLASSES)]
    mirai_te = test[test["label"].isin(HELD_OUT_CLASSES)]
    benign_tr = train[train["label"] == NORMAL_LABEL]
    benign_te = test[test["label"] == NORMAL_LABEL]

    # ==============================================================================
    # MODULE 1: benign baseline + per-class detection rate
    # ==============================================================================
    print("\n=== Module 1: baseline & evaluation ===")
    mu_normal = float(benign_tr["IAT"].mean())
    tmp = FuzzyAnomalyDetector({"mu_normal": mu_normal, "cv_median": 1.0})
    cvs = [tmp.window_features(benign_tr.sample(WINDOW_SIZE, random_state=i))["cv"] for i in range(2000)]
    baseline = {"mu_normal": mu_normal, "cv_median": float(np.median(cvs)),
                "cv_p25": float(np.percentile(cvs, 25)), "cv_p75": float(np.percentile(cvs, 75)),
                "rate_median": float(benign_tr["Rate"].median()), "window_size": WINDOW_SIZE}
    (MODEL_DIR / "module1_baseline.json").write_text(json.dumps(baseline, indent=2))

    # Behaviour model: nearest-neighbour distance to BENIGN flows only (no attack labels used)
    b_fit, b_cal = train_test_split(benign_tr, test_size=0.25, random_state=RANDOM_SEED)
    Xb = log_transform(b_fit[FEATURES].to_numpy(np.float64))
    b_scaler = StandardScaler().fit(Xb)
    b_nn = NearestNeighbors(n_neighbors=5).fit(b_scaler.transform(Xb))
    behaviour = {"scaler": b_scaler, "nn": b_nn, "threshold": 1.0, "features": FEATURES}
    raw = FuzzyAnomalyDetector(baseline, behaviour)
    cal = [raw.behaviour_score(b_cal.sample(WINDOW_SIZE, random_state=i)) for i in range(1000)]
    behaviour["threshold"] = float(np.percentile(cal, 99))     # 1% of benign windows exceed B = 1
    with open(MODEL_DIR / "module1_behaviour.pkl", "wb") as fh:
        pickle.dump(behaviour, fh)

    def eval_m1(det):
        rows = []
        for label, w in windows_by_class(test, per_class=40):
            r = det.analyse_window(w)
            rows.append({"label": label, "family": FAMILY_OF[label], "risk": r.risk,
                         "flagged": r.forward_to_module2, "verdict": r.verdict,
                         **{k: r.features[k] for k in ("cv", "rate", "dev", "beh")}})
        m1 = pd.DataFrame(rows)
        m1["is_attack"] = m1["label"] != NORMAL_LABEL
        tp = int((m1.flagged & m1.is_attack).sum()); fp = int((m1.flagged & ~m1.is_attack).sum())
        fn = int((~m1.flagged & m1.is_attack).sum()); tn = int((~m1.flagged & ~m1.is_attack).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        # balanced accuracy: the test windows are 33 attack classes vs 1 benign class
        bal = 0.5 * (rec + tn / max(tn + fp, 1))
        return m1, {
            "windows": len(m1), "accuracy": (tp + tn) / len(m1), "balanced_accuracy": bal,
            "precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "fpr": fp / max(fp + tn, 1), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "detection_rate_by_family": m1[m1.is_attack].groupby("family")["flagged"].mean().round(4).to_dict(),
            "benign_false_alarm_rate": float(m1[~m1.is_attack]["flagged"].mean()),
            "mirai_zero_day_detection": float(m1[m1.label.isin(HELD_OUT_CLASSES)]["flagged"].mean()),
            "detection_rate_by_class": m1.groupby("label")["flagged"].mean().round(4).to_dict(),
        }

    m1_timing, met_timing = eval_m1(FuzzyAnomalyDetector(baseline, behaviour, mode="timing"))
    m1, met_full = eval_m1(FuzzyAnomalyDetector(baseline, behaviour, mode="full"))
    metrics["module1"] = {"baseline": baseline, "behaviour_threshold": behaviour["threshold"],
                          **met_full, "timing_only": met_timing}
    for name, mm in [("timing-only", met_timing), ("full", met_full)]:
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in mm.items()
                     if k not in ("detection_rate_by_class",)})

    # ==============================================================================
    # MODULE 2: classifiers (Mirai never seen)
    # ==============================================================================
    print("\n=== Module 2: training ===")
    Xtr_all = known_tr[FEATURES].to_numpy(np.float64); ytr_all = known_tr["label"].to_numpy()
    Xtr, Xval, ytr, yval = train_test_split(Xtr_all, ytr_all, test_size=0.15, stratify=ytr_all,
                                            random_state=RANDOM_SEED)
    Xte, yte = known_te[FEATURES].to_numpy(np.float64), known_te["label"].to_numpy()

    # ---- Random Forest -----------------------------------------------------------
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=60, max_depth=20, min_samples_leaf=3, n_jobs=-1,
                                class_weight="balanced_subsample", random_state=RANDOM_SEED)
    rf.fit(Xtr, ytr)
    rf_train_s = time.time() - t0
    t0 = time.time(); rf_pred = rf.predict(Xte); rf_lat = (time.time() - t0) / len(Xte) * 1000
    fam = np.vectorize(FAMILY_OF.get)
    rf_m = {"accuracy": accuracy_score(yte, rf_pred), "macro_f1": f1_score(yte, rf_pred, average="macro"),
            "weighted_f1": f1_score(yte, rf_pred, average="weighted"),
            "family_accuracy": accuracy_score(fam(yte), fam(rf_pred)),
            "family_macro_f1": f1_score(fam(yte), fam(rf_pred), average="macro"),
            "train_seconds": rf_train_s, "latency_ms_per_flow": rf_lat,
            "size_mb": len(pickle.dumps(rf)) / 1e6}
    print("RF", {k: round(v, 4) for k, v in rf_m.items()})

    # ---- INT8 MLP (edge model) ---------------------------------------------------
    quant = Int8Quantizer().fit(log_transform(Xtr))
    Xtr_q = quant.quantize(log_transform(Xtr))
    Xte_q = quant.quantize(log_transform(Xte))
    sc = StandardScaler().fit(quant.dequantize(Xtr_q))
    # Fold the standardisation into the quantiser so the MLP sees well-scaled inputs
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=150, early_stopping=True, n_iter_no_change=8,
                        random_state=RANDOM_SEED, batch_size=512, learning_rate_init=2e-3)
    t0 = time.time()
    mlp.fit(sc.transform(quant.dequantize(Xtr_q)), ytr)
    mlp_train_s = time.time() - t0
    # absorb scaler into first layer: W' = W / std, b' = b - (mean/std) @ W
    W0 = mlp.coefs_[0]
    mlp.intercepts_[0] = mlp.intercepts_[0] - (sc.mean_ / sc.scale_) @ W0
    mlp.coefs_[0] = W0 / sc.scale_[:, None]
    fp32_pred = mlp.predict(quant.dequantize(Xte_q))
    calib = Xtr_q[rng.choice(len(Xtr_q), 5000, replace=False)]
    int8 = Int8MLP(mlp, quant, calib)
    t0 = time.time(); i8_pred = int8.predict(Xte_q); i8_lat = (time.time() - t0) / len(Xte) * 1000
    fp32_bytes = sum(4 * (w.size + b.size) for w, b in zip(mlp.coefs_, mlp.intercepts_))  # float32
    mlp_m = {
        "fp32": {"accuracy": accuracy_score(yte, fp32_pred), "macro_f1": f1_score(yte, fp32_pred, average="macro"),
                 "family_accuracy": accuracy_score(fam(yte), fam(fp32_pred)), "size_kb": fp32_bytes / 1024},
        "int8": {"accuracy": accuracy_score(yte, i8_pred), "macro_f1": f1_score(yte, i8_pred, average="macro"),
                 "family_accuracy": accuracy_score(fam(yte), fam(i8_pred)), "size_kb": int8.nbytes() / 1024,
                 "latency_ms_per_flow": i8_lat},
        "agreement_fp32_int8": float((fp32_pred == i8_pred).mean()),
        "train_seconds": mlp_train_s, "architecture": "46 -> 128 -> 64 -> 31 (ReLU)",
    }
    mlp_m["size_reduction_pct"] = 100 * (1 - mlp_m["int8"]["size_kb"] / mlp_m["fp32"]["size_kb"])
    print("MLP", json.dumps(mlp_m, indent=1))

    # ---- Zero-day detector -------------------------------------------------------
    nn_scaler = StandardScaler().fit(log_transform(Xtr))
    idx = rng.choice(len(Xtr), min(60000, len(Xtr)), replace=False)
    nn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(nn_scaler.transform(log_transform(Xtr[idx])))

    def win_score(Xw):
        d, _ = nn.kneighbors(nn_scaler.transform(log_transform(Xw)))
        return float(np.median(d[:, 0]))

    val_df = pd.DataFrame(Xval, columns=FEATURES).assign(label=yval)
    val_scores = [win_score(w[FEATURES].to_numpy()) for _, w in windows_by_class(val_df, 20)]
    zd_threshold = float(np.percentile(val_scores, 97.5))

    bundle = {"rf": rf, "int8_mlp": int8, "quantizer": quant, "nn": nn, "nn_scaler": nn_scaler,
              "zd_threshold": zd_threshold}
    with open(MODEL_DIR / "module2_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    # ---- Window-level evaluation through the real inference wrapper ---------------
    print("\n=== Module 2: window-level evaluation ===")
    clf = AttackClassifier()
    rows = []
    for label, w in windows_by_class(test[test.label != NORMAL_LABEL], 30, background=0.3, benign_pool=benign_te):
        r = clf.classify_window(w)
        rows.append({"label": label, "pred": r.attack_class, "family": FAMILY_OF[label],
                     "pred_family": r.family, "zero_day": r.is_zero_day, "zd": r.zero_day_score})
    wdf = pd.DataFrame(rows)
    kn = wdf[~wdf.label.isin(HELD_OUT_CLASSES)]
    mi = wdf[wdf.label.isin(HELD_OUT_CLASSES)]
    metrics["module2"] = {
        "random_forest": rf_m, "mlp": mlp_m, "zero_day_threshold": zd_threshold,
        "window_eval": {
            "note": "windows of 20 flows: 70% attack + 30% benign background",
            "known_class_accuracy": float((kn.pred == kn.label).mean()),
            "known_family_accuracy": float((kn.pred_family == kn.family).mean()),
            "known_false_unknown_rate": float(kn.zero_day.mean()),
            "mirai_flagged_unknown": float(mi.zero_day.mean()),
            "mirai_assigned_to": mi[~mi.zero_day].pred.value_counts().to_dict(),
        },
        "per_class_report": classification_report(yte, rf_pred, output_dict=True, zero_division=0),
    }
    print(json.dumps(metrics["module2"]["window_eval"], indent=1))
    (METRICS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    # ---- Figures -------------------------------------------------------------------
    fams = sorted(set(fam(yte)))
    cm = confusion_matrix(fam(yte), fam(rf_pred), labels=fams, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", xticklabels=fams, yticklabels=fams, cmap="Blues", cbar=False)
    plt.title("Module 2 Random Forest: attack-family confusion matrix"); plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(METRICS_DIR / "m2_family_confusion.png", dpi=130); plt.close()

    rep = pd.DataFrame(metrics["module2"]["per_class_report"]).T.iloc[:-3].sort_values("f1-score")
    plt.figure(figsize=(8, 9))
    plt.barh(rep.index, rep["f1-score"], color="#3b82f6")
    plt.xlabel("F1-score"); plt.title("Module 2 Random Forest: per-class F1 (test set)")
    plt.tight_layout(); plt.savefig(METRICS_DIR / "m2_per_class_f1.png", dpi=130); plt.close()

    dr = pd.DataFrame({"IAT timing only": m1_timing.groupby("family")["flagged"].mean(),
                       "IAT + behaviour (final)": m1.groupby("family")["flagged"].mean()}).sort_values("IAT + behaviour (final)")
    dr.plot.barh(figsize=(7, 4.5), color=["#94a3b8", "#ef4444"])
    plt.xlabel("Share of windows flagged by Module 1"); plt.xlim(0, 1)
    plt.title("Module 1: flag rate per family (Benign = false alarms)")
    plt.tight_layout(); plt.savefig(METRICS_DIR / "m1_flag_rate_by_family.png", dpi=130); plt.close()
    print("\nSaved models and metrics.")


if __name__ == "__main__":
    main()
