"""
Comparison of IATFCM with the core techniques of the works in the literature survey,
all re-implemented and evaluated under the SAME protocol (CICIoT2023 windows of 20 records,
Mirai family held out of every supervised model).

    python -m ids.comparison          (after prepare_data, train_module2)

Re-implemented methods (short names used in the paper). These are re-implementations of each
paper's core idea on our data, not the authors' code:
    DT-3sigma  dynamic 3-sigma thresholds on packets/s and bytes/s             (detection)
    PP-TTF     IAT-only temporal features (mean/std/min/max/range/entropy/CV over 3, 5, 10 and
               20 samples + log IAT + burst share) -> Random Forest (100 trees, depth 10)
               [Ertam & Aydogmus, IEEE Access 2026]                             (detection + classification)
    SAE-PSVM   stacked-autoencoder features -> SVM tuned by particle swarm optimisation
               [Nie et al., Egyptian Informatics Journal 2025]                  (detection)
    HFF-XGB    statistical descriptors (mean/std/min/max/median) of all 46 features over the window
               -> XGBoost  [Solpan et al., IJIS 2026]                           (classification)
    DNN-KDQ    knowledge distillation from the Random Forest teacher into a small DNN student,
               then INT8 quantisation  [Umar et al., J. Cloud Computing 2025]  (classification)

Writes results/pipeline/comparison_metrics.json and comparison_per_window.csv.
"""

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ids.config import FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, NORMAL_LABEL, TRAIN_SAMPLE, WINDOW_SIZE
from ids.module2_classification import FEATURES, Int8MLP, Int8Quantizer, log_transform
from ids.pipeline import IDSPipeline

SEED = 42
SHARES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rng = np.random.default_rng(SEED)


# ------------------------------------------------------------------------------------------
# Window construction (same recipe for train and test, different source files)
# ------------------------------------------------------------------------------------------
def build_windows(df, benign, classes, shares, n_per, share_sampler=None):
    out = []
    by_class = {c: g for c, g in df.groupby("label")}
    for c in classes:
        g = by_class[c]
        for sh in shares:
            for _ in range(n_per):
                s = share_sampler() if share_sampler else sh
                n_att = WINDOW_SIZE if c == NORMAL_LABEL else max(1, int(round(s * WINDOW_SIZE)))
                a = g.sample(n_att, replace=len(g) < n_att, random_state=int(rng.integers(1e9)))
                b = benign.sample(WINDOW_SIZE - n_att, random_state=int(rng.integers(1e9)))
                w = pd.concat([a, b]).sample(frac=1.0, random_state=int(rng.integers(1e9)))
                out.append({"label": c, "share": 0.0 if c == NORMAL_LABEL else s, "w": w})
    return out


def make_test_windows(P, seed=2026):
    """The evaluation windows shared by Section 4 (ablation) and Section 5 (comparison):
    8 windows per attack class (33, incl. held-out Mirai) and attack share 50-100%, plus 240 benign."""
    global rng
    saved, rng = rng, np.random.default_rng(seed)
    try:
        attack = [c for c in P.classes if c != NORMAL_LABEL]
        w = build_windows(P.test, P.benign_pool, attack, SHARES, 8)
        w += build_windows(P.test, P.benign_pool, [NORMAL_LABEL], [None], 240)
    finally:
        rng = saved
    return w


# ------------------------------------------------------------------------------------------
# Feature extractors of the compared methods
# ------------------------------------------------------------------------------------------
def entropy(x, bins=5):
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log2(p)).sum())


class TTFFeatures:
    """PP-TTF: temporal features from the IAT sequence only."""
    def __init__(self, burst_thr):
        self.burst_thr = burst_thr

    def __call__(self, w):
        iat = w["IAT"].to_numpy(float)
        f = []
        for k in (3, 5, 10, WINDOW_SIZE):
            x = iat[-k:]
            m, s = x.mean(), x.std()
            f += [m, s, x.min(), x.max(), x.max() - x.min(), entropy(x, bins=min(k, 5)), s / (m + 1e-12)]
        f += [np.log1p(np.abs(iat)).mean(), float((iat < self.burst_thr).mean())]
        return f


def hff_features(w):
    """HFF-XGB: statistical descriptors of every flow feature over the window."""
    X = w[FEATURES].to_numpy(float)
    return np.concatenate([X.mean(0), X.std(0), X.min(0), X.max(0), np.median(X, 0)])


# ------------------------------------------------------------------------------------------
def main():
    P = IDSPipeline()
    train = pd.read_parquet(TRAIN_SAMPLE)
    train = train[~train.label.isin(HELD_OUT_CLASSES)]
    b_tr, b_te = train[train.label == NORMAL_LABEL], P.benign_pool
    known = sorted(c for c in train.label.unique())
    attack_known = [c for c in known if c != NORMAL_LABEL]

    print("building windows ...")
    tr_w = build_windows(train, b_tr, attack_known, [None], 120, share_sampler=lambda: rng.uniform(0.5, 1.0))
    tr_w += build_windows(train, b_tr, [NORMAL_LABEL], [None], 800)
    te_w = make_test_windows(P)
    y_tr_bin = np.array([w["label"] != NORMAL_LABEL for w in tr_w])
    y_tr_cls = np.array([w["label"] for w in tr_w])
    print(len(tr_w), "train windows,", len(te_w), "test windows")

    res = {}          # method -> dict(size_kb, latency_ms, n_features, task)
    preds = {}        # method -> list of predictions per test window

    def timed(fn, ws, n=150):
        idx = rng.choice(len(ws), n, replace=False)
        t0 = time.perf_counter()
        for i in idx:
            fn(ws[i]["w"])
        return (time.perf_counter() - t0) / n * 1000

    # ---------------- IATFCM Module 1 ------------------------------------------------------
    m1 = P.m1
    preds["IATFCM"] = [m1.analyse_window(w["w"]).forward_to_module2 for w in te_w]
    res["IATFCM"] = {"task": "detection", "size_kb": len(pickle.dumps(m1.behaviour)) / 1024,
                     "latency_ms": timed(lambda w: m1.analyse_window(w), te_w), "n_features": 4}

    # ---------------- DT-3sigma -------------------------------------------------------------
    def dt_feats(w):
        return np.array([w["Rate"].mean(), (w["Rate"] * w["AVG"]).mean()])
    bf = np.array([dt_feats(w["w"]) for w in tr_w if w["label"] == NORMAL_LABEL])
    mu, sd = bf.mean(0), bf.std(0)
    thr = mu + 3 * sd
    dt_pred = lambda w: bool((dt_feats(w) > thr).any())
    preds["DT-3sigma"] = [dt_pred(w["w"]) for w in te_w]
    res["DT-3sigma"] = {"task": "detection", "size_kb": 32 / 1024, "latency_ms": timed(dt_pred, te_w), "n_features": 2}

    # ---------------- PP-TTF (binary and multiclass) ---------------------------------------
    ttf = TTFFeatures(burst_thr=float(np.percentile(b_tr["IAT"], 5)))
    Xtr = np.array([ttf(w["w"]) for w in tr_w])
    Xte = np.array([ttf(w["w"]) for w in te_w])
    rf_b = RandomForestClassifier(100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xtr, y_tr_bin)
    rf_m = RandomForestClassifier(100, max_depth=10, class_weight="balanced", random_state=SEED, n_jobs=-1).fit(Xtr, y_tr_cls)
    rf_b.n_jobs = rf_m.n_jobs = 1
    preds["PP-TTF"] = list(rf_b.predict(Xte))
    preds["PP-TTF (cls)"] = list(rf_m.predict(Xte))
    res["PP-TTF"] = {"task": "detection", "size_kb": len(pickle.dumps(rf_b)) / 1024,
                     "latency_ms": timed(lambda w: rf_b.predict([ttf(w)]), te_w), "n_features": Xtr.shape[1]}
    res["PP-TTF (cls)"] = {"task": "classification", "size_kb": len(pickle.dumps(rf_m)) / 1024,
                           "latency_ms": timed(lambda w: rf_m.predict([ttf(w)]), te_w), "n_features": Xtr.shape[1]}

    # ---------------- SAE-PSVM -------------------------------------------------------------
    print("SAE-PSVM ...")
    rec_b = b_tr.sample(4000, random_state=SEED)
    rec_a = train[train.label != NORMAL_LABEL].groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 140), random_state=SEED))
    rec = pd.concat([rec_b, rec_a]).sample(frac=1.0, random_state=SEED)
    sc = StandardScaler().fit(log_transform(rec[FEATURES].to_numpy(float)))
    Xr = sc.transform(log_transform(rec[FEATURES].to_numpy(float)))
    yr = (rec.label != NORMAL_LABEL).to_numpy()
    sae = MLPRegressor(hidden_layer_sizes=(32, 16, 32), max_iter=200, random_state=SEED).fit(Xr, Xr)

    def encode(X):
        h = X
        for W, b in list(zip(sae.coefs_, sae.intercepts_))[:2]:
            h = np.maximum(h @ W + b, 0)
        return h
    Zr = encode(Xr)
    ntr = int(0.6 * len(Zr))

    def fitness(p):
        m = SVC(C=10 ** p[0], gamma=10 ** p[1]).fit(Zr[:3000], yr[:3000])
        return (m.predict(Zr[ntr:ntr + 2000]) == yr[ntr:ntr + 2000]).mean()
    lo, hi = np.array([-1.0, -4.0]), np.array([3.0, 0.0])
    pos = rng.uniform(lo, hi, (8, 2)); vel = np.zeros_like(pos)
    pbest, pfit = pos.copy(), np.array([fitness(p) for p in pos])
    gbest = pbest[pfit.argmax()].copy()
    for _ in range(6):
        r1, r2 = rng.random((8, 2)), rng.random((8, 2))
        vel = 0.7 * vel + 1.5 * r1 * (pbest - pos) + 1.5 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        f = np.array([fitness(p) for p in pos])
        better = f > pfit; pbest[better], pfit[better] = pos[better], f[better]
        gbest = pbest[pfit.argmax()].copy()
    svm = SVC(C=10 ** gbest[0], gamma=10 ** gbest[1]).fit(Zr[:ntr], yr[:ntr])

    def sae_pred(w):
        z = encode(sc.transform(log_transform(w[FEATURES].to_numpy(float))))
        return bool(svm.predict(z).mean() >= 0.5)
    preds["SAE-PSVM"] = [sae_pred(w["w"]) for w in te_w]
    res["SAE-PSVM"] = {"task": "detection", "size_kb": (len(pickle.dumps(sae)) + len(pickle.dumps(svm))) / 1024,
                       "latency_ms": timed(sae_pred, te_w), "n_features": 46,
                       "pso_best": {"C": float(10 ** gbest[0]), "gamma": float(10 ** gbest[1])}}

    # ---------------- IATFCM Module 2 (RF and INT8) ----------------------------------------
    m2 = P.m2
    for key, name in (("rf", "IATFCM-RF"), ("int8", "IATFCM-INT8")):
        preds[name] = [m2.classify_window(w["w"], model=key).attack_class for w in te_w]
        size = len(pickle.dumps(m2.rf)) / 1024 if key == "rf" else m2.int8_mlp.nbytes() / 1024
        res[name] = {"task": "classification", "size_kb": size, "n_features": 46,
                     "latency_ms": timed(lambda w, k=key: m2.classify_window(w, model=k), te_w)}

    # ---------------- HFF-XGB --------------------------------------------------------------
    print("HFF-XGB ...")
    classes = sorted(set(y_tr_cls)); cidx = {c: i for i, c in enumerate(classes)}
    Htr = np.array([hff_features(w["w"]) for w in tr_w])
    xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.9, tree_method="hist",
                        random_state=SEED, n_jobs=8).fit(Htr, np.array([cidx[c] for c in y_tr_cls]))
    xgb.set_params(n_jobs=1)
    Hte = np.array([hff_features(w["w"]) for w in te_w])
    preds["HFF-XGB"] = [classes[i] for i in xgb.predict(Hte)]
    res["HFF-XGB"] = {"task": "classification", "size_kb": len(pickle.dumps(xgb)) / 1024, "n_features": Htr.shape[1],
                      "latency_ms": timed(lambda w: xgb.predict(hff_features(w)[None]), te_w)}

    # ---------------- DNN-KDQ --------------------------------------------------------------
    print("DNN-KDQ ...")
    rec_k = train.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 1500), random_state=SEED))
    Xk = rec_k[FEATURES].to_numpy(float)
    soft = m2.rf.predict_proba(Xk)                         # teacher soft labels
    q = Int8Quantizer().fit(log_transform(Xk))
    Xq = q.dequantize(q.quantize(log_transform(Xk)))
    ksc = StandardScaler().fit(Xq)
    student = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=120, early_stopping=True, random_state=SEED).fit(ksc.transform(Xq), soft)
    W0 = student.coefs_[0]
    student.intercepts_[0] = student.intercepts_[0] - (ksc.mean_ / ksc.scale_) @ W0
    student.coefs_[0] = W0 / ksc.scale_[:, None]
    student.classes_ = m2.rf.classes_
    kdq = Int8MLP(student, q, q.quantize(log_transform(Xk[rng.choice(len(Xk), 4000, replace=False)])))

    def kdq_window(w):
        cls = kdq.predict(q.quantize(log_transform(w[FEATURES].to_numpy(float))))
        att = [c for c in cls if c != NORMAL_LABEL]
        return pd.Series(att).mode().iloc[0] if len(att) >= 0.3 * len(cls) else NORMAL_LABEL
    preds["DNN-KDQ"] = [kdq_window(w["w"]) for w in te_w]
    res["DNN-KDQ"] = {"task": "classification", "size_kb": kdq.nbytes() / 1024, "n_features": 46,
                      "latency_ms": timed(kdq_window, te_w)}

    # ---------------- Metrics --------------------------------------------------------------
    df = pd.DataFrame({"label": [w["label"] for w in te_w], "share": [w["share"] for w in te_w]})
    df["family"] = df.label.map(FAMILY_OF)
    df["attack"] = df.label != NORMAL_LABEL
    df["held_out"] = df.label.isin(HELD_OUT_CLASSES)
    for k, v in preds.items():
        df[k] = v
    df.to_csv(METRICS_DIR / "comparison_per_window.csv", index=False)

    det = ["IATFCM", "DT-3sigma", "PP-TTF", "SAE-PSVM"]
    cls_m = ["IATFCM-RF", "IATFCM-INT8", "HFF-XGB", "PP-TTF (cls)", "DNN-KDQ"]
    out = {"protocol": {"train_windows": len(tr_w), "test_windows": len(te_w), "shares": SHARES,
                        "test_attack_windows_per_class_per_share": 8, "benign_test_windows": 240},
           "resources": res, "detection": {}, "classification": {}}
    for m in det:
        f = df[m].astype(bool)
        a, b = df.attack, ~df.attack
        tp, fp = int((f & a).sum()), int((f & b).sum())
        rec, prec = tp / a.sum(), tp / max(tp + fp, 1)
        out["detection"][m] = {
            "recall": rec, "precision": prec, "f1": 2 * rec * prec / max(rec + prec, 1e-12), "fpr": fp / b.sum(),
            "recall_by_share": df[a].groupby("share")[m].mean().astype(float).to_dict(),
            "recall_by_family": df[a].groupby("family")[m].mean().astype(float).to_dict(),
            "zero_day_detection": float(df[df.held_out][m].mean()),
        }
    kn = df[~df.held_out]
    for m in cls_m:
        pred = kn[m].astype(str)
        fam_pred = pred.map(lambda c: FAMILY_OF.get(c, "Unknown"))
        correct = pred == kn.label
        labels = sorted(kn.label.unique())
        f1s = []
        for c in labels:
            tp = ((pred == c) & (kn.label == c)).sum(); fp = ((pred == c) & (kn.label != c)).sum()
            fn = ((pred != c) & (kn.label == c)).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
        out["classification"][m] = {
            "accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1s)),
            "family_accuracy": float((fam_pred == kn.family).mean()),
            "accuracy_by_share": kn[kn.attack].assign(c=correct[kn.attack]).groupby("share")["c"].mean().astype(float).to_dict(),
            "zero_day_flagged_unknown": float(df[df.held_out][m].astype(str).str.startswith("Unknown").mean()),
            "mirai_assigned_to": df[df.held_out][m].astype(str).value_counts().head(4).to_dict(),
        }
    (METRICS_DIR / "comparison_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: {m: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in d.items()
                              if not isinstance(vv, dict)} for m, d in out[k].items()} for k in ("detection", "classification")}, indent=1))
    print(json.dumps({m: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for m, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
