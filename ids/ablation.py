"""
Ablation study of IATFCM: remove or replace one component at a time.

    python -m ids.ablation          (after prepare_data, train_module2)

A  Module 1 inputs   : timing only (mean rate, original design) | timing only (median rate) |
                       behaviour score B alone | full (timing + B, fuzzy)
B  Module 2          : plain majority vote vs attack-vote rule;
                       zero-day check: none | max-softmax-probability (MSP) | nearest-neighbour (ours)
C  Module 3          : full | no escalation | no shape routing for zero-day | naive drop-whole-window;
                       also how often a one-shot action without verification would leave the attack running
D  Gating            : Module 2 behind Module 1 (ours) vs Module 2 on every window

Writes results/pipeline/ablation_metrics.json.
"""

import json
import time

import numpy as np
import pandas as pd

from ids.comparison import make_test_windows
from ids.config import FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, NORMAL_LABEL, RISK_SUSPICIOUS, WINDOW_SIZE
from ids.evaluate_module3 import MIXED_DAY
from ids.module1_detection import RULES_FULL, RULES_FULL_ORIGINAL_R13
from ids.module2_classification import FEATURES
from ids.module3_mitigation import ACTIONS
from ids.pipeline import IDSPipeline

rng = np.random.default_rng(7)
SHARES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def det_scores(flag, attack, held):
    flag, attack, held = map(lambda a: np.asarray(a, bool), (flag, attack, held))
    tp, fp = (flag & attack).sum(), (flag & ~attack).sum()
    rec, prec = tp / attack.sum(), tp / max(tp + fp, 1)
    return {"recall": float(rec), "precision": float(prec), "f1": float(2 * rec * prec / max(rec + prec, 1e-12)),
            "fpr": float(fp / (~attack).sum()), "zero_day_detection": float(flag[held].mean())}


def main():
    P = IDSPipeline()
    attack_classes = [c for c in P.classes if c != NORMAL_LABEL]
    wins = [(d["label"], d["share"], d["w"]) for d in make_test_windows(P)]   # same windows as Section 5
    labels = np.array([w[0] for w in wins]); attack = labels != NORMAL_LABEL
    held = np.isin(labels, HELD_OUT_CLASSES)
    out = {"windows": len(wins)}

    # ---------------- A: Module 1 inputs ---------------------------------------------------
    full, timing = P.m1, P.m1_timing
    feats = [full.window_features(w) for _, _, w in wins]
    mean_rates = [float(w["Rate"].mean()) for _, _, w in wins]
    A = {
        "Timing only, mean rate (original)": [timing.infer(f["cv"], r, f["dev"]).risk >= RISK_SUSPICIOUS
                                              for f, r in zip(feats, mean_rates)],
        "Timing only, median rate": [timing.infer(f["cv"], f["rate"], f["dev"]).risk >= RISK_SUSPICIOUS for f in feats],
        "Behaviour score B alone (B > 1)": [f["beh"] > 1.0 for f in feats],
    }
    full.rules = RULES_FULL_ORIGINAL_R13
    A["Timing + B, original R13"] = [full.infer(f["cv"], f["rate"], f["dev"], f["beh"]).risk >= RISK_SUSPICIOUS for f in feats]
    full.rules = RULES_FULL
    A["Full IATFCM (timing + B, gated R13)"] = [full.infer(f["cv"], f["rate"], f["dev"], f["beh"]).risk >= RISK_SUSPICIOUS
                                               for f in feats]
    out["A_module1"] = {k: det_scores(v, attack, held) for k, v in A.items()}
    for k, v in A.items():
        out["A_module1"][k]["recall_by_share"] = (pd.DataFrame({"s": [w[1] for w in wins], "f": v})[attack]
                                                  .groupby("s")["f"].mean().to_dict())

    # ---------------- B: Module 2 ----------------------------------------------------------
    m2 = P.m2
    rf = m2.rf
    classes = list(rf.classes_)
    plain, ours = [], []
    msp_conf, zd_scores = [], []
    for c, sh, w in wins:
        proba = rf.predict_proba(w[FEATURES].to_numpy(float))
        pred = [classes[i] for i in proba.argmax(1)]
        plain.append(pd.Series(pred).mode().iloc[0])
        r = m2.classify_window(w)
        ours.append(r.attack_class if not r.is_zero_day else "Unknown")
        msp_conf.append(float(proba.max(1).mean()))
        zd_scores.append(r.zero_day_score)
    # the "ours" label without the zero-day check: rerun the vote only
    vote_only = []
    for c, sh, w in wins:
        proba = rf.predict_proba(w[FEATURES].to_numpy(float))
        pred = [classes[i] for i in proba.argmax(1)]
        att = [p for p in pred if p != NORMAL_LABEL]
        vote_only.append(pd.Series(att).mode().iloc[0] if len(att) >= 0.3 * len(pred) else NORMAL_LABEL)
    known_mask = ~held
    df = pd.DataFrame({"label": labels, "share": [w[1] for w in wins], "plain": plain, "vote": vote_only,
                       "msp": msp_conf, "zd": zd_scores})
    kn = df[known_mask & attack]
    out["B_voting"] = {
        "Plain majority vote": {"accuracy": float((kn.plain == kn.label).mean()),
                                "by_share": kn.assign(c=kn.plain == kn.label).groupby("share")["c"].mean().to_dict()},
        "Attack-vote rule (ours)": {"accuracy": float((kn.vote == kn.label).mean()),
                                    "by_share": kn.assign(c=kn.vote == kn.label).groupby("share")["c"].mean().to_dict()},
    }
    # MSP threshold calibrated like the earlier work: 10th percentile of confidence on correctly classified known windows
    msp_thr = float(np.percentile(kn[kn.vote == kn.label].msp, 10))
    mir = df[held]
    out["B_zero_day"] = {
        "No zero-day check": {"mirai_unknown": 0.0, "known_false_unknown": 0.0},
        "Max-softmax-probability (MSP)": {"mirai_unknown": float((mir.msp < msp_thr).mean()),
                                          "known_false_unknown": float((kn.msp < msp_thr).mean()), "threshold": msp_thr},
        "Nearest-neighbour distance (ours)": {"mirai_unknown": float((mir.zd > 1).mean()),
                                              "known_false_unknown": float((kn.zd > 1).mean())},
        "mean_msp_known": float(kn.msp.mean()), "mean_msp_mirai": float(mir.msp.mean()),
    }
    mlp = P.metrics["module2"]["mlp"]
    out["B_quantisation"] = {"FP32 MLP": mlp["fp32"], "INT8 MLP": mlp["int8"],
                             "agreement": mlp["agreement_fp32_int8"]}

    # ---------------- C: Module 3 ----------------------------------------------------------
    m3 = P.m3
    rows = []
    for c in attack_classes:
        for _ in range(5):
            w = P.make_window(c, 0.8, seed=int(rng.integers(1e9)))
            r = P.run(w, mitigate=False)
            if r.m3 is None:
                continue
            mask = np.asarray(r.m2.attack_flow_mask, bool)
            benign = (w["label"] == NORMAL_LABEL).to_numpy()
            plan_full = m3.execute_and_verify(r.m3, w, mask)
            e = np.array([s.action.efficacy for s in plan_full.steps])
            collateral = (mask & benign).sum() / max(benign.sum(), 1) * (1 - np.prod(1 - e))
            row = {"label": c, "route": plan_full.route, "resolved_full": plan_full.resolved,
                   "resolved_stage1": plan_full.steps[0].resolved, "stages": len(plan_full.steps),
                   "collateral_ours": collateral, "risk_after_stage1": plan_full.steps[0].risk_after}
            if plan_full.route == "zero_day_behaviour":
                # without shape routing every zero-day window gets the generic "unclassified" ladder
                generic = m3.plan(r.m2.attack_class, r.m2.confidence, True, w, targeted=mask)
                generic.actions = [ACTIONS[k] for k in ("RATE_LIMIT_SOURCE", "BLOCK_SOURCE", "ISOLATE_SEGMENT")]
                g = m3.execute_and_verify(generic, w, mask)
                row.update({"generic_resolved": g.resolved, "generic_stages": len(g.steps),
                            "generic_disruptive": g.needed_disruptive_action,
                            "shape_disruptive": plan_full.needed_disruptive_action})
            rows.append(row)
    c3 = pd.DataFrame(rows)
    zd = c3[c3.route == "zero_day_behaviour"]
    out["C_module3"] = {
        "windows": len(c3),
        "Full (escalation + verification)": {"resolved": float(c3.resolved_full.mean()),
                                             "mean_stages": float(c3.stages.mean())},
        "No escalation (first action only)": {"resolved": float(c3.resolved_stage1.mean())},
        "No verification (one-shot, assumed done)": {
            "attacks_left_running_unnoticed": float(1 - c3.resolved_stage1.mean())},
        "Legitimate flows dropped": {"ours": float(c3.collateral_ours.mean()), "drop whole window": 1.0},
        "Zero-day: shape routing (ours) vs generic ladder": {
            "windows": len(zd),
            "shape_resolved": float(zd.resolved_full.mean()) if len(zd) else None,
            "generic_resolved": float(zd.generic_resolved.mean()) if len(zd) else None,
            "shape_mean_stages": float(zd.stages.mean()) if len(zd) else None,
            "generic_mean_stages": float(zd.generic_stages.mean()) if len(zd) else None,
            "shape_needed_blocking": float(zd.shape_disruptive.mean()) if len(zd) else None,
            "generic_needed_blocking": float(zd.generic_disruptive.mean()) if len(zd) else None,
        },
    }

    # ---------------- D: gating ------------------------------------------------------------
    stream = []
    for seed in range(10):
        stream += P.stream(MIXED_DAY, 0.8, seed=seed)
    t_m1 = t_m2 = 0.0
    flagged = benign_plans_gated = benign_plans_ungated = 0
    n_benign = 0
    for w in stream:
        is_benign = P.true_label_of(w) == NORMAL_LABEL
        n_benign += is_benign
        t0 = time.perf_counter(); r1 = P.m1.analyse_window(w); t_m1 += time.perf_counter() - t0
        t0 = time.perf_counter(); r2 = P.m2.classify_window(w); t_m2 += time.perf_counter() - t0
        m2_attack = r2.family != "Benign"
        if r1.forward_to_module2:
            flagged += 1
            benign_plans_gated += is_benign and m2_attack
        benign_plans_ungated += is_benign and m2_attack
    n = len(stream)
    m2_ms = t_m2 / n * 1000
    out["D_gating"] = {
        "windows": n, "benign_windows": n_benign,
        "With Module 1 gating (ours)": {"module2_calls": flagged, "ms_per_window": (t_m1 + flagged * t_m2 / n) / n * 1000,
                                        "benign_windows_given_a_plan": int(benign_plans_gated)},
        "Module 2 on every window": {"module2_calls": n, "ms_per_window": m2_ms,
                                     "benign_windows_given_a_plan": int(benign_plans_ungated)},
        "module1_ms": t_m1 / n * 1000, "module2_ms": m2_ms,
    }
    (METRICS_DIR / "ablation_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
