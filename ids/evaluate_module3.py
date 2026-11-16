"""
Evaluate the merged Module 3 end to end (Module 1 -> Module 2 -> Module 3), driven by Module 2's
PREDICTIONS (not ground truth), so classification errors carry through to mitigation.

    python -m ids.evaluate_module3          (after ids.prepare_data and ids.train_module2)

Two evaluations:
    A. coverage  : 15 windows per class for all 33 attack classes + 150 benign windows
                   (80% attack flows + 20% benign background), so every policy family is used
    B. stream    : the "Mixed day" scenario of the app, 10 seeds x 39 windows (end-to-end numbers)

Writes results/pipeline/module3_metrics.json and module3_per_window.csv.
"""

import json
from collections import Counter

import numpy as np
import pandas as pd

from ids.config import FAMILY_OF, HELD_OUT_CLASSES, METRICS_DIR, NORMAL_LABEL
from ids.pipeline import IDSPipeline

MIXED_DAY = [(NORMAL_LABEL, 8), ("DDoS-SYN_Flood", 4), (NORMAL_LABEL, 4), ("Recon-PortScan", 4),
             (NORMAL_LABEL, 4), ("Mirai-udpplain", 4), (NORMAL_LABEL, 4), ("DoS-UDP_Flood", 4),
             (NORMAL_LABEL, 3)]


def row_of(P, w, scenario):
    r = P.run(w)
    row = {"scenario": scenario, "true_label": r.true_label, "true_family": FAMILY_OF[r.true_label],
           "m1_risk": r.m1.risk, "m1_flagged": r.m1.forward_to_module2,
           "m2_class": r.m2.attack_class if r.m2 else None, "m2_zero_day": bool(r.m2 and r.m2.is_zero_day),
           "m3": r.m3 is not None}
    if r.m3:
        p = r.m3
        row.update({"route": p.route, "policy_family": p.family, "behaviour": p.behaviour,
                    "auto": p.auto_execute, "stages_used": len(p.steps), "resolved": p.resolved,
                    "first_action": p.steps[0].action.key, "final_action": p.steps[-1].action.key,
                    "needed_disruptive": p.needed_disruptive_action,
                    "risk_before": p.steps[0].risk_before, "risk_after": p.steps[-1].risk_after,
                    "cv_after": p.steps[-1].cv_after,
                    "volume_suppression_first": p.steps[0].volume_suppression})
    return row


def pct(x):
    return round(100 * float(x), 2)


def summarise(df):
    m3 = df[df.m3]
    att = df[df.true_label != NORMAL_LABEL]
    ben = df[df.true_label == NORMAL_LABEL]
    out = {
        "windows": len(df), "attack_windows": len(att), "benign_windows": len(ben),
        "module1_detected_attack_pct": pct(att.m1_flagged.mean()) if len(att) else None,
        "module1_false_alarm_pct": pct(ben.m1_flagged.mean()) if len(ben) else None,
        "bypassed_module2_pct": pct(1 - df.m1_flagged.mean()),
        "reached_module3": len(m3),
        "route_class_policy_pct": pct((m3.route == "class_policy").mean()) if len(m3) else None,
        "route_zero_day_pct": pct((m3.route == "zero_day_behaviour").mean()) if len(m3) else None,
        "resolved_pct": pct(m3.resolved.mean()) if len(m3) else None,
        "resolved_at_stage": {int(k): int(v) for k, v in m3[m3.resolved].stages_used.value_counts().sort_index().items()},
        "auto_execute_pct": pct(m3.auto.mean()) if len(m3) else None,
        "needed_disruptive_action_pct": pct(m3.needed_disruptive.mean()) if len(m3) else None,
        "mean_risk_before": round(float(m3.risk_before.mean()), 4) if len(m3) else None,
        "mean_risk_after": round(float(m3.risk_after.mean()), 4) if len(m3) else None,
        "mean_cv_after": round(float(m3.cv_after.mean()), 4) if len(m3) else None,
        "mean_volume_suppression_first_action_pct": pct(m3.volume_suppression_first.mean()) if len(m3) else None,
        "benign_windows_reaching_module3": int((ben.m3).sum()),
        "benign_windows_blocked": int(ben[ben.m3].needed_disruptive.astype(bool).sum()) if ben.m3.any() else 0,
    }
    return out


def main():
    P = IDSPipeline()
    rng = np.random.default_rng(42)
    rows = []

    # A. coverage over every class
    for cls in P.classes:
        n = 150 if cls == NORMAL_LABEL else 15
        for _ in range(n):
            rows.append(row_of(P, P.make_window(cls, 0.8, seed=int(rng.integers(1e9))), "coverage"))
    # B. mixed-day stream, 10 seeds
    for seed in range(10):
        for w in P.stream(MIXED_DAY, 0.8, seed=seed):
            rows.append(row_of(P, w, "mixed_day"))

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "module3_per_window.csv", index=False)
    cov = df[df.scenario == "coverage"]
    m3 = cov[cov.m3]
    known = m3[~m3.true_label.isin(HELD_OUT_CLASSES)]
    mirai = cov[cov.true_label.isin(HELD_OUT_CLASSES)]
    mirai_m3 = mirai[mirai.m3]

    per_family = (m3.groupby("policy_family")
                  .agg(windows=("resolved", "size"), resolved_pct=("resolved", "mean"),
                       auto_pct=("auto", "mean"), stage1_resolved=("stages_used", lambda s: (s == 1).mean()),
                       volume_suppression=("volume_suppression_first", "mean"))
                  .round(4))
    metrics = {
        "note": ("Driven by Module 2 predictions. The network controller is simulated: an action removes "
                 "each flow Module 2 marked as attack with the action's assumed efficacy, removed flows are "
                 "replaced by benign traffic, and Module 1 is re-run on the result. 'resolved' = risk < 0.35."),
        "coverage": summarise(cov),
        "coverage_per_policy_family": per_family.reset_index().to_dict(orient="records"),
        "known_attacks_routed_to_correct_policy_pct": pct(
            (known.policy_family == known.true_label.map(P.m3.family_of)).mean()) if len(known) else None,
        "zero_day": {
            "mirai_windows": len(mirai), "reached_module3": int(mirai.m3.sum()),
            "routed_zero_day_pct": pct((mirai_m3.route == "zero_day_behaviour").mean()),
            "behaviour_distribution": mirai_m3.behaviour.value_counts().to_dict(),
            "resolved_pct": pct(mirai_m3.resolved.mean()),
        },
        "mixed_day_stream": summarise(df[df.scenario == "mixed_day"]),
        "first_actions": Counter(m3.first_action).most_common(),
    }
    (METRICS_DIR / "module3_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(json.dumps(metrics, indent=1, default=str))


if __name__ == "__main__":
    main()
