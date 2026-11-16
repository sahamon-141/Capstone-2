"""
Figures for Section 4 (ablation) and Section 5 (comparison with existing algorithms).

    python -m ids.paper_comparison_figures     (after ids.comparison and ids.ablation)

Grouped bar charts in the style of the reference paper: existing algorithms by short name,
IATFCM last (green), across a varying condition (attack share of the window).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ids.config import METRICS_DIR

C = json.loads((METRICS_DIR / "comparison_metrics.json").read_text())
A = json.loads((METRICS_DIR / "ablation_metrics.json").read_text())
PALETTE = ["#4472c4", "#ed7d31", "#a5a5a5", "#ffc000", "#5b9bd5", "#70ad47"]
GREEN = "#70ad47"


def grouped(ax, groups, series, values, ylabel, ylim=(0, 105), fmt="{:.0f}", colors=None, label_bars=False):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    colors = colors or PALETTE
    for i, s in enumerate(series):
        v = values[s]
        bars = ax.bar(x + (i - (n - 1) / 2) * w, v, w, label=s, color=colors[i % len(colors)], edgecolor="white", lw=0.5)
        if label_bars:
            for b, val in zip(bars, v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + ylim[1] * 0.01, fmt.format(val),
                        ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x, groups); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=min(n, 5), loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)


def save(fig, name):
    fig.tight_layout(); fig.savefig(METRICS_DIR / name, dpi=170); plt.close(fig)


def pct(d, keys):
    return [100 * float(d[str(k)] if str(k) in d else d[k]) for k in keys]


def main():
    shares = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]
    share_lbl = [f"{int(float(s) * 100)}%" for s in shares]
    det = C["detection"]
    det_methods = ["DT-3sigma", "SAE-PSVM", "PP-TTF", "IATFCM"]
    det_names = {"DT-3sigma": "DT-3σ", "SAE-PSVM": "SAE-PSVM", "PP-TTF": "PP-TTF", "IATFCM": "IATFCM"}
    det_cols = [PALETTE[0], PALETTE[1], PALETTE[2], GREEN]

    # ---- Fig: detection rate vs attack share --------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    grouped(ax, share_lbl, [det_names[m] for m in det_methods],
            {det_names[m]: pct(det[m]["recall_by_share"], shares) for m in det_methods},
            "Attack windows detected (%)", colors=det_cols)
    ax.set_xlabel("Share of attack records in the window")
    save(fig, "fig_cmp_detection_by_share.png")

    # ---- Fig: precision / recall / F1 / FPR ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    mets = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score"), ("fpr", "False-positive rate")]
    grouped(ax, [m[1] for m in mets], [det_names[m] for m in det_methods],
            {det_names[m]: [100 * det[m][k] for k, _ in mets] for m in det_methods}, "%", colors=det_cols,
            label_bars=True, fmt="{:.1f}")
    save(fig, "fig_cmp_detection_metrics.png")

    # ---- Fig: detection by family incl. unseen Mirai ---------------------------------------
    fams = ["DDoS", "DoS", "Recon", "Spoofing", "Web", "BruteForce", "Mirai"]
    fam_lbl = ["DDoS", "DoS", "Recon", "Spoofing", "Web", "Brute force", "Mirai\n(unseen)"]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    grouped(ax, fam_lbl, [det_names[m] for m in det_methods],
            {det_names[m]: [100 * det[m]["recall_by_family"].get(f, 0) for f in fams] for m in det_methods},
            "Attack windows detected (%)", colors=det_cols)
    save(fig, "fig_cmp_detection_family.png")

    # ---- Fig: classification accuracy vs share --------------------------------------------
    cls = C["classification"]
    cls_methods = ["PP-TTF (cls)", "DNN-KDQ", "HFF-XGB", "IATFCM-INT8", "IATFCM-RF"]
    cls_names = {"PP-TTF (cls)": "PP-TTF", "DNN-KDQ": "DNN-KDQ", "HFF-XGB": "HFF-XGB",
                 "IATFCM-INT8": "IATFCM (INT8)", "IATFCM-RF": "IATFCM (RF)"}
    cls_cols = [PALETTE[2], PALETTE[3], PALETTE[1], "#a9d18e", GREEN]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    grouped(ax, share_lbl, [cls_names[m] for m in cls_methods],
            {cls_names[m]: pct(cls[m]["accuracy_by_share"], shares) for m in cls_methods},
            "Windows classified correctly (%)", colors=cls_cols)
    ax.set_xlabel("Share of attack records in the window")
    save(fig, "fig_cmp_classification_by_share.png")

    # ---- Fig: classification summary incl. zero-day flagging -------------------------------
    mets = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("family_accuracy", "Family accuracy"),
            ("zero_day_flagged_unknown", "Unseen Mirai\nflagged unknown")]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    grouped(ax, [m[1] for m in mets], [cls_names[m] for m in cls_methods],
            {cls_names[m]: [100 * cls[m][k] for k, _ in mets] for m in cls_methods}, "%", colors=cls_cols,
            label_bars=True, fmt="{:.1f}")
    save(fig, "fig_cmp_classification_metrics.png")

    # ---- Fig: computational cost ----------------------------------------------------------
    R = C["resources"]
    order = ["DT-3sigma", "SAE-PSVM", "PP-TTF", "IATFCM", "PP-TTF (cls)", "DNN-KDQ", "HFF-XGB", "IATFCM-INT8", "IATFCM-RF"]
    names = ["DT-3σ", "SAE-PSVM", "PP-TTF\n(det.)", "IATFCM\nModule 1", "PP-TTF\n(cls.)", "DNN-KDQ", "HFF-XGB",
             "IATFCM\nINT8", "IATFCM\nRF"]
    cols = [PALETTE[0], PALETTE[1], PALETTE[2], GREEN, PALETTE[2], PALETTE[3], PALETTE[1], "#a9d18e", GREEN]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    a1.bar(names, [R[m]["latency_ms"] for m in order], color=cols)
    a1.set_ylabel("ms per window (20 records)"); a1.set_title("(a) Inference latency", fontsize=10)
    a1.axvline(3.5, color="k", lw=0.8, ls=":"); a1.tick_params(axis="x", labelsize=7.5)
    a2.bar(names, [R[m]["size_kb"] for m in order], color=cols); a2.set_yscale("log")
    a2.set_ylabel("KB (log scale)"); a2.set_title("(b) Model size", fontsize=10)
    a2.axvline(3.5, color="k", lw=0.8, ls=":"); a2.tick_params(axis="x", labelsize=7.5)
    for a in (a1, a2):
        a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
    save(fig, "fig_cmp_cost.png")

    # ================= Section 4: ablation =================================================
    m1 = A["A_module1"]
    variants = list(m1.keys())
    short = {"Timing only, mean rate (original)": "Timing only\n(mean rate)",
             "Timing only, median rate": "Timing only\n(median rate)",
             "Behaviour score B alone (B > 1)": "B alone",
             "Timing + B, original R13": "Timing + B\n(original R13)",
             "Full IATFCM (timing + B, gated R13)": "IATFCM\n(full)"}
    ab_cols = [PALETTE[2], PALETTE[0], PALETTE[3], PALETTE[1], GREEN]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    mets = [("recall", "Recall"), ("precision", "Precision"), ("f1", "F1-score"), ("fpr", "FPR"),
            ("zero_day_detection", "Unseen Mirai")]
    grouped(ax, [m[1] for m in mets], [short[v].replace("\n", " ") for v in variants],
            {short[v].replace("\n", " "): [100 * m1[v][k] for k, _ in mets] for v in variants}, "%", colors=ab_cols,
            label_bars=True, fmt="{:.1f}")
    save(fig, "fig_abl_module1.png")

    v = A["B_voting"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    grouped(a1, share_lbl, list(v.keys()), {k: pct(v[k]["by_share"], shares) for k in v},
            "Windows classified correctly (%)", colors=[PALETTE[2], GREEN])
    a1.set_xlabel("Share of attack records in the window"); a1.set_title("(a) Window voting rule", fontsize=10, pad=28)
    z = A["B_zero_day"]
    zk = ["No zero-day check", "Max-softmax-probability (MSP)", "Nearest-neighbour distance (ours)"]
    zl = ["No check", "MSP", "NN distance\n(ours)"]
    grouped(a2, zl, ["Unseen Mirai flagged unknown", "Known attacks wrongly unknown"],
            {"Unseen Mirai flagged unknown": [100 * z[k]["mirai_unknown"] for k in zk],
             "Known attacks wrongly unknown": [100 * z[k]["known_false_unknown"] for k in zk]},
            "%", colors=[GREEN, PALETTE[1]], label_bars=True, fmt="{:.1f}")
    a2.set_title("(b) Zero-day check", fontsize=10, pad=28)
    save(fig, "fig_abl_module2.png")

    c = A["C_module3"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    labels = ["Full ladder +\nverification (ours)", "First action only\n(no escalation)"]
    vals = [100 * c["Full (escalation + verification)"]["resolved"], 100 * c["No escalation (first action only)"]["resolved"]]
    b = a1.bar(labels, vals, color=[GREEN, PALETTE[2]])
    for bb, vv in zip(b, vals):
        a1.text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.1f}%", ha="center", fontsize=9)
    a1.set_ylim(0, 105); a1.set_ylabel("Attacks resolved (%)"); a1.set_title("(a) Escalation", fontsize=10)
    labels = ["Targeted filtering\n(ours)", "Drop the whole\nflagged window"]
    vals = [100 * c["Legitimate flows dropped"]["ours"], 100 * c["Legitimate flows dropped"]["drop whole window"]]
    b = a2.bar(labels, vals, color=[GREEN, PALETTE[1]])
    for bb, vv in zip(b, vals):
        a2.text(bb.get_x() + bb.get_width() / 2, vv + 1, f"{vv:.1f}%", ha="center", fontsize=9)
    a2.set_ylim(0, 105); a2.set_ylabel("Legitimate flows dropped (%)"); a2.set_title("(b) Collateral damage", fontsize=10)
    for a in (a1, a2):
        a.grid(axis="y", alpha=0.3); a.set_axisbelow(True)
    save(fig, "fig_abl_module3.png")
    print("figures written to", METRICS_DIR)


if __name__ == "__main__":
    main()
