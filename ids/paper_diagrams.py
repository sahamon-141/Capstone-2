"""
Illustrations for the paper (no data needed beyond the trained baseline):
    fig_membership_functions.png   membership functions of CV, rate, D, B and the risk output
    iat_diagram.png                packet inter-arrival times and overlapping sliding windows
    iot_attack_landscape.png       IoT network, attack types and their impact

    python -m ids.paper_diagrams
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from ids.config import METRICS_DIR
from ids.module1_detection import BEH_MAX, CV_MAX, DEV_MAX, membership_functions

TERM_COL = {"low": "#16a34a", "medium": "#f59e0b", "high": "#dc2626"}


def membership_figure():
    mf = membership_functions(1.0)
    panels = [("cv", r"$CV_t$ (coefficient of variation)", np.linspace(0, CV_MAX, 600)),
              ("rate", r"$\tilde{R}_t$ (median packet rate, pkt/s)", np.linspace(0, 5000, 600)),
              ("dev", r"$D_t$ (IAT deviation)", np.linspace(0, 2.0, 600)),
              ("beh", r"$B_t$ (behaviour deviation)", np.linspace(0, 4.0, 600)),
              ("risk", r"$A_t$ (attack risk)", np.linspace(0, 1, 600))]
    fig, axes = plt.subplots(3, 2, figsize=(11, 9))
    axes = axes.ravel()
    for ax, (var, title, x) in zip(axes, panels):
        for term, f in mf[var].items():
            ax.plot(x, f(x), lw=2.2, color=TERM_COL[term], label=term.capitalize())
        ax.set_title(title, fontsize=10); ax.set_ylim(-0.03, 1.08); ax.set_ylabel("membership")
        ax.grid(alpha=.3)
        if var == "risk":
            for a, b, c, n in [(0, .35, "#16a34a", "Normal"), (.35, .65, "#f59e0b", "Suspicious"), (.65, 1, "#dc2626", "Attack")]:
                ax.axvspan(a, b, color=c, alpha=.08); ax.text((a + b) / 2, 1.02, n, ha="center", fontsize=8)
        if var == "rate":
            ax.text(0.98, 0.9, "HIGH continues to 50,000 pkt/s", transform=ax.transAxes, ha="right", fontsize=8)
        ax.legend(fontsize=8, loc={"risk": "lower right", "rate": "center right"}.get(var, "center right"),
                  bbox_to_anchor=(1.0, 0.45) if var == "rate" else None)
    axes[-1].axis("off")
    axes[-1].text(0.02, 0.5, "Rules: AND = min, implication = clip,\naggregation = max, defuzzification = centroid\n\n"
                  "Decision: risk < 0.35 Normal,\n0.35-0.65 Suspicious, >= 0.65 Attack", fontsize=10, va="center")
    plt.tight_layout(); plt.savefig(METRICS_DIR / "fig_membership_functions.png", dpi=150); plt.close()


def iat_figure():
    rng = np.random.default_rng(3)
    t_norm = np.cumsum(rng.exponential(1.0, 14))
    t_flood = t_norm[-1] + 0.6 + np.arange(16) * 0.25
    t = np.concatenate([t_norm, t_flood])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.2), gridspec_kw={"height_ratios": [1, 1.3]})
    a1.vlines(t_norm, 0, 1, color="#16a34a", lw=2, label="normal packets")
    a1.vlines(t_flood, 0, 1, color="#dc2626", lw=2, label="flood packets")
    i = int(np.argmax(np.diff(t_norm)))       # widest gap, so the arrow is readable
    a1.annotate("", xy=(t[i + 1], 1.15), xytext=(t[i], 1.15), arrowprops=dict(arrowstyle="<->"))
    a1.text((t[i] + t[i + 1]) / 2, 1.27, r"$IAT_i = t_i - t_{i-1}$", ha="center", fontsize=10)
    a1.set_ylim(0, 1.6); a1.set_yticks([]); a1.set_xlabel("arrival time"); a1.legend(loc="upper right", fontsize=8)
    a1.set_title("Packet arrivals: irregular normal traffic, then a machine-regular flood", fontsize=10)
    iat = np.diff(t)
    col = ["#16a34a" if i < 13 else "#dc2626" for i in range(len(iat))]
    a2.bar(np.arange(len(iat)), iat, color=col)
    W, S = 10, 5
    for k, s in enumerate(range(0, len(iat) - W + 1, S)):
        y = iat.max() * (1.05 + 0.22 * k)
        a2.add_patch(FancyBboxPatch((s - .45, 0), W - .1, y, boxstyle="round,pad=0.02", fill=False,
                                    ec="#2563eb", lw=1.4, ls="--"))
        w = iat[s:s + W]
        a2.text(s + 0.1, y + 0.06, f"window {k + 1}: CV = {w.std() / w.mean():.2f}", ha="left", fontsize=8, color="#1e3a8a")
    a2.set_xlabel("consecutive IAT values"); a2.set_ylabel("IAT"); a2.set_ylim(0, iat.max() * 2.0)
    a2.set_title("Overlapping sliding windows (50% overlap): CV falls towards 0 when the flood starts", fontsize=10)
    plt.tight_layout(); plt.savefig(METRICS_DIR / "iat_diagram.png", dpi=150); plt.close()


def landscape_figure():
    fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=9, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec="#334155"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)

    # IoT network in the middle
    box(4.0, 2.6, 3.0, 1.2, "IoT gateway\n(resource constrained)", "#dbeafe", 10, True)
    devices = ["Camera", "Smart plug", "Sensor", "Thermostat", "Speaker"]
    for i, d in enumerate(devices):
        x = 3.6 + i * 0.8
        box(x - 0.35, 1.2, 0.75, 0.55, d, "#f1f5f9", 7)
        ax.plot([x, 5.5], [1.75, 2.6], color="#94a3b8", lw=1)
    attacks = [("DoS / DDoS floods", "overwhelm bandwidth\nand device resources"),
               ("Mirai botnet", "infected devices\nflood others"),
               ("Reconnaissance", "scan ports, hosts\nand vulnerabilities"),
               ("Spoofing / MITM", "impersonate gateway\n(ARP / DNS)"),
               ("Web attacks", "SQLi, XSS, command\ninjection, uploads"),
               ("Brute force", "guess device\npasswords")]
    for i, (name, desc) in enumerate(attacks):
        x, y = (0.2, 5.0 - i * 1.0) if i < 3 else (8.6, 5.0 - (i - 3) * 1.0)
        box(x, y, 2.2, 0.8, f"{name}\n{desc}", "#fee2e2", 7.5)
        ax.annotate("", xy=(4.0 if i < 3 else 7.0, 3.2), xytext=(x + (2.2 if i < 3 else 0), y + 0.4),
                    arrowprops=dict(arrowstyle="-|>", color="#dc2626", lw=1.3))
    impacts = ["Loss of availability", "Unauthorised access", "Data interception", "Resource exhaustion",
               "Compromised devices"]
    for i, imp in enumerate(impacts):
        box(0.4 + i * 2.1, 0.1, 1.9, 0.55, imp, "#fef3c7", 8)
    ax.text(5.5, 0.85, "Impact", ha="center", fontsize=9, fontweight="bold")
    ax.text(5.5, 6.2, "IoT network attack landscape", ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(METRICS_DIR / "iot_attack_landscape.png", dpi=150); plt.close()


if __name__ == "__main__":
    membership_figure(); iat_figure(); landscape_figure()
    print("written to", METRICS_DIR)
