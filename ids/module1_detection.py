"""
MODULE 1: IAT-based anomaly detection with a Mamdani fuzzy inference system.

Question answered: "Is this traffic window anomalous?"

For each window of W = 20 consecutive flows:
    1. mean IAT            mu    = mean(IAT_i)
    2. std  IAT            sigma = std(IAT_i)
    3. coefficient of var. CV    = sigma / mu          (floods -> CV ~ 0, scans -> CV high)
    4. mean packet rate    R     = mean(Rate_i)
    5. IAT deviation       D     = |mu - mu_normal| / mu_normal   (mu_normal from benign training data)
    6. behaviour deviation B     = median distance of the window's flows to the nearest benign
                                   training flows / benign threshold       (B > 1: unlike normal)
R is the window MEDIAN rate, so one extreme flow cannot raise a false alarm on its own.

Two modes:
    "timing"  : inputs CV, R, D and the 13 original rules (the paper's IAT-only design)
    "full"    : adds B. Timing catches floods (CV -> 0), B catches attacks whose timing looks
                normal (recon, spoofing, web, brute force). B is learned from BENIGN traffic only,
                so Module 1 stays unsupervised and still detects unseen attacks.
The fuzzy system turns the inputs into an attack risk in [0, 1]:
    risk < 0.35 NORMAL | 0.35-0.65 SUSPICIOUS | >= 0.65 ATTACK
SUSPICIOUS and ATTACK windows are handed to Module 2.

The membership functions and rules are the same as src/fuzzy_system.py (skfuzzy). They
are re-implemented here in plain NumPy so the app can show every membership degree and
rule firing strength, and so a whole stream can be scored quickly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ids.config import RISK_ATTACK, RISK_SUSPICIOUS

EPS = 1e-12
CV_MAX, RATE_MAX, DEV_MAX, BEH_MAX = 6.0, 50_000.0, 5.0, 10.0
RISK_UNIVERSE = np.linspace(0.0, 1.0, 201)


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + EPS), (x >= b).astype(float))
    right = np.where(c > b, (c - x) / (c - b + EPS), (x <= b).astype(float))
    return np.clip(np.minimum(left, right), 0.0, 1.0)


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.where(b > a, (x - a) / (b - a + EPS), (x >= b).astype(float))
    right = np.where(d > c, (d - x) / (d - c + EPS), (x <= c).astype(float))
    return np.clip(np.minimum(np.minimum(left, 1.0), right), 0.0, 1.0)


# --- Membership functions (identical to src/fuzzy_system.py) ------------------------
def membership_functions(cv_median: float = 1.0):
    return {
        "cv": {
            "low": lambda x: trimf(x, 0.0, 0.0, 0.65),
            "medium": lambda x: trimf(x, 0.50, cv_median, 1.55),
            "high": lambda x: trapmf(x, 1.40, 1.80, CV_MAX, CV_MAX),
        },
        "rate": {
            "low": lambda x: trimf(x, 0.0, 0.0, 350.0),
            "medium": lambda x: trimf(x, 200.0, 600.0, 1500.0),
            "high": lambda x: trapmf(x, 1000.0, 2500.0, RATE_MAX, RATE_MAX),
        },
        "dev": {
            "low": lambda x: trimf(x, 0.0, 0.0, 0.18),
            "medium": lambda x: trimf(x, 0.12, 0.35, 0.65),
            "high": lambda x: trapmf(x, 0.50, 0.90, DEV_MAX, DEV_MAX),
        },
        "beh": {
            # B is scaled so that B = 1 is the 99th percentile of benign windows
            "low": lambda x: trimf(x, 0.0, 0.0, 1.15),
            "medium": lambda x: trimf(x, 1.0, 1.3, 1.7),
            "high": lambda x: trapmf(x, 1.4, 1.9, BEH_MAX, BEH_MAX),
        },
        "risk": {
            "low": lambda x: trimf(x, 0.0, 0.0, 0.35),
            "medium": lambda x: trimf(x, 0.25, 0.50, 0.75),
            "high": lambda x: trapmf(x, 0.60, 0.85, 1.0, 1.0),
        },
    }


# (rule id, antecedents {input: term}, output term, plain-English meaning)
RULES_TIMING = [
    ("R1", {"cv": "medium", "rate": "low", "dev": "low"}, "low", "Normal IoT behaviour: natural jitter, low rate, usual timing"),
    ("R2", {"cv": "low", "rate": "low", "dev": "low"}, "low", "Regular low-rate sensor beaconing"),
    ("R3", {"cv": "medium", "rate": "low", "dev": "medium"}, "low", "Normal traffic with slightly shifted timing"),
    ("R4", {"cv": "high", "rate": "high"}, "high", "Bursty high-volume traffic (burst DDoS)"),
    ("R5", {"cv": "high", "dev": "high"}, "high", "Erratic timing far from normal (scan / pulse attack)"),
    ("R6", {"rate": "high", "dev": "high"}, "high", "Volumetric flood with shifted timing"),
    ("R7", {"cv": "medium", "rate": "high"}, "high", "High-rate flood that keeps normal-looking jitter"),
    ("R8", {"cv": "high", "rate": "medium"}, "high", "Irregular medium-rate burst attack"),
    ("R9", {"cv": "medium", "rate": "medium", "dev": "medium"}, "medium", "Borderline activity: worth a closer look"),
    ("R10", {"rate": "high"}, "high", "Packet rate alone is far above normal"),
    ("R11", {"dev": "high"}, "high", "Packet timing has collapsed or stretched abnormally"),
    ("R12", {"cv": "low", "rate": "high"}, "high", "Machine-like constant-rate flood (DoS / Mirai)"),
    ("R13", {"cv": "high", "rate": "low"}, "medium", "Slow, irregular probing (stealthy scan)"),
]

# Full mode: the "normal" rules also require normal behaviour, plus two behaviour rules.
RULES_FULL = [
    ("R1", {"cv": "medium", "rate": "low", "dev": "low", "beh": "low"}, "low", "Normal IoT behaviour: natural jitter, low rate, usual timing, normal flows"),
    ("R2", {"cv": "low", "rate": "low", "dev": "low", "beh": "low"}, "low", "Regular low-rate sensor beaconing with normal flows"),
    ("R3", {"cv": "medium", "rate": "low", "dev": "medium", "beh": "low"}, "low", "Normal traffic with slightly shifted timing"),
] + RULES_TIMING[3:] + [
    ("R14", {"beh": "high"}, "high", "Flows look unlike any normal traffic (learned from benign data only)"),
    ("R15", {"beh": "medium"}, "medium", "Flows are somewhat unusual: worth a closer look"),
]
RULES = RULES_FULL


def verdict_of(risk: float) -> str:
    if risk >= RISK_ATTACK:
        return "ATTACK"
    if risk >= RISK_SUSPICIOUS:
        return "SUSPICIOUS"
    return "NORMAL"


@dataclass
class Module1Result:
    features: dict
    memberships: dict          # {"cv": {"low": .., ..}, ...}
    fired_rules: list          # [{id, strength, output, meaning}], strongest first
    risk: float
    verdict: str
    forward_to_module2: bool
    aggregated: np.ndarray = field(repr=False, default=None)


class FuzzyAnomalyDetector:
    def __init__(self, baseline: dict, behaviour: dict | None = None, mode: str = "full"):
        """
        baseline : {"mu_normal": benign mean IAT, "cv_median": benign window CV median, ...}
        behaviour: {"scaler", "nn" (NearestNeighbors on benign flows), "threshold", "features"}
        """
        self.baseline = baseline
        self.behaviour = behaviour
        self.mode = mode if behaviour is not None else "timing"
        self.rules = RULES_FULL if self.mode == "full" else RULES_TIMING
        self.inputs = ("cv", "rate", "dev", "beh") if self.mode == "full" else ("cv", "rate", "dev")
        self.mu_normal = float(baseline["mu_normal"])
        self.mf = membership_functions(float(baseline.get("cv_median", 1.0)))
        self._risk_mf = {t: f(RISK_UNIVERSE) for t, f in self.mf["risk"].items()}

    # -- Step 1: window statistics --------------------------------------------------
    def window_features(self, window: pd.DataFrame) -> dict:
        iat = window["IAT"].to_numpy(dtype=float)
        mu, sigma = iat.mean(), iat.std()
        f = {
            "n_flows": len(window),
            "mean_iat": mu,
            "std_iat": sigma,
            "cv": sigma / (mu + EPS),
            "rate": float(window["Rate"].median()),
            "dev": abs(mu - self.mu_normal) / (self.mu_normal + EPS),
            "beh": None,
        }
        if self.behaviour is not None:
            f["beh"] = self.behaviour_score(window)
        return f

    def behaviour_score(self, window: pd.DataFrame) -> float:
        b = self.behaviour
        X = window[b["features"]].to_numpy(dtype=float)
        X = np.sign(X) * np.log1p(np.abs(X))
        d, _ = b["nn"].kneighbors(b["scaler"].transform(X))
        return float(np.median(d.mean(axis=1)) / b["threshold"])

    # -- Step 2-4: fuzzify, infer, defuzzify ---------------------------------------
    def infer(self, cv: float, rate: float, dev: float, beh: float | None = None) -> Module1Result:
        x = {
            "cv": float(np.clip(cv, 0, CV_MAX)),
            "rate": float(np.clip(rate, 0, RATE_MAX)),
            "dev": float(np.clip(dev, 0, DEV_MAX)),
            "beh": float(np.clip(beh if beh is not None else 0.0, 0, BEH_MAX)),
        }
        mem = {v: {t: float(f(x[v])) for t, f in self.mf[v].items()} for v in self.inputs}

        aggregated = np.zeros_like(RISK_UNIVERSE)
        fired = []
        for rid, ante, out, meaning in self.rules:
            strength = min(mem[v][t] for v, t in ante.items())      # AND = min
            if strength > 0:
                fired.append({"id": rid, "strength": strength, "output": out, "meaning": meaning,
                              "if": " AND ".join(f"{v.upper()} is {t.upper()}" for v, t in ante.items())})
                # Mamdani implication (clip) + aggregation (max)
                aggregated = np.maximum(aggregated, np.minimum(strength, self._risk_mf[out]))

        if aggregated.sum() > 0:
            risk = float((RISK_UNIVERSE * aggregated).sum() / aggregated.sum())   # centroid
        else:
            risk = 0.5  # no rule fired: undecided, same fallback as the skfuzzy version
        verdict = verdict_of(risk)
        fired.sort(key=lambda r: -r["strength"])
        return Module1Result(
            features={"cv": cv, "rate": rate, "dev": dev, "beh": beh},
            memberships=mem, fired_rules=fired, risk=risk, verdict=verdict,
            forward_to_module2=verdict != "NORMAL", aggregated=aggregated,
        )

    def analyse_window(self, window: pd.DataFrame) -> Module1Result:
        f = self.window_features(window)
        res = self.infer(f["cv"], f["rate"], f["dev"], f["beh"])
        res.features = f
        return res




def build_windows(df: pd.DataFrame, window_size: int, step: int | None = None) -> list[pd.DataFrame]:
    step = step or window_size // 2
    return [df.iloc[s:s + window_size] for s in range(0, len(df) - window_size + 1, step)]
