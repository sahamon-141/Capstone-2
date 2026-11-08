"""
MODULE 2: multiclass attack identification (only runs on windows Module 1 flagged).

Question answered: "Which attack is this, or is it something we have never seen?"

Models (trained by ids/train_module2.py, Mirai family held out as a zero-day proxy):
    * Random Forest (FP32)  - main classifier, 31 known classes, per-flow prediction
    * INT8 MLP              - edge model: a small neural network whose weights AND inputs are
                              quantised to 8-bit integers (4x smaller, integer arithmetic)
    * Zero-day detector     - nearest-neighbour distance of the window to known training
                              traffic; too far from everything known  ->  "Unknown / zero-day"

A window has 20 flows. Each flow is classified, then the window label is a majority vote
over the flows predicted as attacks (benign background flows are ignored if attacks dominate).
"""

import pickle
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ids.config import FAMILY_OF, MODEL_DIR, NORMAL_LABEL

FEATURES = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count", "fin_count", "urg_count",
    "rst_count", "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP", "DHCP",
    "ARP", "ICMP", "IPv", "LLC", "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT",
    "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]


def log_transform(X: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed counters (rates, byte totals) so distances and the MLP behave."""
    return np.sign(X) * np.log1p(np.abs(X))


# ------------------------------------------------------------------------------------
# INT8 quantisation  (q = round(r / s) + z, clipped to [-128, 127])
# ------------------------------------------------------------------------------------
class Int8Quantizer:
    """Per-feature affine quantiser fitted on training min/max."""

    def fit(self, X):
        self.rmin, self.rmax = X.min(axis=0), X.max(axis=0)
        self.scale = np.maximum((self.rmax - self.rmin) / 255.0, 1e-12)
        self.zero = np.round(-128 - self.rmin / self.scale)
        return self

    def quantize(self, X):
        return np.clip(np.round(X / self.scale) + self.zero, -128, 127).astype(np.int8)

    def dequantize(self, Q):
        return (Q.astype(np.float32) - self.zero) * self.scale


class Int8MLP:
    """
    Post-training INT8 quantisation of a scikit-learn MLP.
    Weights: symmetric per-layer int8.  Activations: int8 with calibrated scales.
    Matrix products run in int32 (as on an MCU / DSP), then are rescaled.
    """

    def __init__(self, mlp, quantizer: Int8Quantizer, X_calib_q: np.ndarray):
        self.classes_ = mlp.classes_
        self.in_q = quantizer
        self.layers = []
        a = quantizer.dequantize(X_calib_q)
        a_scale = None
        for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            w_scale = np.abs(W).max() / 127.0
            Wq = np.clip(np.round(W / w_scale), -127, 127).astype(np.int8)
            if i == 0:
                # input already int8 with per-feature scale/zero: fold into first layer in float
                self.first = (W.astype(np.float32), b.astype(np.float32))
                a = np.maximum(a @ W + b, 0)
                a_scale = np.percentile(a, 99.9) / 127.0 + 1e-12
                continue
            self.layers.append((Wq, w_scale, b.astype(np.float32), a_scale))
            a = a @ W + b
            if i < len(mlp.coefs_) - 1:
                a = np.maximum(a, 0)
                a_scale = np.percentile(a, 99.9) / 127.0 + 1e-12

    def predict_proba(self, Xq: np.ndarray) -> np.ndarray:
        x = self.in_q.dequantize(Xq)
        W, b = self.first
        a = np.maximum(x @ W + b, 0)
        for j, (Wq, w_scale, bias, a_scale) in enumerate(self.layers):
            aq = np.clip(np.round(a / a_scale), -128, 127).astype(np.int32)   # int8 activations
            acc = aq @ Wq.astype(np.int32)                                    # int32 accumulate
            a = acc.astype(np.float32) * (a_scale * w_scale) + bias
            if j < len(self.layers) - 1:
                a = np.maximum(a, 0)
        a = a - a.max(axis=1, keepdims=True)
        e = np.exp(a)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, Xq):
        return self.classes_[self.predict_proba(Xq).argmax(axis=1)]

    def nbytes(self) -> int:
        n = sum(w.nbytes for w in self.first) // 4  # first layer stored as int8 too on device
        return n + sum(Wq.nbytes + bias.nbytes for Wq, _, bias, _ in self.layers)


# ------------------------------------------------------------------------------------
# Inference wrapper used by the pipeline / app
# ------------------------------------------------------------------------------------
@dataclass
class Module2Result:
    attack_class: str
    family: str
    confidence: float
    top5: list                 # [(class, probability)]
    flow_votes: dict           # class -> number of flows voted
    zero_day_score: float      # distance ratio; > 1 means outside known traffic
    is_zero_day: bool
    model: str
    latency_ms: float
    attack_flow_mask: list     # per flow: True if predicted as an attack (Module 3 targets these)


class AttackClassifier:
    def __init__(self, model_dir=MODEL_DIR):
        with open(model_dir / "module2_bundle.pkl", "rb") as fh:
            b = pickle.load(fh)
        self.rf = b["rf"]
        self.rf.n_jobs = 1   # 20 rows per window: worker start-up would cost more than the prediction
        self.int8_mlp: Int8MLP = b["int8_mlp"]
        self.quantizer: Int8Quantizer = b["quantizer"]
        self.nn = b["nn"]                  # NearestNeighbors on scaled known training flows
        self.nn_scaler = b["nn_scaler"]
        self.zd_threshold = b["zd_threshold"]
        self.classes = list(self.rf.classes_)

    def zero_day_score(self, X: np.ndarray) -> float:
        Z = self.nn_scaler.transform(log_transform(X))
        d, _ = self.nn.kneighbors(Z, n_neighbors=1)
        return float(np.median(d[:, 0]) / self.zd_threshold)

    def classify_window(self, window: pd.DataFrame, model: str = "rf") -> Module2Result:
        X = window[FEATURES].to_numpy(dtype=np.float64)
        t0 = time.perf_counter()
        if model == "int8":
            proba = self.int8_mlp.predict_proba(self.quantizer.quantize(log_transform(X)))
            classes = list(self.int8_mlp.classes_)
        else:
            proba = self.rf.predict_proba(X)
            classes = self.classes
        latency = (time.perf_counter() - t0) * 1000

        flow_pred = [classes[i] for i in proba.argmax(axis=1)]
        votes = Counter(flow_pred)
        attack_votes = {c: n for c, n in votes.items() if c != NORMAL_LABEL}
        # Window label: the dominant attack class if attacks are a meaningful share of the window
        if attack_votes and sum(attack_votes.values()) >= 0.3 * len(flow_pred):
            label = max(attack_votes, key=attack_votes.get)
            rows = [i for i, c in enumerate(flow_pred) if c != NORMAL_LABEL]
        else:
            label = NORMAL_LABEL
            rows = list(range(len(flow_pred)))
        mean_p = proba[rows].mean(axis=0)
        order = np.argsort(mean_p)[::-1][:5]
        top5 = [(classes[i], float(mean_p[i])) for i in order]

        attack_rows = window.iloc[rows] if label != NORMAL_LABEL else window
        zd = self.zero_day_score(attack_rows[FEATURES].to_numpy(dtype=np.float64))
        # Far from all known traffic (attack or benign) -> do not force a known label on it
        is_zd = zd > 1.0
        return Module2Result(
            attack_class="Unknown (zero-day)" if is_zd else label,
            family="Unknown" if is_zd else FAMILY_OF.get(label, "Unknown"),
            confidence=float(mean_p[classes.index(label)]),
            top5=top5, flow_votes=dict(votes), zero_day_score=zd, is_zero_day=is_zd,
            model="INT8 MLP (edge)" if model == "int8" else "Random Forest",
            latency_ms=latency,
            attack_flow_mask=[c != NORMAL_LABEL for c in flow_pred],
        )
