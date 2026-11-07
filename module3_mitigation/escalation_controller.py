"""
Graduated containment with re-verification through Module 1.

The paper's mitigation loop is: apply the least disruptive action that fits,
keep watching the same timing statistics, and escalate only if the traffic does
not settle. This module implements that loop.

What is measured and what is modelled
-------------------------------------
The packet rate, CV and IAT deviation of each window are measured from the
dataset. Nothing here enforces anything on a real network, so the *effect* of an
action has to be modelled. The model is deliberately simple and is stated in
``ACTION_MODELS`` below:

* ``cap``   - an admission filter. The residual rate is ``min(observed_rate, C)``
              where ``C`` is a percentile of the normal-traffic rate taken from
              the baseline, so a source is clamped to roughly what a normal
              device sends. The default cap is the 75th percentile rather than
              the 95th: on CICIoT2023 the normal 95th percentile is about
              10,500 pkt/s, which is already inside the "rate is high" region of
              the fuzzy membership functions, so capping there would not change
              any verdict.
* ``stop``  - the source is blocked, isolated or quarantined, so no traffic from
              it reaches the network and the window is contained by construction.
* ``alert`` - an operator notification that changes no traffic.

Two different things are reported, and only one of them is modelled.

*Volume suppression* is the fraction of the window's packet rate a ``cap``
removes, ``(rate - min(rate, C)) / rate``. Both quantities are measured, so this
is arithmetic on real numbers.

*Risk containment* feeds the residual rate back through the **real** Mamdani
system from ``src/fuzzy_system.py``, together with the window's CV and IAT
deviation, and uses the result to decide whether to escalate. CV and deviation
are held at their observed values because there is no way to predict, without a
live testbed, how a token bucket reshapes packet timing. That makes risk
containment a conservative lower bound: a filter that also regularised the
timing would score better. In particular, rule 11 of the fuzzy base raises risk
on IAT deviation alone, so a window with anomalous timing stays flagged however
far its rate is cut, and the controller escalates.

``is_model=True`` is carried on each result so these outcomes cannot be mistaken
for measurements from a live network.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# action -> (kind, baseline percentile used as the cap)
ACTION_MODELS: Dict[str, tuple] = {
    "RATE_LIMIT_SOURCE": ("cap", "p75"),
    "ENABLE_SYN_COOKIES": ("cap", "p75"),
    "DROP_MALFORMED_FRAGMENTS": ("cap", "p75"),
    "ACTIVATE_WAF_RULE": ("cap", "p75"),
    "LIMIT_CONCURRENT_CONNECTIONS": ("cap", "p50"),
    "THROTTLE_AUTH_ATTEMPTS": ("cap", "p50"),
    "BLOCK_SOURCE": ("stop", None),
    "BLOCK_SCANNING_SOURCE": ("stop", None),
    "UPSTREAM_BLACKHOLE": ("stop", None),
    "ISOLATE_SEGMENT": ("stop", None),
    "ISOLATE_INFECTED_DEVICE": ("stop", None),
    "QUARANTINE_ENDPOINT": ("stop", None),
    "ENFORCE_ARP_DNS_INTEGRITY": ("stop", None),
    "NO_ACTION": ("alert", None),
    "ALERT_ADMIN_AND_LOG": ("alert", None),
}

DISRUPTIVE_KINDS = {"stop"}


@dataclass
class ContainmentStep:
    """One rung of the escalation ladder."""
    action: str
    kind: str
    rate_before: float
    rate_after: float
    risk_after: float
    contained: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "rate_before": round(self.rate_before, 2),
            "rate_after": round(self.rate_after, 2),
            "risk_after": round(self.risk_after, 4),
            "contained": self.contained,
        }


@dataclass
class ContainmentOutcome:
    """Result of walking the ladder for one window."""
    contained: bool
    steps_used: int
    final_action: str
    initial_risk: float
    final_risk: float
    initial_rate: float
    final_rate: float
    needed_disruptive_action: bool
    steps: List[ContainmentStep] = field(default_factory=list)
    is_model: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contained": self.contained,
            "steps_used": self.steps_used,
            "final_action": self.final_action,
            "initial_risk": round(self.initial_risk, 4),
            "final_risk": round(self.final_risk, 4),
            "initial_rate": round(self.initial_rate, 2),
            "final_rate": round(self.final_rate, 2),
            "needed_disruptive_action": self.needed_disruptive_action,
            "steps": [s.to_dict() for s in self.steps],
            "is_model": True,
        }


class EscalationController:
    """Applies a containment ladder and re-scores each rung with Module 1."""

    def __init__(
        self,
        fuzzy_detector: Any,
        baseline_stats: Dict[str, Any],
        risk_threshold: float = 0.35,
        max_steps: int = 3,
    ):
        """
        Args:
            fuzzy_detector: An ``IoTFuzzyAnomalyDetector``; only
                ``evaluate_window(cv, rate, dev)`` is used.
            baseline_stats: Normal baseline, for the rate percentiles used as caps.
            risk_threshold: Fuzzy risk below which a window counts as contained
                (0.35, the same boundary Module 1 uses for "Normal").
            max_steps: Most rungs to try before giving up.
        """
        self.fuzzy = fuzzy_detector
        self.risk_threshold = risk_threshold
        self.max_steps = max_steps
        pct = (baseline_stats or {}).get("rate_percentiles", {})
        self.rate_caps = {
            "p50": float(pct.get("p50", 200.0)),
            "p75": float(pct.get("p75", 500.0)),
            "p95": float(pct.get("p95", 1000.0)),
        }
        self._risk_cache: Dict[tuple, float] = {}

    def _risk(self, cv: float, rate: float, dev: float) -> float:
        """Scores a window with the real fuzzy system, memoised."""
        key = (round(cv, 3), round(rate, 1), round(dev, 3))
        cached = self._risk_cache.get(key)
        if cached is None:
            cached = float(self.fuzzy.evaluate_window(cv, rate, dev))
            self._risk_cache[key] = cached
        return cached

    def apply_action(self, action: str, rate: float) -> tuple:
        """Returns ``(kind, residual_rate)`` for one action under the stated model."""
        kind, cap_key = ACTION_MODELS.get(action, ("alert", None))
        if kind == "stop":
            return kind, 0.0
        if kind == "cap":
            return kind, min(float(rate), self.rate_caps[cap_key])
        return kind, float(rate)

    def contain(
        self,
        cv: float,
        rate: float,
        iat_deviation: float,
        ladder: List[str],
        initial_risk: Optional[float] = None,
    ) -> ContainmentOutcome:
        """
        Walks ``ladder`` until the modelled fuzzy risk falls below the threshold.

        A ``stop`` action removes the source's traffic, so the window is
        contained by construction and its risk is reported as 0. A ``cap`` is
        re-scored through the fuzzy system at the residual rate. An ``alert``
        changes nothing, so it can never contain a window on its own.
        """
        cv = float(cv)
        rate = float(rate)
        iat_deviation = float(iat_deviation)
        risk0 = float(initial_risk) if initial_risk is not None else self._risk(cv, rate, iat_deviation)

        outcome = ContainmentOutcome(
            contained=False,
            steps_used=0,
            final_action="NO_ACTION",
            initial_risk=risk0,
            final_risk=risk0,
            initial_rate=rate,
            final_rate=rate,
            needed_disruptive_action=False,
        )

        if not ladder:
            return outcome

        current_rate = rate
        for action in ladder[: self.max_steps]:
            kind, residual = self.apply_action(action, current_rate)

            if kind == "stop":
                risk_after = 0.0
                contained = True
            elif kind == "cap":
                risk_after = self._risk(cv, residual, iat_deviation)
                contained = risk_after < self.risk_threshold
            else:
                risk_after = self._risk(cv, residual, iat_deviation)
                contained = False

            outcome.steps.append(
                ContainmentStep(
                    action=action,
                    kind=kind,
                    rate_before=current_rate,
                    rate_after=residual,
                    risk_after=risk_after,
                    contained=contained,
                )
            )
            outcome.steps_used += 1
            outcome.final_action = action
            outcome.final_risk = risk_after
            outcome.final_rate = residual
            if kind in DISRUPTIVE_KINDS:
                outcome.needed_disruptive_action = True

            current_rate = residual
            if contained:
                outcome.contained = True
                break

        return outcome
