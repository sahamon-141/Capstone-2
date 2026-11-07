"""
Behaviour-based containment for unknown / zero-day attacks.

A supervised classifier cannot name an attack it has never been trained on, so
no attack-specific policy applies to a zero-day window. Module 1 has already
measured the *shape* of that window, however: the coefficient of variation of
the inter-arrival times (CV), the mean packet rate and the relative IAT
deviation from the normal baseline. Those three quantities are enough to pick a
generic containment action without knowing the attack name:

    CV low,  rate high    -> constant-rate flood     -> cap the source rate
    CV high, rate low     -> slow probe or scan      -> block the source
    rate high, D high     -> volumetric surge        -> blackhole upstream
    anything else flagged -> unclassified anomaly    -> isolate the segment

The CV and deviation breakpoints below are deliberately the same ones the
Mamdani membership functions use in ``src/fuzzy_system.py`` (CV low ends at
0.65, CV high starts at 1.40, IAT deviation high starts at 0.50), so the
behaviour classes line up with the rules that flagged the window in the first
place. The two rate breakpoints come from the normal-traffic baseline rather
than constants, so they adapt to the deployment:

* *elevated* - above the median normal rate. Used together with a low CV,
  because near-constant packet timing is itself the flood signature; on
  CICIoT2023 the held-out Mirai windows sit at CV = 0.0 but only about three
  times the median benign rate, so requiring an extreme rate would miss them.
* *high* - above the 95th percentile of the normal rate, which benign traffic
  reaches only rarely. Used on its own to call a volumetric surge.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ZeroDayBehaviour(str, Enum):
    """Traffic shapes that can be recognised without knowing the attack class."""
    CONSTANT_RATE_FLOOD = "CONSTANT_RATE_FLOOD"
    SLOW_PROBE = "SLOW_PROBE"
    VOLUMETRIC_SURGE = "VOLUMETRIC_SURGE"
    UNCLASSIFIED_ANOMALY = "UNCLASSIFIED_ANOMALY"


# Containment ladders, least disruptive action first.
BEHAVIOUR_LADDERS: Dict[ZeroDayBehaviour, List[str]] = {
    ZeroDayBehaviour.CONSTANT_RATE_FLOOD: [
        "RATE_LIMIT_SOURCE", "BLOCK_SOURCE", "UPSTREAM_BLACKHOLE",
    ],
    ZeroDayBehaviour.VOLUMETRIC_SURGE: [
        "RATE_LIMIT_SOURCE", "UPSTREAM_BLACKHOLE", "ISOLATE_SEGMENT",
    ],
    ZeroDayBehaviour.SLOW_PROBE: [
        "BLOCK_SCANNING_SOURCE", "ISOLATE_SEGMENT",
    ],
    ZeroDayBehaviour.UNCLASSIFIED_ANOMALY: [
        "RATE_LIMIT_SOURCE", "BLOCK_SOURCE", "ISOLATE_SEGMENT",
    ],
}

BEHAVIOUR_REASONS: Dict[ZeroDayBehaviour, str] = {
    ZeroDayBehaviour.CONSTANT_RATE_FLOOD:
        "Packet arrivals are far more regular than normal traffic while the rate "
        "is above baseline, which is how automated flooding tools emit packets.",
    ZeroDayBehaviour.VOLUMETRIC_SURGE:
        "The packet rate is above baseline and the mean inter-arrival time has "
        "shifted sharply away from the normal profile.",
    ZeroDayBehaviour.SLOW_PROBE:
        "Arrivals are highly irregular at a low packet rate, which matches "
        "sporadic probing rather than a flood.",
    ZeroDayBehaviour.UNCLASSIFIED_ANOMALY:
        "The window was flagged as anomalous but its timing and rate do not "
        "match any recognised attack shape.",
}


@dataclass
class ZeroDayDecision:
    """Containment choice made for a window with no matching class policy."""
    behaviour: ZeroDayBehaviour
    escalation_ladder: List[str]
    reason: str
    cv: float
    rate: float
    iat_deviation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behaviour": self.behaviour.value,
            "escalation_ladder": list(self.escalation_ladder),
            "reason": self.reason,
            "cv": round(self.cv, 4),
            "mean_rate": round(self.rate, 2),
            "iat_deviation": round(self.iat_deviation, 4),
        }


class ZeroDayContainment:
    """Chooses a containment ladder from observed traffic behaviour."""

    # Breakpoints shared with the Mamdani membership functions.
    CV_LOW_MAX = 0.65
    CV_HIGH_MIN = 1.40
    DEV_HIGH_MIN = 0.50

    def __init__(
        self,
        baseline_stats: Optional[Dict[str, Any]] = None,
        rate_high_threshold: Optional[float] = None,
        rate_elevated_threshold: Optional[float] = None,
    ):
        """
        Args:
            baseline_stats: Normal baseline from ``NormalBaselineBuilder.fit``.
                Supplies the median and 95th-percentile normal rates.
            rate_high_threshold: Override for "rate is high", packets per second.
            rate_elevated_threshold: Override for "rate is above normal".
        """
        pct = (baseline_stats or {}).get("rate_percentiles", {})
        self.rate_high_threshold = float(
            rate_high_threshold if rate_high_threshold is not None
            else pct.get("p95", 1000.0)
        )
        self.rate_elevated_threshold = float(
            rate_elevated_threshold if rate_elevated_threshold is not None
            else pct.get("p50", 200.0)
        )

    def classify(self, cv: float, rate: float, iat_deviation: float) -> ZeroDayBehaviour:
        """Maps the three Module 1 window features onto a behaviour class."""
        rate_is_high = rate > self.rate_high_threshold
        rate_is_elevated = rate > self.rate_elevated_threshold
        cv_is_low = cv < self.CV_LOW_MAX
        cv_is_high = cv > self.CV_HIGH_MIN
        dev_is_high = iat_deviation > self.DEV_HIGH_MIN

        # Near-constant packet timing above the normal median rate is the
        # signature of an automated flooding tool, whatever its name.
        if cv_is_low and rate_is_elevated:
            return ZeroDayBehaviour.CONSTANT_RATE_FLOOD
        if rate_is_high and dev_is_high:
            return ZeroDayBehaviour.VOLUMETRIC_SURGE
        if cv_is_high and not rate_is_high:
            return ZeroDayBehaviour.SLOW_PROBE
        if rate_is_high:
            return ZeroDayBehaviour.VOLUMETRIC_SURGE
        return ZeroDayBehaviour.UNCLASSIFIED_ANOMALY

    def decide(self, cv: float, rate: float, iat_deviation: float) -> ZeroDayDecision:
        """Returns the behaviour class and the containment ladder to walk."""
        behaviour = self.classify(cv, rate, iat_deviation)
        return ZeroDayDecision(
            behaviour=behaviour,
            escalation_ladder=list(BEHAVIOUR_LADDERS[behaviour]),
            reason=BEHAVIOUR_REASONS[behaviour],
            cv=float(cv),
            rate=float(rate),
            iat_deviation=float(iat_deviation),
        )
