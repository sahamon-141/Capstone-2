"""
Mitigation Verification Engine for Module 3.

Evaluates mitigation effectiveness by comparing pre- and post-mitigation traffic measurements:
    Reduction (%) = ((Before - After) / Before) * 100

Assigns one of five exact statuses:
- MITIGATION_EFFECTIVE (Reduction >= 80%)
- PARTIALLY_EFFECTIVE (30% <= Reduction < 80%)
- MITIGATION_FAILED (Reduction < 30% or traffic increased)
- INSUFFICIENT_DATA (Zero baseline traffic or missing measurements)
- VERIFICATION_PENDING (Awaiting observation window)

Handles zero-baseline, missing measurements, and legitimate traffic collateral impact.
All evaluations are explicitly marked as [SIMULATED].
"""

from typing import Optional

from module3_mitigation.models import VerificationResult, VerificationStatus


class MitigationVerifier:
    """Mathematical verification engine assessing mitigation effectiveness."""

    def __init__(
        self,
        effective_threshold_pct: float = 80.0,
        partial_threshold_pct: float = 30.0,
        default_window_seconds: int = 60,
    ):
        self.effective_threshold_pct = effective_threshold_pct
        self.partial_threshold_pct = partial_threshold_pct
        self.default_window_seconds = default_window_seconds

    def verify(
        self,
        traffic_before: Optional[float],
        traffic_after: Optional[float],
        legitimate_traffic_impact: float = 0.0,
        observation_window_seconds: Optional[int] = None,
        is_pending: bool = False,
    ) -> VerificationResult:
        """
        Calculates percentage traffic reduction and determines verification status.

        Args:
            traffic_before: Suspicious traffic before mitigation (pkts/min)
            traffic_after: Suspicious traffic after mitigation (pkts/min)
            legitimate_traffic_impact: Estimated impact % on benign flows
            observation_window_seconds: Duration of equal observation windows
            is_pending: If True, returns VERIFICATION_PENDING
        """
        window_sec = observation_window_seconds or self.default_window_seconds

        # 1. Verification Pending
        if is_pending:
            return VerificationResult(
                traffic_before=traffic_before or 0.0,
                traffic_after=0.0,
                reduction_percentage=0.0,
                status=VerificationStatus.VERIFICATION_PENDING,
                observation_window_seconds=window_sec,
                legitimate_traffic_impact=legitimate_traffic_impact,
                message="[SIMULATED] Observation window in progress. Awaiting post-mitigation stream.",
                is_simulation=True,
            )

        # 2. Handle Missing Measurements
        if traffic_before is None or traffic_after is None:
            return VerificationResult(
                traffic_before=traffic_before or 0.0,
                traffic_after=traffic_after or 0.0,
                reduction_percentage=0.0,
                status=VerificationStatus.INSUFFICIENT_DATA,
                observation_window_seconds=window_sec,
                legitimate_traffic_impact=legitimate_traffic_impact,
                message="[SIMULATED] Incomplete traffic telemetry. Before or after measurement is missing.",
                is_simulation=True,
            )

        # 3. Handle Zero Baseline Traffic
        if traffic_before <= 0.0:
            return VerificationResult(
                traffic_before=0.0,
                traffic_after=traffic_after,
                reduction_percentage=0.0,
                status=VerificationStatus.INSUFFICIENT_DATA,
                observation_window_seconds=window_sec,
                legitimate_traffic_impact=legitimate_traffic_impact,
                message=(
                    "[SIMULATED] Baseline suspicious traffic is zero. Cannot compute percentage reduction."
                ),
                is_simulation=True,
            )

        # 4. Compute Percentage Reduction
        # Reduction (%) = ((Before - After) / Before) * 100
        reduction_pct = ((traffic_before - traffic_after) / traffic_before) * 100.0
        reduction_pct = round(reduction_pct, 2)

        # 5. Determine Effectiveness Status
        if reduction_pct >= self.effective_threshold_pct:
            status = VerificationStatus.MITIGATION_EFFECTIVE
            msg = (
                f"[SIMULATED] Mitigation effective! Suspicious traffic dropped by {reduction_pct:.1f}% "
                f"(from {traffic_before:.1f} to {traffic_after:.1f} pkts/min). "
                f"Legitimate traffic impact: {legitimate_traffic_impact:.1f}%."
            )
        elif reduction_pct >= self.partial_threshold_pct:
            status = VerificationStatus.PARTIALLY_EFFECTIVE
            msg = (
                f"[SIMULATED] Mitigation partially effective. Suspicious traffic reduced by {reduction_pct:.1f}% "
                f"(from {traffic_before:.1f} to {traffic_after:.1f} pkts/min). "
                f"Further throttling recommended."
            )
        else:
            status = VerificationStatus.MITIGATION_FAILED
            msg = (
                f"[SIMULATED] Mitigation failed to suppress attack. Traffic reduction was only {reduction_pct:.1f}% "
                f"(from {traffic_before:.1f} to {traffic_after:.1f} pkts/min)."
            )

        return VerificationResult(
            traffic_before=round(traffic_before, 2),
            traffic_after=round(traffic_after, 2),
            reduction_percentage=reduction_pct,
            status=status,
            observation_window_seconds=window_sec,
            legitimate_traffic_impact=round(legitimate_traffic_impact, 2),
            message=msg,
            is_simulation=True,
        )

