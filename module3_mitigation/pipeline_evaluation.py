"""
Runs Module 3 over every window Module 1 produced and reports what it did.

This is the batch counterpart to the dashboard: instead of a handful of
hand-picked scenarios, it walks the full evaluation stream, selects a mitigation
policy for each flagged window, escalates until the modelled fuzzy risk returns
below the normal threshold, and writes the aggregate to
``results/metrics/module3_metrics.json``.

Two things are worth being explicit about, because they decide what may be
claimed from the output.

1. **Which label drives the policy.** By default the evaluation is driven by the
   window's ground-truth class (``label_source="ground_truth"``), so that
   Module 3's policy coverage and containment are measured on their own rather
   than confounded with Module 2's classification error. Pass a prediction
   column to drive it from Module 2 instead. Held-out classes are never looked
   up in the policy table: they take the zero-day route, which is the point of
   holding them out.

2. **What "contained" means.** Nothing is enforced on a real network, so the
   effect of an action is modelled by ``EscalationController`` under a stated
   filter model, and the residual is re-scored by the real Mamdani system. A
   window counts as contained when that modelled risk falls below 0.35. These
   are modelled outcomes, not measurements from a live testbed.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from module3_mitigation.escalation_controller import EscalationController
from module3_mitigation.mitigation_policies import MitigationPolicyEngine
from module3_mitigation.zero_day_containment import ZeroDayContainment

NORMAL_LABEL = "BenignTraffic"


class Module3PipelineEvaluator:
    """Applies the mitigation stage to every window and aggregates the result."""

    def __init__(
        self,
        fuzzy_detector: Any,
        baseline_stats: Dict[str, Any],
        held_out_classes: Sequence[str] = (),
        policy_engine: Optional[MitigationPolicyEngine] = None,
        risk_threshold: float = 0.35,
        max_steps: int = 3,
    ):
        self.policy_engine = policy_engine or MitigationPolicyEngine()
        self.zero_day = ZeroDayContainment(baseline_stats=baseline_stats)
        self.escalator = EscalationController(
            fuzzy_detector=fuzzy_detector,
            baseline_stats=baseline_stats,
            risk_threshold=risk_threshold,
            max_steps=max_steps,
        )
        self.held_out = {h.strip().lower() for h in held_out_classes}
        self.risk_threshold = risk_threshold

    # ----------------------------------------------------------------------
    def _route(self, label: str, cv: float, rate: float, dev: float) -> Dict[str, Any]:
        """Picks the policy family and escalation ladder for one window."""
        is_held_out = str(label).strip().lower() in self.held_out

        if not is_held_out:
            policy = self.policy_engine.get_policy_by_name(str(label))
            matched = policy is not None and policy.attack_type != "Unknown"
            if matched:
                return {
                    "family": policy.attack_type,
                    "ladder": list(policy.escalation_ladder) or [policy.mitigation_action],
                    "route": "class_policy",
                    "behaviour": None,
                    "requires_admin_approval": policy.requires_admin_approval,
                }

        # Held out, or no class policy matched: choose from observed behaviour.
        decision = self.zero_day.decide(cv, rate, dev)
        return {
            "family": "Unknown (zero-day route)",
            "ladder": decision.escalation_ladder,
            "route": "zero_day_behaviour",
            "behaviour": decision.behaviour.value,
            "requires_admin_approval": False,
        }

    # ----------------------------------------------------------------------
    def evaluate(
        self,
        windows_df: pd.DataFrame,
        label_column: str = "original_label",
        flag_column: str = "fuzzy_pred",
        risk_column: str = "attack_risk",
    ) -> Dict[str, Any]:
        """Walks every window and returns per-window rows plus the aggregate."""
        rows: List[Dict[str, Any]] = []

        for r in windows_df.itertuples(index=False):
            label = str(getattr(r, label_column))
            flagged = int(getattr(r, flag_column)) == 1
            cv = float(r.cv_iat)
            rate = float(r.mean_rate)
            dev = float(r.iat_deviation)
            risk = float(getattr(r, risk_column))
            truly_benign = label == NORMAL_LABEL

            if not flagged:
                rows.append({
                    "window_id": int(r.window_id),
                    "true_label": label,
                    "truly_benign": truly_benign,
                    "flagged": False,
                    "route": "bypassed",
                    "family": "None",
                    "behaviour": None,
                    "first_action": "NO_ACTION",
                    "final_action": "NO_ACTION",
                    "steps_used": 0,
                    "initial_risk": round(risk, 4),
                    "final_risk": round(risk, 4),
                    "contained": None,
                    "needed_disruptive_action": False,
                    "would_block_if_misclassified": False,
                })
                continue

            routing = self._route(label, cv, rate, dev)
            outcome = self.escalator.contain(
                cv=cv, rate=rate, iat_deviation=dev,
                ladder=routing["ladder"], initial_risk=risk,
            )

            # First rung: how much packet volume the least disruptive action
            # removes. Both rates are measured, so this is not modelled.
            first = outcome.steps[0] if outcome.steps else None
            if first is not None and first.rate_before > 0:
                suppressed = 100.0 * (first.rate_before - first.rate_after) / first.rate_before
            else:
                suppressed = 0.0

            # Worst-case collateral. Driving Module 3 from the true label means a
            # benign window that Module 1 flagged is correctly left alone, which
            # assumes Module 2 never mislabels a false positive as an attack. For
            # the other extreme, work out what this window would have received had
            # it been routed as an attack with no class policy to match.
            would_block = False
            if truly_benign:
                alt = self.zero_day.decide(cv, rate, dev).escalation_ladder
                if alt:
                    kind, _ = self.escalator.apply_action(alt[0], rate)
                    would_block = kind == "stop"

            rows.append({
                "window_id": int(r.window_id),
                "true_label": label,
                "truly_benign": truly_benign,
                "flagged": True,
                "route": routing["route"],
                "family": routing["family"],
                "behaviour": routing["behaviour"],
                "first_action": routing["ladder"][0] if routing["ladder"] else "NO_ACTION",
                "first_action_kind": first.kind if first else "none",
                "final_action": outcome.final_action,
                "steps_used": outcome.steps_used,
                "observed_rate": round(rate, 2),
                "rate_after_first_action": round(first.rate_after, 2) if first else round(rate, 2),
                "volume_suppressed_pct": round(suppressed, 2),
                "initial_risk": round(outcome.initial_risk, 4),
                "final_risk": round(outcome.final_risk, 4),
                "risk_after_first_action": round(first.risk_after, 4) if first else round(risk, 4),
                "contained_by_first_action": bool(first.contained) if first else False,
                "contained": outcome.contained,
                "needed_disruptive_action": outcome.needed_disruptive_action,
                "would_block_if_misclassified": would_block,
            })

        df = pd.DataFrame(rows)
        return {"per_window": df, "summary": self._summarise(df)}

    # ----------------------------------------------------------------------
    def _summarise(self, df: pd.DataFrame) -> Dict[str, Any]:
        total = len(df)
        flagged = df[df["flagged"]]
        n_flagged = len(flagged)

        def pct(n: int, d: int) -> float:
            return round(100.0 * n / d, 2) if d else 0.0

        # Policy coverage: flagged windows that matched a class-specific policy.
        by_class = flagged[flagged["route"] == "class_policy"]
        by_behaviour = flagged[flagged["route"] == "zero_day_behaviour"]

        # Containment, excluding windows whose correct outcome is "do nothing".
        actionable = flagged[flagged["first_action"] != "NO_ACTION"]
        contained = actionable[actionable["contained"]]
        by_filter = actionable[actionable["contained_by_first_action"]]
        disruptive = actionable[actionable["needed_disruptive_action"]]

        steps = Counter(int(s) for s in contained["steps_used"])

        # Volume removed by the least disruptive action. Measured, not modelled.
        filtered = actionable[actionable["first_action_kind"] == "cap"]
        if len(filtered):
            suppression = {
                "windows": len(filtered),
                "mean_pct": round(float(filtered["volume_suppressed_pct"].mean()), 2),
                "median_pct": round(float(filtered["volume_suppressed_pct"].median()), 2),
                "mean_rate_before": round(float(filtered["observed_rate"].mean()), 2),
                "mean_rate_after": round(float(filtered["rate_after_first_action"].mean()), 2),
            }
        else:
            suppression = {"windows": 0}

        # Collateral: flagged windows that are actually benign and still got an
        # action. These are Module 1 false positives reaching the mitigation stage.
        fp = flagged[flagged["truly_benign"]]
        fp_acted = fp[fp["first_action"] != "NO_ACTION"]
        fp_blocked = fp_acted[fp_acted["needed_disruptive_action"]]

        # Zero-day subset: held-out traffic that reached the behaviour route.
        zd = by_behaviour[~by_behaviour["truly_benign"]]
        zd_contained = zd[zd["contained"]]

        risk_drop = None
        if len(actionable):
            risk_drop = round(
                float(actionable["initial_risk"].mean() - actionable["final_risk"].mean()), 4
            )

        return {
            "note": (
                "Nothing here is enforced on a live network. 'volume_suppression' "
                "is arithmetic on measured packet rates. 'containment' is modelled: "
                "the residual rate is re-scored by the Mamdani system of Module 1 "
                f"and a window counts as contained when that risk falls below "
                f"{self.risk_threshold}. A blocking action contains by construction, "
                "so 'contained_by_filter_alone' is the informative figure and "
                "'contained_overall' is not an accuracy."
            ),
            "total_windows": total,
            "windows_bypassed_by_module1": total - n_flagged,
            "windows_sent_to_mitigation": n_flagged,
            "windows_sent_to_mitigation_pct": pct(n_flagged, total),
            "policy_coverage": {
                "matched_class_policy": len(by_class),
                "matched_class_policy_pct": pct(len(by_class), n_flagged),
                "zero_day_behaviour_route": len(by_behaviour),
                "zero_day_behaviour_route_pct": pct(len(by_behaviour), n_flagged),
            },
            "action_distribution": {
                str(k): int(v) for k, v in
                Counter(flagged["first_action"]).most_common()
            },
            "family_distribution": {
                str(k): int(v) for k, v in
                Counter(flagged["family"]).most_common()
            },
            "volume_suppression_by_first_action": suppression,
            "containment": {
                "actionable_windows": len(actionable),
                "contained_by_filter_alone": len(by_filter),
                "contained_by_filter_alone_pct": pct(len(by_filter), len(actionable)),
                "needed_disruptive_action": len(disruptive),
                "needed_disruptive_action_pct": pct(len(disruptive), len(actionable)),
                "contained_overall": len(contained),
                "contained_at_step": {str(k): int(steps[k]) for k in sorted(steps)},
                "mean_risk_drop": risk_drop,
            },
            "collateral_on_benign_windows": {
                "explanation": (
                    "Module 1 false positives that reach the mitigation stage. The "
                    "best case assumes Module 2 labels them benign, so Module 3 "
                    "leaves them alone. The worst case assumes Module 2 calls them "
                    "an attack, so they are routed by behaviour; 'would_block' "
                    "counts those that would then lose all traffic."
                ),
                "benign_windows_flagged_by_module1": len(fp),
                "best_case_receiving_an_action": len(fp_acted),
                "best_case_receiving_a_blocking_action": len(fp_blocked),
                "worst_case_receiving_a_blocking_action": int(fp["would_block_if_misclassified"].sum()),
                "worst_case_blocked_pct_of_flagged_benign": pct(
                    int(fp["would_block_if_misclassified"].sum()), len(fp)
                ),
            },
            "families_exercised_by_this_stream": sorted(
                {str(f) for f in flagged["family"].unique()}
            ),
            "families_defined_in_policy_file": sorted(
                {p.attack_type for p in self.policy_engine.list_all_policies()}
            ),
            "zero_day": {
                "held_out_classes": sorted(self.held_out),
                "held_out_windows_reaching_mitigation": len(zd),
                "contained_overall": len(zd_contained),
                "contained_overall_pct": pct(len(zd_contained), len(zd)),
                "mean_volume_suppressed_pct": (
                    round(float(zd[zd["first_action_kind"] == "cap"]["volume_suppressed_pct"].mean()), 2)
                    if len(zd[zd["first_action_kind"] == "cap"]) else 0.0
                ),
                "behaviour_distribution": {
                    str(k): int(v) for k, v in
                    Counter(zd["behaviour"].dropna()).most_common()
                },
            },
        }


def run_module3_evaluation(
    windows_df: pd.DataFrame,
    fuzzy_detector: Any,
    baseline_stats: Dict[str, Any],
    held_out_classes: Sequence[str],
    metrics_dir: Path,
    risk_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Convenience wrapper that evaluates and writes both output files."""
    evaluator = Module3PipelineEvaluator(
        fuzzy_detector=fuzzy_detector,
        baseline_stats=baseline_stats,
        held_out_classes=held_out_classes,
        risk_threshold=risk_threshold,
    )
    result = evaluator.evaluate(windows_df)
    summary = result["summary"]
    summary["label_source"] = (
        "ground_truth: Module 3 is driven by the window's true class so that its "
        "coverage and containment are measured independently of Module 2's error."
    )

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "module3_metrics.json").write_text(
        json.dumps(summary, indent=4), encoding="utf-8"
    )
    result["per_window"].to_csv(metrics_dir / "module3_per_window.csv", index=False)
    return summary
