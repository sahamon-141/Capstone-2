"""
Mitigation Policy Engine for Module 3.

Loads policies from an external JSON configuration file (not hardcoded conditional statements).
Resolves attack categories, validates required input attributes, evaluates confidence
thresholds, and distinguishes between automated simulated actions and recommended manual remediations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from module3_mitigation.models import (
    MitigationPolicy,
    RemediationType,
    AttackEvent,
)


class MitigationPolicyEngine:
    """Policy engine responsible for selecting and validating mitigation policies."""

    def __init__(self, policy_file_path: Optional[Path] = None):
        if policy_file_path is None:
            policy_file_path = Path(__file__).resolve().parent / "policies.json"
        self.policy_file_path = Path(policy_file_path)
        self.policies: Dict[str, MitigationPolicy] = {}
        self.alias_map: Dict[str, str] = {}
        self.fallback_policy: Optional[MitigationPolicy] = None
        self._load_policies()

    def _normalize_name(self, name: str) -> str:
        """Normalizes an attack name for robust key comparison."""
        return (
            name.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("(", "")
            .replace(")", "")
        )

    def _load_policies(self) -> None:
        """Loads and parses policy configuration from JSON file."""
        if not self.policy_file_path.exists():
            raise FileNotFoundError(f"Policy configuration file not found at: {self.policy_file_path}")

        with open(self.policy_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.policies.clear()
        self.alias_map.clear()

        for p_data in data.get("policies", []):
            policy = MitigationPolicy(
                attack_type=p_data["attack_type"],
                mitigation_action=p_data["mitigation_action"],
                description=p_data["description"],
                required_info=p_data.get("required_info", []),
                automated_execution=p_data.get("automated_execution", True),
                requires_admin_approval=p_data.get("requires_admin_approval", False),
                remediation_type=RemediationType(p_data.get("remediation_type", "AUTOMATED_ACTION")),
                confidence_threshold=float(p_data.get("confidence_threshold", 0.60)),
                aliases=p_data.get("aliases", []),
                action_class=p_data.get("action_class", "FILTER"),
                escalation_ladder=p_data.get(
                    "escalation_ladder", [p_data["mitigation_action"]]
                ),
            )
            norm_key = self._normalize_name(policy.attack_type)
            self.policies[norm_key] = policy

            # Register primary key and aliases
            self.alias_map[norm_key] = norm_key
            for alias in policy.aliases:
                self.alias_map[self._normalize_name(alias)] = norm_key

        # Fallback policy
        fb = data.get("fallback_policy", {})
        if fb:
            self.fallback_policy = MitigationPolicy(
                attack_type=fb.get("attack_type", "Unknown"),
                mitigation_action=fb.get("mitigation_action", "ALERT_ADMIN_AND_LOG"),
                description=fb.get("description", "No automated mitigation policy matched."),
                required_info=fb.get("required_info", []),
                automated_execution=fb.get("automated_execution", False),
                requires_admin_approval=fb.get("requires_admin_approval", True),
                remediation_type=RemediationType(fb.get("remediation_type", "RECOMMENDED_MANUAL")),
                confidence_threshold=float(fb.get("confidence_threshold", 0.0)),
                aliases=fb.get("aliases", []),
                action_class=fb.get("action_class", "FILTER"),
                escalation_ladder=fb.get("escalation_ladder", []),
            )

    def select_policy(self, event: AttackEvent) -> Tuple[MitigationPolicy, bool, List[str]]:
        """
        Selects the appropriate mitigation policy for an incoming attack event.

        Returns:
            Tuple of:
                - policy: The matched MitigationPolicy (or fallback policy)
                - is_low_confidence: True if event confidence < policy threshold
                - missing_info: List of required fields not present in event
        """
        norm_type = self._normalize_name(event.attack_type)
        canonical_key = self.alias_map.get(norm_type)

        is_unknown = False
        if canonical_key and canonical_key in self.policies:
            policy = self.policies[canonical_key]
        else:
            policy = self.fallback_policy
            is_unknown = True

        # Check confidence threshold
        is_low_confidence = event.confidence < policy.confidence_threshold

        # Validate required fields
        event_dict = event.to_dict()
        missing_info = [
            req_field
            for req_field in policy.required_info
            if not event_dict.get(req_field)
        ]

        return policy, is_low_confidence, missing_info

    def get_policy_by_name(self, attack_name: str) -> Optional[MitigationPolicy]:
        """Direct lookup by attack category name or alias."""
        canonical_key = self.alias_map.get(self._normalize_name(attack_name))
        if canonical_key and canonical_key in self.policies:
            return self.policies[canonical_key]
        return self.fallback_policy

    def list_all_policies(self) -> List[MitigationPolicy]:
        """Returns all registered mitigation policies."""
        return list(self.policies.values())

    def get_supported_attack_types(self) -> List[str]:
        """Returns list of all supported canonical attack types."""
        return [p.attack_type for p in self.policies.values()]

