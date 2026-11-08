"""
Data models, enumerations, and type definitions for Module 3:
Attack-Specific Mitigation and Verification.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional


class ExecutionStatus(str, Enum):
    """Execution status returned by the network controller."""
    SIMULATED_SUCCESS = "SIMULATED_SUCCESS"
    REQUIRES_ADMIN_APPROVAL = "REQUIRES_ADMIN_APPROVAL"
    SIMULATION_FAILED = "SIMULATION_FAILED"
    UNSUPPORTED_ACTION = "UNSUPPORTED_ACTION"


class VerificationStatus(str, Enum):
    """Mitigation verification status."""
    MITIGATION_EFFECTIVE = "MITIGATION_EFFECTIVE"
    PARTIALLY_EFFECTIVE = "PARTIALLY_EFFECTIVE"
    MITIGATION_FAILED = "MITIGATION_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    VERIFICATION_PENDING = "VERIFICATION_PENDING"


class RemediationType(str, Enum):
    """Distinguishes simulated network action from manual remediation."""
    AUTOMATED_ACTION = "AUTOMATED_ACTION"
    RECOMMENDED_MANUAL = "RECOMMENDED_MANUAL"


@dataclass
class AttackEvent:
    """Incoming attack information received from Module 2 or simulation."""
    attack_type: str
    confidence: float
    source_mac: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sensor_id: str = "ESP32_01"
    traffic_stats: Dict[str, Any] = field(default_factory=dict)
    raw_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MitigationPolicy:
    """Policy definition for handling a specific attack category."""
    attack_type: str
    mitigation_action: str
    description: str
    required_info: List[str] = field(default_factory=list)
    automated_execution: bool = True
    requires_admin_approval: bool = False
    remediation_type: RemediationType = RemediationType.AUTOMATED_ACTION
    confidence_threshold: float = 0.60
    aliases: List[str] = field(default_factory=list)
    # How the action changes traffic: FILTER caps the source rate, BLOCK stops
    # the source outright, ALERT leaves traffic untouched.
    action_class: str = "FILTER"
    # Actions to try in order, least disruptive first.
    escalation_ladder: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["remediation_type"] = self.remediation_type.value
        return data


@dataclass
class ExecutionResult:
    """Result of attempting to apply a mitigation action."""
    action: str
    status: ExecutionStatus
    message: str
    target: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_simulation: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class VerificationResult:
    """Evaluation of post-mitigation traffic reduction and effectiveness."""
    traffic_before: float
    traffic_after: float
    reduction_percentage: float
    status: VerificationStatus
    observation_window_seconds: int = 60
    legitimate_traffic_impact: float = 0.0  # Estimated % disruption to benign traffic
    message: str = ""
    is_simulation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass
class MitigationRecord:
    """Consolidated record combining attack, mitigation, verification, and persistence."""
    id: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sensor_id: str = "ESP32_01"
    attack_type: str = "Unknown"
    confidence: float = 0.0
    source_mac: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    mitigation_action: str = "NO_ACTION"
    action_description: str = ""
    execution_status: ExecutionStatus = ExecutionStatus.UNSUPPORTED_ACTION
    traffic_before: float = 0.0
    traffic_after: float = 0.0
    reduction_percentage: float = 0.0
    verification_status: VerificationStatus = VerificationStatus.INSUFFICIENT_DATA
    simulation_mode: bool = True
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["execution_status"] = (
            self.execution_status.value
            if isinstance(self.execution_status, ExecutionStatus)
            else str(self.execution_status)
        )
        data["verification_status"] = (
            self.verification_status.value
            if isinstance(self.verification_status, VerificationStatus)
            else str(self.verification_status)
        )
        return data

