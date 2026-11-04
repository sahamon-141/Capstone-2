"""
Mitigation Engine for Module 3: Attack-Specific Mitigation and Verification.

Orchestrates the 4 core responsibilities:
1. Attack Information Receiver (normalizes Module 2 / sensor payloads)
2. Attack-Specific Mitigation Policy Engine (selects policy from policies.json)
3. Simulated Network Controller (safely simulates in-memory changes)
4. Mitigation Verification Engine (evaluates % traffic reduction)

Persists records to SQLite (data/mitigation_history.db) and logs to results/mitigation.log.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from module3_mitigation.models import (
    AttackEvent,
    MitigationPolicy,
    ExecutionResult,
    VerificationResult,
    MitigationRecord,
    ExecutionStatus,
    VerificationStatus,
)
from module3_mitigation.mitigation_policies import MitigationPolicyEngine
from module3_mitigation.simulated_network_controller import (
    BaseNetworkController,
    SimulatedNetworkController,
)
from module3_mitigation.traffic_simulator import TrafficSimulator
from module3_mitigation.mitigation_verifier import MitigationVerifier
from module3_mitigation.database import MitigationDatabase
from module3_mitigation.scenarios import get_scenario


class MitigationEngine:
    """Master orchestrator for Module 3."""

    def __init__(
        self,
        policy_engine: Optional[MitigationPolicyEngine] = None,
        controller: Optional[BaseNetworkController] = None,
        traffic_simulator: Optional[TrafficSimulator] = None,
        verifier: Optional[MitigationVerifier] = None,
        database: Optional[MitigationDatabase] = None,
        log_dir: Optional[Path] = None,
    ):
        self.policy_engine = policy_engine or MitigationPolicyEngine()
        self.controller = controller or SimulatedNetworkController()
        self.traffic_simulator = traffic_simulator or TrafficSimulator()
        self.verifier = verifier or MitigationVerifier()
        self.database = database or MitigationDatabase()

        # Set up logging
        project_root = Path(__file__).resolve().parent.parent
        self.log_dir = log_dir or (project_root / "results")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "mitigation.log"

        self.logger = logging.getLogger("Module3Mitigation")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(str(self.log_file), encoding="utf-8")
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] [Module3] %(message)s")
            fh.setFormatter(fmt)
            self.logger.addHandler(fh)

    # --------------------------------------------------------------------------
    # 1. Attack Information Receiver
    # --------------------------------------------------------------------------
    def parse_attack_payload(self, payload: Dict[str, Any]) -> AttackEvent:
        """
        Normalizes arbitrary prediction payloads from Module 2 or simulation into AttackEvent.
        """
        attack_type = (
            payload.get("attack_type")
            or payload.get("predicted_attack")
            or payload.get("raw_class")
            or "Unknown"
        )
        confidence = float(payload.get("confidence", 0.0))
        source_mac = payload.get("source_mac")
        source_ip = payload.get("source_ip")
        dest_ip = payload.get("destination_ip") or payload.get("dest_ip")
        timestamp = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()
        sensor_id = payload.get("sensor_id") or "ESP32_01"
        traffic_stats = payload.get("traffic_stats", {})

        return AttackEvent(
            attack_type=attack_type,
            confidence=confidence,
            source_mac=source_mac,
            source_ip=source_ip,
            destination_ip=dest_ip,
            timestamp=timestamp,
            sensor_id=sensor_id,
            traffic_stats=traffic_stats,
            raw_class=payload.get("raw_class"),
        )

    # --------------------------------------------------------------------------
    # 2. Complete Workflow Execution
    # --------------------------------------------------------------------------
    def process_attack(
        self,
        event: AttackEvent,
        traffic_before: Optional[float] = None,
        traffic_after: Optional[float] = None,
        force_status: Optional[ExecutionStatus] = None,
        outcome_mode: Optional[str] = None,
    ) -> MitigationRecord:
        """
        Executes the end-to-end Module 3 workflow:
        1. Select policy
        2. Simulate mitigation
        3. Obtain traffic before & after
        4. Verify effectiveness
        5. Persist to SQLite & log
        """
        self.logger.info(
            f"Received attack event: type={event.attack_type}, conf={event.confidence:.3f}, "
            f"MAC={event.source_mac}, IP={event.source_ip}"
        )

        # 1. Policy selection
        policy, is_low_conf, missing_info = self.policy_engine.select_policy(event)

        # If confidence is lower than threshold, require administrator approval
        effective_policy = policy
        if is_low_conf and not policy.requires_admin_approval:
            # Create a policy clone that requires approval
            effective_policy = MitigationPolicy(
                attack_type=policy.attack_type,
                mitigation_action=policy.mitigation_action,
                description=f"{policy.description} [LOW CONFIDENCE ({event.confidence:.2f} < {policy.confidence_threshold:.2f}): Confirmation Required]",
                required_info=policy.required_info,
                automated_execution=False,
                requires_admin_approval=True,
                remediation_type=policy.remediation_type,
                confidence_threshold=policy.confidence_threshold,
                aliases=policy.aliases,
            )

        # 2. Simulated Network Controller Execution
        exec_res = self.controller.execute_mitigation(
            policy=effective_policy, event=event, force_status=force_status
        )

        # 3. Generate or use traffic measurements
        if traffic_before is not None and traffic_after is not None:
            # Use explicit measurements
            t_before = float(traffic_before)
            t_after = float(traffic_after)
            legit_impact = 0.0
            obs_window = 60
        else:
            # Generate simulated telemetry
            # If execution required admin approval or unsupported, after traffic is not reduced unless forced
            active_mode = outcome_mode
            if exec_res.status in (
                ExecutionStatus.REQUIRES_ADMIN_APPROVAL,
                ExecutionStatus.UNSUPPORTED_ACTION,
            ) and outcome_mode is None:
                # Without admin action, traffic remains at before level
                active_mode = None
                measurements = self.traffic_simulator.generate_measurements(
                    attack_type=effective_policy.attack_type,
                    traffic_before=traffic_before,
                    traffic_after=traffic_before or 300.0,
                )
            else:
                measurements = self.traffic_simulator.generate_measurements(
                    attack_type=effective_policy.attack_type,
                    traffic_before=traffic_before,
                    traffic_after=traffic_after,
                    outcome_mode=active_mode,
                )
            t_before = measurements["traffic_before"]
            t_after = measurements["traffic_after"]
            legit_impact = measurements["legitimate_traffic_impact"]
            obs_window = measurements["observation_window_seconds"]

        # 4. Mitigation Verification Engine
        verif_res = self.verifier.verify(
            traffic_before=t_before,
            traffic_after=t_after,
            legitimate_traffic_impact=legit_impact,
            observation_window_seconds=obs_window,
        )

        # 5. Build Consolidated Mitigation Record
        record = MitigationRecord(
            timestamp=event.timestamp,
            sensor_id=event.sensor_id,
            attack_type=event.attack_type,
            confidence=round(event.confidence, 4),
            source_mac=event.source_mac,
            source_ip=event.source_ip,
            destination_ip=event.destination_ip,
            mitigation_action=exec_res.action,
            action_description=exec_res.message,
            execution_status=exec_res.status,
            traffic_before=verif_res.traffic_before,
            traffic_after=verif_res.traffic_after,
            reduction_percentage=verif_res.reduction_percentage,
            verification_status=verif_res.status,
            simulation_mode=True,
            notes=verif_res.message,
        )

        # 6. Database Persistence
        try:
            rec_id = self.database.insert_record(record)
            record.id = rec_id
        except Exception as e:
            self.logger.error(f"Database insertion error: {e}")

        # 7. Logging
        log_level = (
            logging.WARNING
            if record.verification_status == VerificationStatus.MITIGATION_FAILED
            or record.execution_status == ExecutionStatus.SIMULATION_FAILED
            else logging.INFO
        )
        self.logger.log(
            log_level,
            f"Mitigation completed [ID={record.id}]: action={record.mitigation_action}, "
            f"exec_status={record.execution_status.value}, "
            f"traffic: {record.traffic_before:.1f} -> {record.traffic_after:.1f} "
            f"({record.reduction_percentage:+.1f}%), verif_status={record.verification_status.value}"
        )

        return record

    def run_scenario(self, scenario_id: int) -> MitigationRecord:
        """Runs one of the 8 predefined reproducible simulation scenarios."""
        sc = get_scenario(scenario_id)
        event = AttackEvent(
            attack_type=sc["attack_type"],
            confidence=sc["confidence"],
            source_mac=sc.get("source_mac"),
            source_ip=sc.get("source_ip"),
            destination_ip=sc.get("destination_ip"),
            sensor_id=sc.get("sensor_id", "ESP32_01"),
        )
        return self.process_attack(
            event=event,
            traffic_before=sc["traffic_before"],
            traffic_after=sc["traffic_after"],
            force_status=sc.get("force_status"),
        )

    def mitigate_from_pipeline(
        self,
        pipeline_output: Dict[str, Any],
        source_mac: Optional[str] = None,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        sensor_id: str = "ESP32_01",
    ) -> Optional[MitigationRecord]:
        """
        Direct handoff from PipelineIntegrator / Module 2 output dictionary.
        If pipeline output indicated normal traffic, returns None (no mitigation required).
        """
        # If normal, no mitigation needed
        if not pipeline_output.get("module2_invoked", True):
            return None

        attack_type = pipeline_output.get("predicted_attack") or "Unknown"
        confidence = float(pipeline_output.get("confidence", 0.0))

        event = AttackEvent(
            attack_type=attack_type,
            confidence=confidence,
            source_mac=source_mac or "AA:BB:CC:DD:EE:FF",
            source_ip=source_ip,
            destination_ip=destination_ip,
            sensor_id=sensor_id,
            raw_class=pipeline_output.get("raw_class"),
        )
        return self.process_attack(event)
