"""
SQLite persistence layer for Module 3 mitigation history.

Stores all mitigation attempts, outcomes, and traffic verifications in:
    data/mitigation_history.db
Records persist across application restarts.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from datetime import datetime, timezone

from module3_mitigation.models import MitigationRecord, ExecutionStatus, VerificationStatus


class MitigationDatabase:
    """Manages SQLite storage and retrieval for mitigation event records."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            # Default to data/mitigation_history.db at project root
            project_root = Path(__file__).resolve().parent.parent
            db_path = project_root / "data" / "mitigation_history.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that guarantees the SQLite connection is cleanly closed."""
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes the database schema if not already present."""
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mitigation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    sensor_id TEXT NOT NULL,
                    attack_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source_mac TEXT,
                    source_ip TEXT,
                    destination_ip TEXT,
                    mitigation_action TEXT NOT NULL,
                    action_description TEXT,
                    execution_status TEXT NOT NULL,
                    traffic_before REAL NOT NULL,
                    traffic_after REAL NOT NULL,
                    reduction_percentage REAL NOT NULL,
                    verification_status TEXT NOT NULL,
                    simulation_mode INTEGER NOT NULL DEFAULT 1,
                    notes TEXT
                )
            """)
            conn.commit()

    def insert_record(self, record: MitigationRecord) -> int:
        """Inserts a new mitigation attempt into the database."""
        exec_status_str = (
            record.execution_status.value
            if isinstance(record.execution_status, ExecutionStatus)
            else str(record.execution_status)
        )
        verif_status_str = (
            record.verification_status.value
            if isinstance(record.verification_status, VerificationStatus)
            else str(record.verification_status)
        )

        with self._connection() as conn:
            cursor = conn.execute("""
                INSERT INTO mitigation_history (
                    timestamp,
                    sensor_id,
                    attack_type,
                    confidence,
                    source_mac,
                    source_ip,
                    destination_ip,
                    mitigation_action,
                    action_description,
                    execution_status,
                    traffic_before,
                    traffic_after,
                    reduction_percentage,
                    verification_status,
                    simulation_mode,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp or datetime.now(timezone.utc).isoformat(),
                record.sensor_id or "ESP32_01",
                record.attack_type,
                float(record.confidence),
                record.source_mac,
                record.source_ip,
                record.destination_ip,
                record.mitigation_action,
                record.action_description,
                exec_status_str,
                float(record.traffic_before),
                float(record.traffic_after),
                float(record.reduction_percentage),
                verif_status_str,
                1 if record.simulation_mode else 0,
                record.notes or "",
            ))
            conn.commit()
            record_id = cursor.lastrowid
            record.id = record_id
            return record_id

    def get_recent_records(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves most recent mitigation attempts ordered newest first."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM mitigation_history ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single record by primary key."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM mitigation_history WHERE id = ?", (record_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates aggregated historical metrics."""
        with self._connection() as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_events,
                    SUM(CASE WHEN verification_status = 'MITIGATION_EFFECTIVE' THEN 1 ELSE 0 END) as effective_count,
                    SUM(CASE WHEN verification_status = 'PARTIALLY_EFFECTIVE' THEN 1 ELSE 0 END) as partial_count,
                    SUM(CASE WHEN verification_status = 'MITIGATION_FAILED' THEN 1 ELSE 0 END) as failed_count,
                    SUM(CASE WHEN execution_status = 'REQUIRES_ADMIN_APPROVAL' THEN 1 ELSE 0 END) as approval_count,
                    AVG(reduction_percentage) as avg_reduction
                FROM mitigation_history
            """)
            row = cursor.fetchone()
            if row and row["total_events"] > 0:
                return {
                    "total_events": row["total_events"],
                    "effective_count": row["effective_count"] or 0,
                    "partial_count": row["partial_count"] or 0,
                    "failed_count": row["failed_count"] or 0,
                    "approval_count": row["approval_count"] or 0,
                    "avg_reduction_pct": round(float(row["avg_reduction"] or 0.0), 2),
                }
            return {
                "total_events": 0,
                "effective_count": 0,
                "partial_count": 0,
                "failed_count": 0,
                "approval_count": 0,
                "avg_reduction_pct": 0.0,
            }

    def clear_history(self) -> None:
        """Clears all records in the mitigation history table."""
        with self._connection() as conn:
            conn.execute("DELETE FROM mitigation_history")
            conn.commit()
