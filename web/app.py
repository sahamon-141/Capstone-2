"""
Flask Web Dashboard and REST API for Module 3:
Attack-Specific Mitigation and Verification (Windows Simulation).

Runs standalone on Windows:
    python web/app.py
Access at: http://localhost:5000
"""

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from module3_mitigation.mitigation_engine import MitigationEngine
from module3_mitigation.database import MitigationDatabase
from module3_mitigation.models import AttackEvent, ExecutionStatus
from module3_mitigation.scenarios import SCENARIOS_CATALOG, list_all_scenarios, get_scenario

app = Flask(__name__, template_folder="templates")
engine = MitigationEngine()
database = engine.database


# ------------------------------------------------------------------------------
# Frontend Route
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    """Renders the main Module 3 web dashboard."""
    scenarios = list_all_scenarios()
    stats = database.get_summary_stats()
    recent_history = database.get_recent_records(limit=25)
    supported_attacks = engine.policy_engine.get_supported_attack_types()
    return render_template(
        "index.html",
        scenarios=scenarios,
        stats=stats,
        recent_history=recent_history,
        supported_attacks=supported_attacks,
    )


# ------------------------------------------------------------------------------
# REST API Endpoints
# ------------------------------------------------------------------------------
@app.route("/api/scenarios", methods=["GET"])
def api_get_scenarios():
    """Returns catalog of the 8 reproducible simulation scenarios."""
    return jsonify({"status": "success", "scenarios": list_all_scenarios()})


@app.route("/api/run-scenario/<int:scenario_id>", methods=["POST"])
def api_run_scenario(scenario_id: int):
    """Executes a specific scenario by ID (1..8)."""
    try:
        record = engine.run_scenario(scenario_id)
        return jsonify({
            "status": "success",
            "record": record.to_dict(),
            "stats": database.get_summary_stats(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/mitigate", methods=["POST"])
def api_mitigate_custom():
    """
    Ingests an attack prediction and executes simulated mitigation & verification.
    Expected JSON body:
    {
        "attack_type": "Deauth",
        "confidence": 0.95,
        "source_mac": "00:11:22:33:44:55",
        "source_ip": "192.168.1.50",
        "destination_ip": "192.168.1.1",
        "sensor_id": "ESP32_01",
        "traffic_before": 500.0,
        "traffic_after": 20.0,
        "force_status": null
    }
    """
    data = request.get_json(force=True) or {}
    try:
        event = AttackEvent(
            attack_type=data.get("attack_type", "Unknown"),
            confidence=float(data.get("confidence", 0.95)),
            source_mac=data.get("source_mac"),
            source_ip=data.get("source_ip"),
            destination_ip=data.get("destination_ip"),
            sensor_id=data.get("sensor_id", "ESP32_01"),
        )
        t_before = float(data["traffic_before"]) if "traffic_before" in data and data["traffic_before"] is not None else None
        t_after = float(data["traffic_after"]) if "traffic_after" in data and data["traffic_after"] is not None else None
        force_status_str = data.get("force_status")
        force_status = ExecutionStatus(force_status_str) if force_status_str else None

        record = engine.process_attack(
            event=event,
            traffic_before=t_before,
            traffic_after=t_after,
            force_status=force_status,
        )
        return jsonify({
            "status": "success",
            "record": record.to_dict(),
            "stats": database.get_summary_stats(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/api/history", methods=["GET"])
def api_get_history():
    """Returns recent mitigation history records."""
    limit = int(request.args.get("limit", 50))
    records = database.get_recent_records(limit=limit)
    return jsonify({"status": "success", "count": len(records), "records": records})


@app.route("/api/stats", methods=["GET"])
def api_get_stats():
    """Returns summary analytics on historical mitigations."""
    return jsonify({"status": "success", "stats": database.get_summary_stats()})


@app.route("/api/clear-history", methods=["POST"])
def api_clear_history():
    """Clears SQLite mitigation history."""
    database.clear_history()
    return jsonify({"status": "success", "message": "Mitigation history cleared."})


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("   MODULE 3: ATTACK MITIGATION & VERIFICATION DASHBOARD (FLASK)")
    print("           Mode: WINDOWS NETWORK SIMULATION (SAFE)")
    print("=" * 70)
    print("Starting Flask web server at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

