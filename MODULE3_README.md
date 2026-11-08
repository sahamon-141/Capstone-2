# Module 3: Attack-Specific Mitigation and Verification (Windows Simulation)

## Table of Contents
1. [Module 3 Architecture](#1-module-3-architecture)
2. [Explanation of Each Component](#2-explanation-of-each-component)
3. [How Module 3 Receives Attack Information](#3-how-module-3-receives-attack-information)
4. [How Mitigation Policies Work](#4-how-mitigation-policies-work)
5. [How the Simulated Controller Works](#5-how-the-simulated-controller-works)
6. [How Mitigation Verification Works](#6-how-mitigation-verification-works)
7. [How to Run Module 3 on Windows](#7-how-to-run-module-3-on-windows)
8. [How to Run Individual Simulation Scenarios](#8-how-to-run-individual-simulation-scenarios)
9. [How to Run Automated Tests](#9-how-to-run-automated-tests)
10. [Limitations of Simulation Mode](#10-limitations-of-simulation-mode)
11. [What Changes Will Be Needed for Raspberry Pi Deployment](#11-what-changes-will-be-needed-for-raspberry-pi-deployment)

---

## 1. Module 3 Architecture

Module 3 completes the **Three-Stage Hierarchical IoT Intrusion Defense System**:

```
[ Incoming Network Flows ]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Mamdani Fuzzy Anomaly Detector (Module 1)    │
│  - Sliding Window (W=20), IAT Dispersion (CV), Rate   │
│  - Attack Risk ∈ [0, 1]                                │
└────────────────────────────────────────────────────────┘
           │
           │  Attack Risk ≥ 0.35 (Anomaly Alert)
           ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: Deep Threat Attribution & Class (Module 2)   │
│  - Multiclass Classifier (LSTM / Pure-NumPy RF)        │
│  - Class Label, Confidence, Zero-Day MSP Rejection     │
└────────────────────────────────────────────────────────┘
           │
           │  Prediction: Attack Type + Confidence + Source MAC/IP
           ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Attack-Specific Mitigation & Verification (Module 3 - This Module) │
│                                                                              │
│   ┌─────────────────────────────┐         ┌──────────────────────────────┐   │
│   │ A. Attack Info Receiver     │ ──────> │ B. Mitigation Policy Engine  │   │
│   │    (Module 2/ESP32 payload) │         │    (policies.json, CICIoT2023)│   │
│   └─────────────────────────────┘         └──────────────┬───────────────┘   │
│                                                          │                   │
│                                                          ▼                   │
│   ┌─────────────────────────────┐         ┌──────────────────────────────┐   │
│   │ D. Verification Engine      │ <────── │ C. Simulated Network Ctrl    │   │
│   │    % Reduction & Efficacy   │         │    (In-Memory Safe Windows)  │   │
│   └──────────────┬──────────────┘         └──────────────┬───────────────┘   │
│                  │                                       │                   │
│                  ▼                                       ▼                   │
│   ┌──────────────────────────────────────────────────────────────────────┐   │
│   │ Persistence & Logging: SQLite (mitigation_history.db) + Log File     │   │
│   └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────┐
│  PRESENTATION TIERS:                                   │
│  1. Streamlit Dashboard (app.py - Tab 2)               │
│  2. Standalone Flask Web App & REST API (web/app.py)   │
└────────────────────────────────────────────────────────┘
```

---

## 2. Explanation of Each Component

| Component | Class / File | Primary Responsibility |
|---|---|---|
| **Attack Information Receiver** | `MitigationEngine.parse_attack_payload()` in `mitigation_engine.py` | Ingests predictions from Module 2 or simulation, normalizes timestamps, sensor ID (`ESP32_01`), source MAC/IP, and traffic statistics. |
| **Mitigation Policy Engine** | `MitigationPolicyEngine` in `mitigation_policies.py` | Evaluates external `policies.json`. Resolves CICIoT2023 classes onto attack families via aliases, validates confidence thresholds, and enforces administrator approval flags. |
| **Simulated Network Controller** | `SimulatedNetworkController` in `simulated_network_controller.py` | Implements an abstract `BaseNetworkController` adapter. Safely simulates PMF activation, VLAN isolation, rate limiting, and WAF rules in-memory on Windows without altering OS network settings. |
| **Traffic Simulator** | `TrafficSimulator` in `traffic_simulator.py` | Standalone generator of reproducible before- and after-mitigation traffic measurements across all attack profiles. Kept strictly separate from verification math. |
| **Mitigation Verification Engine** | `MitigationVerifier` in `mitigation_verifier.py` | Computes percentage traffic reduction over equal observation windows. Assigns effectiveness status (`MITIGATION_EFFECTIVE`, `PARTIALLY_EFFECTIVE`, `MITIGATION_FAILED`, `INSUFFICIENT_DATA`). |
| **Database Persistence** | `MitigationDatabase` in `database.py` | Thread-safe SQLite manager persisting all mitigation records to `data/mitigation_history.db`. |
| **Simulation Scenarios Catalog** | `SCENARIOS_CATALOG` in `scenarios.py` | Predefined suite of 8 reproducible simulation scenarios covering all execution and verification edge cases. |

---

## 3. How Module 3 Receives Attack Information

Module 3 ingests prediction events through two primary mechanisms:

### A. Programmatic Pipeline Handoff (Module 1 → Module 2 → Module 3)
```python
from module3_mitigation.mitigation_engine import MitigationEngine

engine = MitigationEngine()

# pipeline_output is the dictionary returned by PipelineIntegrator.process_module1_window()
module2_prediction = {
    "window_id": 42,
    "module1_verdict": "ATTACK",
    "attack_risk": 0.89,
    "module2_invoked": True,
    "predicted_attack": "Deauth",
    "raw_class": "Deauth",
    "confidence": 0.97,
    "is_zero_day": False
}

record = engine.mitigate_from_pipeline(
    pipeline_output=module2_prediction,
    source_mac="00:14:22:01:23:45",
    sensor_id="ESP32_01"
)
```

### B. Standardized Dictionary / REST API Ingestion
Module 3 can receive attack events directly via JSON payload or REST endpoint `POST /api/mitigate`:
```json
{
    "attack_type": "Deauth",
    "confidence": 0.97,
    "source_mac": "00:14:22:01:23:45",
    "source_ip": "192.168.1.185",
    "destination_ip": "192.168.1.1",
    "timestamp": "2026-11-04T12:30:00Z",
    "sensor_id": "ESP32_01",
    "traffic_before": 500.0,
    "traffic_after": 15.0
}
```

---

## 4. How Mitigation Policies Work

Policies are defined in an external JSON configuration file (`module3_mitigation/policies.json`), eliminating hardcoded `if/elif` statements.

### Supported CICIoT2023 attack families and their actions

The policy file is keyed to the classes Module 2 actually predicts. Every
CICIoT2023 class reaches one of these families; `action_class` tells the
escalation controller how the action changes traffic (`FILTER` caps the source
rate, `BLOCK` stops it, `ALERT` changes nothing).

| Family | Action | action_class | Automated? | Approval? | Covers |
|---|---|---|---|---|---|
| **Benign** | `NO_ACTION` | ALERT | Yes | No | BenignTraffic |
| **Volumetric_Flood** | `RATE_LIMIT_SOURCE` | FILTER | Yes | No | DDoS/DoS UDP, ICMP and TCP floods, SynonymousIP |
| **TCP_State_Exhaustion** | `ENABLE_SYN_COOKIES` | FILTER | Yes | No | SYN, PSHACK and RSTFIN floods |
| **Fragmentation_Attack** | `DROP_MALFORMED_FRAGMENTS` | FILTER | Yes | No | ICMP, UDP and ACK fragmentation |
| **Slow_Application_Attack** | `LIMIT_CONCURRENT_CONNECTIONS` | FILTER | Yes | No | SlowLoris, HTTP floods |
| **Reconnaissance** | `BLOCK_SCANNING_SOURCE` | BLOCK | Yes | No | PortScan, OSScan, HostDiscovery, PingSweep, VulnerabilityScan |
| **Spoofing_MITM** | `ENFORCE_ARP_DNS_INTEGRITY` | BLOCK | Yes | No | MITM-ArpSpoofing, DNS_Spoofing |
| **Web_Application_Attack** | `ACTIVATE_WAF_RULE` | FILTER | Yes | No | SqlInjection, CommandInjection, XSS, BrowserHijacking, Uploading_Attack |
| **Credential_Attack** | `THROTTLE_AUTH_ATTEMPTS` | FILTER | Yes | No | DictionaryBruteForce |
| **Malware_Backdoor** | `QUARANTINE_ENDPOINT` | BLOCK | No | **Yes** | Backdoor_Malware |
| **Botnet** | `ISOLATE_INFECTED_DEVICE` | BLOCK | Yes | No | a named botnet (see the note below on Mirai) |
| **Unknown (fallback)** | `BEHAVIOUR_BASED_CONTAINMENT` | FILTER | Yes | No | anything with no class policy, including zero-day |

Two choices worth explaining:

* **Floods are rate limited, not isolated.** A DDoS flood usually arrives from
  spoofed or remote sources, so isolating a local device does not stop it and
  cuts off a device that may be innocent.
* **The Mirai classes are deliberately not aliased to Botnet.** They are held
  out of training as the zero-day proxy, so letting them match a class policy
  would contaminate that experiment. They take the behaviour route instead. Add
  the aliases back for a deployment where Mirai is a known class.

### Zero-day containment

A classifier cannot name an attack it has never seen, so the fallback policy
does not just raise an alert. `zero_day_containment.py` picks a containment
ladder from the behaviour Module 1 already measured:

| Observed | Behaviour | Ladder |
|---|---|---|
| CV low, rate above the normal median | `CONSTANT_RATE_FLOOD` | rate limit → block → upstream blackhole |
| rate above normal p95, deviation high | `VOLUMETRIC_SURGE` | rate limit → blackhole → isolate |
| CV high, rate not high | `SLOW_PROBE` | block the source → isolate segment |
| flagged but no recognised shape | `UNCLASSIFIED_ANOMALY` | rate limit → block → isolate |

The CV and deviation breakpoints are the same ones the Mamdani membership
functions use, so the behaviour classes line up with the rules that flagged the
window. On CICIoT2023 this identifies all 759 held-out Mirai windows as
constant-rate floods from timing alone, without knowing the class name.

### Escalation and re-verification

`escalation_controller.py` walks the ladder least disruptive rung first, and
after each rung re-scores the window with the **real** Mamdani system from
`src/fuzzy_system.py`. A window is contained once that risk falls below 0.35;
otherwise the controller escalates.

Be careful what you claim from this. Two different things are reported:

* **Volume suppression** is `(rate − min(rate, cap)) / rate`. Both rates are
  measured, so this is arithmetic on real numbers.
* **Risk containment** is modelled. The residual rate is re-scored while CV and
  IAT deviation are held at their observed values, because there is no way to
  predict how a token bucket reshapes packet timing without a live testbed.
  That makes it a conservative lower bound. A blocking action contains by
  construction, so `contained_by_filter_alone` is the informative figure and
  `contained_overall` is not an accuracy.

### Automated vs. recommended manual remediations

Quarantining an endpoint cuts off all of its traffic, so that policy requires an
administrator to approve it even at high confidence.
- `AUTOMATED_ACTION`: execution proceeds immediately in simulation.
- `RECOMMENDED_MANUAL`: flags `REQUIRES_ADMIN_APPROVAL` with advisory detail.

### Low-confidence predictions

If `confidence < policy.confidence_threshold` (for example SQL injection at 0.42
against a 0.70 threshold), the policy engine marks the action as requiring
administrator confirmation.

---

## 5. How the Simulated Controller Works

The `SimulatedNetworkController` subclass implements the `BaseNetworkController` interface:

```python
class BaseNetworkController(ABC):
    @abstractmethod
    def execute_mitigation(self, policy, event, force_status=None) -> ExecutionResult:
        pass
    @abstractmethod
    def rollback_mitigation(self, action, target) -> ExecutionResult:
        pass
    @abstractmethod
    def get_active_mitigations(self) -> Dict[str, Any]:
        pass
```

### Safety Guarantee
- **Zero OS Modifications:** The controller strictly manipulates in-memory Python structures (`self.isolated_macs`, `self.rate_limited_ips`, `self.active_waf_rules`, `self.pmf_enabled`).
- **No System Commands:** It never invokes `netsh`, `iptables`, Windows Defender Firewall, or router CLI tools.

### Distinct Execution Statuses
1. `SIMULATED_SUCCESS`: The simulated security primitive was successfully applied in-memory.
2. `REQUIRES_ADMIN_APPROVAL`: The action is manual, or the classification confidence was low.
3. `SIMULATION_FAILED`: An explicit failure or simulated controller malfunction occurred.
4. `UNSUPPORTED_ACTION`: No policy exists for the supplied exploit category.

---

## 6. How Mitigation Verification Works

Verification assesses whether suspicious network traffic decreased after the simulated mitigation:

### Formula
$$\text{Reduction (\%)} = \left( \frac{\text{Traffic}_{\text{Before}} - \text{Traffic}_{\text{After}}}{\text{Traffic}_{\text{Before}}} \right) \times 100$$

### Decision Thresholds
- **`MITIGATION_EFFECTIVE`**: $\text{Reduction} \ge 80.0\%$
- **`PARTIALLY_EFFECTIVE`**: $30.0\% \le \text{Reduction} < 80.0\%$
- **`MITIGATION_FAILED`**: $\text{Reduction} < 30.0\%$ (including negative reductions where traffic surged)
- **`INSUFFICIENT_DATA`**: $\text{Traffic}_{\text{Before}} \le 0.0$ or missing telemetry measurements
- **`VERIFICATION_PENDING`**: Observation window still in progress

### Observation Windows & Collateral Impact
- Both before and after observations utilize equal time horizons (default: 60 seconds).
- Legitimate traffic collateral impact is reported (e.g. device isolation imposes a 100% impact on that specific endpoint's normal traffic, while PMF activation imposes 0%).

---

## 7. How to Run Module 3 on Windows

### Option A: Interactive Streamlit Dashboard (Recommended)
Module 3 is integrated into your main Streamlit application under Tab 2 (`🛡️ Module 3: Mitigation & Verification`):

```powershell
# In PowerShell from project root (d:\Projects\capstone):
py -3.14 -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Dedicated Flask Web Dashboard & REST API
A dedicated Flask application is provided in `web/app.py`:

```powershell
# Launch Flask server on port 5000:
py -3.14 web/app.py
```
Open your browser at `http://localhost:5000`.

---

## 8. How to Run Individual Simulation Scenarios

The system includes 8 reproducible test scenarios. You can run them via the Streamlit UI, the Flask UI, the REST API, or Python CLI:

### Via Python Interactive CLI
```powershell
py -3.14 -c "
from module3_mitigation.mitigation_engine import MitigationEngine
engine = MitigationEngine()

for sc_id in range(1, 9):
    rec = engine.run_scenario(sc_id)
    print(f'Scenario #{sc_id}: {rec.attack_type:<15} | Exec: {rec.execution_status:<24} | Verif: {rec.verification_status:<20} | Reduction: {rec.reduction_percentage}%')
"
```

### Scenario Breakdown

| ID | Scenario Name | Attack Category | Confidence | Pre-Traffic | Post-Traffic | Expected Exec Status | Expected Verif Status | Expected Reduction |
|---|---|---|---|---|---|---|---|---|
| **1** | Deauth Attack | `Deauth` | 0.97 | 500 pkts/min | 15 pkts/min | `SIMULATED_SUCCESS` | `MITIGATION_EFFECTIVE` | 97.0% |
| **2** | Botnet Attack | `Botnet` | 0.98 | 1200 pkts/min | 0 pkts/min | `SIMULATED_SUCCESS` | `MITIGATION_EFFECTIVE` | 100.0% |
| **3** | SSH Brute-Force | `SSH` | 0.92 | 400 pkts/min | 180 pkts/min | `SIMULATED_SUCCESS` | `PARTIALLY_EFFECTIVE` | 55.0% |
| **4** | Rogue AP Alert | `Rogue_AP` | 0.94 | 180 pkts/min | 180 pkts/min | `REQUIRES_ADMIN_APPROVAL` | `MITIGATION_FAILED` | 0.0% |
| **5** | Malware C2 | `Malware` | 0.95 | 800 pkts/min | 20 pkts/min | `SIMULATED_SUCCESS` | `MITIGATION_EFFECTIVE` | 97.5% |
| **6** | Unknown Exploit | `Unknown` | 0.70 | 300 pkts/min | 300 pkts/min | `UNSUPPORTED_ACTION` | `MITIGATION_FAILED` | 0.0% |
| **7** | Low-Confidence | `SQL_Injection` | 0.42 | 350 pkts/min | 350 pkts/min | `REQUIRES_ADMIN_APPROVAL` | `MITIGATION_FAILED` | 0.0% |
| **8** | Failed Mitigation | `SSDP` | 0.88 | 1000 pkts/min | 950 pkts/min | `SIMULATED_SUCCESS` | `MITIGATION_FAILED` | 5.0% |

---

## 9. How to Run Automated Tests

Execute the complete Module 3 test suite covering verification math, simulated controller actions, policy selection, database persistence, and end-to-end integration:

```powershell
py -3.14 -m unittest tests/test_mitigation_verifier.py tests/test_simulated_controller.py tests/test_mitigation_engine.py tests/test_module3_e2e.py
```

### Expected Test Output
```
.........................
----------------------------------------------------------------------
Ran 25 tests in 0.356s

OK
```

---

## 10. Limitations of Simulation Mode

Read this before quoting any Module 3 number in a report or paper.

1. **Nothing is enforced.** No packet on a real interface is throttled or
   dropped. Every action is recorded in memory only.
2. **Risk containment is modelled.** The residual rate after a filter is
   re-scored by the real Mamdani system, but CV and IAT deviation are held at
   their observed values, so the figure is a conservative lower bound rather
   than a measurement. A blocking action contains by construction, so
   `contained_overall` is not an accuracy.
3. **Volume suppression is measured.** `(rate − min(rate, cap)) / rate` uses
   the real packet rate of each window, so this one is arithmetic on real data.
4. **The dashboard profiles are illustrative.** `traffic_simulator.py` drives
   the single-event demo path. The authoritative numbers come from
   `pipeline_evaluation.py`, which runs over every window of the real stream.
5. **Percentage reduction suits volumetric attacks only.** A WAF rule or an
   authentication back-off stops an attack without moving the packet rate much,
   so a low reduction there does not mean the mitigation failed.
6. **Only some families are exercised.** The sampled evaluation stream contains
   floods, Mirai and benign traffic. The reconnaissance, spoofing, web,
   credential and malware policies are defined and unit tested but never
   triggered by this stream; `families_exercised_by_this_stream` in
   `module3_metrics.json` records which ones ran.
7. **Driven by the true class.** The batch evaluation uses each window's
   ground-truth label so Module 3's coverage and containment are measured
   independently of Module 2's error. That means Module 1 false positives are
   correctly left alone, which is the best case for collateral;
   `worst_case_receiving_a_blocking_action` gives the other bound.

---

## 11. What Changes Will Be Needed for Raspberry Pi Deployment

When deploying to a physical Raspberry Pi gateway (e.g. Raspberry Pi 4 / 5 running Raspberry Pi OS with a monitor-mode Wi-Fi interface):

```
                   Current Windows Simulation
                   ┌────────────────────────┐
                   │ SimulatedNetworkCtrl   │ (In-Memory State)
                   └────────────────────────┘
                               │
               Replace via Adapter Pattern with:
                               │
                               ▼
                   Physical Raspberry Pi Gateway
                   ┌────────────────────────┐
                   │ LinuxNetworkController │
                   └───────────┬────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  hostapd_cli / iw          iptables / nftables    tc (traffic control)
  (Enforce 802.11w PMF,     (Isolate MACs, Drop    (Token Bucket Filter
   AP station deauth)        outbound C2, WAF)      rate limiting port 22)
```

1. **Implement `LinuxNetworkController(BaseNetworkController)`:**
   - **Deauth / PMF:** Execute `hostapd_cli set ieee80211w 2` (PMF Required) and `hostapd_cli disassociate <MAC>`.
   - **Device Isolation:** Execute `iptables -I FORWARD -m mac --mac-source <MAC> -j DROP` or `ebtables -A FORWARD -s <MAC> -j DROP`.
   - **SSH Rate Limiting:** Execute `iptables -I INPUT -p tcp --dport 22 -m state --state NEW -m recent --set` followed by `--update --seconds 60 --hitcount 4 -j DROP`.
   - **UDP Port 1900 Restriction:** Execute `iptables -A INPUT -p udp --dport 1900 -j DROP`.
2. **Real Packet Capture for Verification:**
   - In place of `TrafficSimulator`, query live packet counts via `libpcap` / `scapy` / `tshark` over consecutive 60-second observation intervals directly before and after executing the iptables/hostapd command.
3. **Hardware Requirements:**
   - Wi-Fi adapter supporting AP mode + monitor mode with 802.11w PMF hardware support (e.g. Alfa AWUS036ACM / MediaTek MT7612U).

