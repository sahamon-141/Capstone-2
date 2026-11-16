"""
MODULE 3: attack-specific mitigation, suggestions and verification.

Question answered: "What should we do about it, and did it work?"

This is the single Module 3 of the project. It merges the two earlier versions:
    * policies    : the 11 CICIoT2023 policy families and their escalation ladders come from
                    module3_mitigation/policies.json (one source of truth)
    * zero-day    : unknown windows are routed by their traffic SHAPE with
                    module3_mitigation.zero_day_containment (constant-rate flood, slow probe,
                    volumetric surge, unclassified anomaly) instead of one generic playbook
    * verification: actions are applied to the actual flows of the window and the FULL
                    Module 1 (timing + behaviour) is re-run on the result

Steps
1. Route: Module 2's class -> policy family ("class_policy"), or, if Module 2 says
   Unknown, Module 1's CV / rate / IAT deviation -> behaviour ladder ("zero_day_behaviour").
2. Approval: the first, non-disruptive action may run automatically when the attack is known,
   Module 2 is at least as confident as the policy requires, and the policy does not demand an
   administrator. Disruptive actions (block / isolate / quarantine) always need an analyst.
   A zero-day window may start with a reversible rate limit automatically.
3. Apply (simulated controller): the flows Module 2 marked as attack are filtered with the
   action's efficacy; legitimate traffic keeps arriving. Ground-truth labels are never used.
4. Verify: re-run Module 1. Risk < 0.35 -> resolved. Otherwise escalate to the next action.

Nothing here touches a real firewall. The commands show what each action would be on a Linux
IoT gateway, and the efficacy values are assumptions of the simulation, not measurements.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ids.config import NORMAL_LABEL, ROOT, RISK_SUSPICIOUS, WINDOW_SIZE
from ids.module1_detection import FuzzyAnomalyDetector
from module3_mitigation.zero_day_containment import ZeroDayBehaviour, ZeroDayContainment

POLICY_FILE = ROOT / "module3_mitigation" / "policies.json"


@dataclass
class Action:
    key: str
    name: str
    description: str
    kind: str                  # "cap" (filter / limit), "stop" (block / isolate), "alert"
    efficacy: float            # share of targeted attack flows removed (simulation assumption)
    command: str               # what it would be on a Linux IoT gateway

    @property
    def disruptive(self) -> bool:
        return self.kind == "stop"


# Every action that appears in a ladder of policies.json or of the zero-day router.
ACTIONS = {a.key: a for a in [
    Action("RATE_LIMIT_SOURCE", "Per-source rate limit",
           "Token bucket caps each source at a normal-traffic packet rate.", "cap", 0.75,
           "iptables -A INPUT -m hashlimit --hashlimit-above 200/sec --hashlimit-mode srcip "
           "--hashlimit-name m3 -j DROP"),
    Action("ENABLE_SYN_COOKIES", "SYN cookies + drop invalid TCP flags",
           "Half-open connections no longer use memory; packets with impossible flag combinations are dropped.",
           "cap", 0.80,
           "sysctl -w net.ipv4.tcp_syncookies=1\n"
           "iptables -A INPUT -p tcp -m conntrack --ctstate INVALID -j DROP"),
    Action("DROP_MALFORMED_FRAGMENTS", "Drop malformed / overlapping fragments",
           "Limit the reassembly buffer per source and drop incomplete or overlapping fragments.", "cap", 0.80,
           "sysctl -w net.ipv4.ipfrag_high_thresh=262144\niptables -A INPUT -f -m limit --limit 50/sec -j ACCEPT\n"
           "iptables -A INPUT -f -j DROP"),
    Action("LIMIT_CONCURRENT_CONNECTIONS", "Limit concurrent connections + shorter timeouts",
           "Cap connections per client and close slow / partial requests early.", "cap", 0.80,
           "iptables -A INPUT -p tcp --syn --dport 80 -m connlimit --connlimit-above 20 -j REJECT\n"
           "# web server: RequestReadTimeout header=10-20,MinRate=500"),
    Action("ACTIVATE_WAF_RULE", "Activate the matching WAF rule",
           "Web application firewall filters the malicious payloads; legitimate requests still pass.", "cap", 0.85,
           "SecRuleEngine On   # ModSecurity + OWASP Core Rule Set\n"
           "SecRule ARGS \"@detectSQLi\" \"id:1001,deny,status:403\""),
    Action("THROTTLE_AUTH_ATTEMPTS", "Throttle failed logins (back-off + lockout)",
           "Exponential back-off on failed authentication and lockout after repeated failures.", "cap", 0.85,
           "fail2ban-client set sshd banip <attacker-ip>   # maxretry=5, bantime.increment=true"),
    Action("BLOCK_SOURCE", "Block the source address",
           "Drop all traffic from the attacking source for a fixed period.", "stop", 0.98,
           "iptables -I INPUT -s <attacker-ip> -j DROP   # auto-expire after 30 min"),
    Action("BLOCK_SCANNING_SOURCE", "Block the scanner (+ honeypot redirect)",
           "Temporarily block the probing source, or redirect it to a honeypot to observe it.", "stop", 0.97,
           "iptables -I INPUT -s <scanner-ip> -j DROP\n"
           "# or: iptables -t nat -A PREROUTING -s <scanner-ip> -j DNAT --to <honeypot-ip>"),
    Action("UPSTREAM_BLACKHOLE", "Upstream filtering / blackhole",
           "Ask the ISP / upstream router to drop the flood before it reaches the IoT network.", "stop", 0.97,
           "ip route add blackhole <victim-prefix>   # or a BGP FlowSpec rule at the ISP"),
    Action("ISOLATE_SEGMENT", "Isolate the affected network segment",
           "Cut the affected segment off until an analyst has investigated.", "stop", 0.97,
           "iptables -A FORWARD -i <segment-if> -j DROP"),
    Action("ISOLATE_INFECTED_DEVICE", "Isolate infected devices",
           "Move the source devices to a quarantine VLAN with no internet access.", "stop", 0.95,
           "bridge vlan add dev <port> vid 999   # quarantine VLAN\niptables -A FORWARD -i vlan999 -j DROP"),
    Action("QUARANTINE_ENDPOINT", "Quarantine the endpoint + block C&C",
           "Quarantine the host and block outbound command-and-control destinations.", "stop", 0.97,
           "bridge vlan add dev <port> vid 999\niptables -A FORWARD -d <c2-ip-list> -j DROP"),
    Action("ENFORCE_ARP_DNS_INTEGRITY", "Enforce ARP / DNS integrity",
           "Dynamic ARP inspection with static bindings and DNSSEC validation; forged replies are dropped.",
           "stop", 0.95,
           "arp -s <gateway-ip> <gateway-mac>\nip arp inspection vlan <id>   # managed switch\n"
           "iptables -A FORWARD -p udp --dport 53 ! -d <trusted-dns> -j DROP"),
    Action("ALERT_ADMIN_AND_LOG", "Alert the administrator", "Notify the operator; no traffic is changed.",
           "alert", 0.0, "logger -p auth.alert 'IoT Sentinel: anomaly requires review'"),
]}

SEVERITY = {
    "Volumetric_Flood": "CRITICAL", "TCP_State_Exhaustion": "CRITICAL", "Fragmentation_Attack": "HIGH",
    "Slow_Application_Attack": "HIGH", "Reconnaissance": "MEDIUM", "Spoofing_MITM": "HIGH",
    "Web_Application_Attack": "HIGH", "Credential_Attack": "MEDIUM", "Malware_Backdoor": "CRITICAL",
    "Botnet": "CRITICAL", "Zero-day": "CRITICAL",
}

WHY = {
    "Volumetric_Flood": "A flood of packets is exhausting bandwidth and the target's capacity.",
    "TCP_State_Exhaustion": "Forged TCP handshakes / flags fill the target's connection table.",
    "Fragmentation_Attack": "Fragmented packets exhaust the target's reassembly memory.",
    "Slow_Application_Attack": "Many slow or heavy HTTP requests keep the web service busy.",
    "Reconnaissance": "Someone is mapping the network: usually the first stage before an attack.",
    "Spoofing_MITM": "An attacker impersonates a trusted host to intercept or redirect traffic.",
    "Web_Application_Attack": "Malicious payloads target a web interface (injection, XSS, hijacking, upload).",
    "Credential_Attack": "Repeated login attempts are trying to guess a password.",
    "Malware_Backdoor": "A host may already be compromised and controlled remotely.",
    "Botnet": "Infected IoT devices are controlled by an attacker and can spread further.",
}

RECOMMENDATIONS = {
    "Volumetric_Flood": ["Put exposed IoT services behind an upstream DDoS scrubbing service.",
                         "Apply ingress filtering (BCP 38 / uRPF) so spoofed sources are dropped at the edge."],
    "TCP_State_Exhaustion": ["Keep SYN cookies enabled permanently and tune the SYN backlog.",
                             "Use a stateful firewall that drops packets for unknown connections."],
    "Fragmentation_Attack": ["Drop fragments for services that never need them.",
                             "Keep the gateway's reassembly limits small."],
    "Slow_Application_Attack": ["Run the web interface behind a reverse proxy with strict timeouts.",
                                "Rate-limit requests per client and add a JS / CAPTCHA challenge for floods."],
    "Reconnaissance": ["Close unused ports and services on IoT devices.",
                       "Patch firmware: vulnerability scans look for known CVEs.",
                       "Watch the scanner's address for follow-up attacks."],
    "Spoofing_MITM": ["Enable DNSSEC validation on the resolver.",
                      "Use authenticated, encrypted protocols (TLS, SSH) so intercepted traffic is useless."],
    "Web_Application_Attack": ["Validate and parameterise all inputs (prepared statements for SQL).",
                               "Patch the device's web interface; scan the host for uploaded files."],
    "Credential_Attack": ["Use long unique passwords or keys; enable MFA.",
                          "Move management interfaces off the public network."],
    "Malware_Backdoor": ["Re-image or re-flash the host before it rejoins the network.",
                         "Review outbound connections from other devices for the same C&C."],
    "Botnet": ["Reset every device's default credentials (Mirai spreads with factory logins).",
               "Re-flash firmware; disable Telnet / UPnP on IoT devices."],
    "Zero-day": ["Capture the traffic (tcpdump) and send it to a security analyst.",
                 "Once understood, add the new attack as a class and retrain Module 2."],
}

CLASS_TIPS = {
    "DDoS-SYN_Flood": "SYN flood: enable SYN cookies and shorten SYN-RECEIVED timeout.",
    "DoS-SYN_Flood": "SYN flood: enable SYN cookies and shorten SYN-RECEIVED timeout.",
    "DDoS-ICMP_Flood": "ICMP flood: rate-limit ICMP echo requests (iptables -p icmp -m limit).",
    "DDoS-ICMP_Fragmentation": "Fragmented ICMP: drop fragmented ICMP packets at the edge.",
    "DDoS-UDP_Flood": "UDP flood: drop UDP to ports with no listening service.",
    "DoS-UDP_Flood": "UDP flood: drop UDP to ports with no listening service.",
    "DDoS-UDP_Fragmentation": "Fragmented UDP: drop non-initial fragments for UDP services.",
    "DDoS-ACK_Fragmentation": "ACK fragmentation: drop fragmented TCP ACKs without a session.",
    "DDoS-PSHACK_Flood": "PSH+ACK flood: stateful filtering drops packets with no session.",
    "DDoS-RSTFINFlood": "RST/FIN flood: stateful firewall drops RST/FIN for unknown connections.",
    "DDoS-TCP_Flood": "TCP flood: connection-rate limits per source.",
    "DoS-TCP_Flood": "TCP flood: connection-rate limits per source.",
    "DDoS-SynonymousIP_Flood": "Spoofed-source flood: ingress filtering (uRPF) drops forged sources.",
    "DDoS-HTTP_Flood": "HTTP flood: rate-limit requests per client and add a JS / CAPTCHA challenge.",
    "DoS-HTTP_Flood": "HTTP flood: rate-limit requests per client and add a JS / CAPTCHA challenge.",
    "DDoS-SlowLoris": "SlowLoris: reduce header timeout and limit connections per IP on the web server.",
    "Recon-PortScan": "Port scan: close unused ports; block sources hitting many ports.",
    "Recon-OSScan": "OS fingerprinting: normalise TTL/TCP options; block the scanner.",
    "Recon-HostDiscovery": "Host discovery: drop ICMP echo / ARP sweeps from untrusted segments.",
    "Recon-PingSweep": "Ping sweep: rate-limit ICMP echo requests.",
    "VulnerabilityScan": "Vulnerability scan: patch the CVEs being probed; block the scanner.",
    "MITM-ArpSpoofing": "ARP spoofing: static ARP for the gateway, dynamic ARP inspection on the switch.",
    "DNS_Spoofing": "DNS spoofing: DNSSEC validation and force the trusted resolver.",
    "SqlInjection": "SQL injection: parameterised queries + WAF SQLi rules.",
    "CommandInjection": "Command injection: never pass input to a shell; WAF RCE rules.",
    "XSS": "XSS: output encoding + Content-Security-Policy header.",
    "BrowserHijacking": "Browser hijacking: block the malicious script host; CSP header.",
    "Backdoor_Malware": "Back-door malware: isolate the host and run a malware scan.",
    "Uploading_Attack": "Malicious upload: restrict file types and scan uploads.",
    "DictionaryBruteForce": "Dictionary attack: fail2ban + MFA on the login service.",
}

BEHAVIOUR_LABEL = {
    "CONSTANT_RATE_FLOOD": "Constant-rate flood",
    "VOLUMETRIC_SURGE": "Volumetric surge",
    "SLOW_PROBE": "Slow probe / scan",
    "UNCLASSIFIED_ANOMALY": "Unclassified anomaly",
}


def load_policies(path=POLICY_FILE) -> tuple[dict, dict]:
    """policies.json -> ({family: policy}, {class name lower-case: family})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    policies, class_to_family = {}, {}
    for p in data["policies"]:
        policies[p["attack_type"]] = p
        for alias in p.get("aliases", []):
            class_to_family[alias.lower()] = p["attack_type"]
    return policies, class_to_family


class WindowShapeRouter(ZeroDayContainment):
    """
    The zero-day router of module3_mitigation, plus one rule: a CV below 0.05 is machine-made
    timing whatever the rate (no benign window in CICIoT2023 goes below CV ~ 0.23), so it is a
    constant-rate flood. CICIoT2023's per-flow Rate of some Mirai GRE floods is below the normal
    median, which the original rate condition would miss.
    """
    CV_MACHINE_MAX = 0.05

    def classify(self, cv, rate, iat_deviation):
        if cv < self.CV_MACHINE_MAX:
            return ZeroDayBehaviour.CONSTANT_RATE_FLOOD
        return super().classify(cv, rate, iat_deviation)


@dataclass
class MitigationStep:
    stage: int
    action: Action
    flows_before: int
    attack_flows_before: int
    attack_flows_after: int
    risk_before: float
    risk_after: float
    cv_before: float
    cv_after: float
    rate_before: float
    rate_after: float
    volume_before: float        # packet rate of the targeted (attack-predicted) flows, summed
    volume_after: float         # packet rate of the targeted flows still getting through
    resolved: bool

    @property
    def volume_suppression(self) -> float:
        return 0.0 if self.volume_before <= 0 else 1 - self.volume_after / self.volume_before


@dataclass
class MitigationPlan:
    route: str                  # "class_policy" or "zero_day_behaviour"
    family: str                 # policy family, or "Zero-day"
    attack_class: str
    severity: str
    why: str
    class_tip: str
    actions: list
    recommendations: list
    auto_execute: bool
    approval_reason: str
    confidence: float
    behaviour: str | None = None
    behaviour_reason: str | None = None
    policy_description: str = ""
    steps: list = field(default_factory=list)
    final_window: pd.DataFrame = None
    resolved: bool = False

    @property
    def needed_disruptive_action(self) -> bool:
        return any(s.action.disruptive for s in self.steps)


class MitigationEngine:
    def __init__(self, detector: FuzzyAnomalyDetector, benign_pool: pd.DataFrame, seed: int = 7):
        self.detector = detector
        self.benign_pool = benign_pool
        self.rng = np.random.default_rng(seed)
        self.policies, self.class_to_family = load_policies()

        # Behaviour thresholds for the zero-day router come from BENIGN window mean rates,
        # the same statistic the router was designed for (median and 95th percentile).
        rates = [benign_pool.sample(WINDOW_SIZE, random_state=i)["Rate"].mean() for i in range(500)]
        self.rate_percentiles = {"p50": float(np.percentile(rates, 50)), "p95": float(np.percentile(rates, 95))}
        self.zero_day = WindowShapeRouter(baseline_stats={"rate_percentiles": self.rate_percentiles})

    # -- 1 + 2: route and approval ----------------------------------------------------
    def family_of(self, attack_class: str) -> str | None:
        return self.class_to_family.get(str(attack_class).lower())

    def plan(self, attack_class: str, confidence: float, is_zero_day: bool, window: pd.DataFrame,
             targeted=None) -> MitigationPlan:
        """targeted: per-flow mask of the flows Module 2 marked as attack (used for the zero-day shape)."""
        family = None if is_zero_day else self.family_of(attack_class)
        if family is not None and family != "Benign":
            pol = self.policies[family]
            actions = [ACTIONS[k] for k in pol["escalation_ladder"]]
            reasons = []
            if confidence < pol["confidence_threshold"]:
                reasons.append(f"Module 2 confidence {confidence:.0%} < {pol['confidence_threshold']:.0%} required by this policy")
            if pol.get("requires_admin_approval"):
                reasons.append("policy requires an administrator")
            if actions[0].disruptive:
                reasons.append("first action blocks traffic")
            return MitigationPlan(
                route="class_policy", family=family, attack_class=attack_class,
                severity=SEVERITY.get(family, "HIGH"), why=WHY.get(family, pol["description"]),
                class_tip=CLASS_TIPS.get(attack_class, ""), actions=actions,
                recommendations=RECOMMENDATIONS.get(family, []), auto_execute=not reasons,
                approval_reason="; ".join(reasons) or "known attack, confident, reversible first action",
                confidence=confidence, policy_description=pol["description"],
            )

        # Zero-day (or a class without a policy): choose a containment ladder from the traffic
        # shape. The shape is measured on the suspicious flows only, because benign background
        # flows in the same window would hide e.g. the regular timing of a flood.
        sus = window
        if targeted is not None and int(np.sum(targeted)) >= 3:
            sus = window[np.asarray(targeted, dtype=bool)]
        f = self.detector.window_features(sus)
        mean_rate = float(sus["Rate"].mean())
        d = self.zero_day.decide(f["cv"], mean_rate, f["dev"])
        actions = [ACTIONS[k] for k in d.escalation_ladder]
        auto = not actions[0].disruptive
        return MitigationPlan(
            route="zero_day_behaviour", family="Zero-day", attack_class=attack_class,
            severity=SEVERITY["Zero-day"], why="Anomalous traffic that matches no known attack (possible zero-day).",
            class_tip="", actions=actions, recommendations=RECOMMENDATIONS["Zero-day"],
            auto_execute=auto,
            approval_reason=("reversible rate limit may start automatically; any blocking stage needs an analyst"
                             if auto else "zero-day: first action blocks traffic"),
            confidence=confidence, behaviour=BEHAVIOUR_LABEL[d.behaviour.value], behaviour_reason=d.reason,
        )

    # -- 3: simulated controller ---------------------------------------------------------
    def _apply(self, window: pd.DataFrame, action: Action) -> pd.DataFrame:
        """
        Filter the flows Module 2 marked as attack ("_targeted"), each with probability =
        efficacy. Legitimate traffic keeps arriving, so removed slots are refilled with
        benign flows. Ground-truth labels are NOT used.
        """
        targeted = window["_targeted"].to_numpy(dtype=bool)
        keep = ~targeted | (self.rng.random(len(window)) > action.efficacy)
        kept = window[keep]
        refill = self.benign_pool.sample(len(window) - len(kept), random_state=int(self.rng.integers(1e9)))
        refill = refill.assign(_targeted=False)
        return pd.concat([kept, refill]).sample(frac=1.0, random_state=int(self.rng.integers(1e9)))

    # -- 4: verify and escalate ----------------------------------------------------------
    def execute_and_verify(self, plan: MitigationPlan, window: pd.DataFrame, targeted) -> MitigationPlan:
        """targeted: boolean per flow, True where Module 2 predicted an attack class."""
        plan.steps = []
        current = window.assign(_targeted=np.asarray(targeted, dtype=bool))
        before = self.detector.analyse_window(current)
        for stage, action in enumerate(plan.actions, start=1):
            after_window = self._apply(current, action)
            after = self.detector.analyse_window(after_window)
            resolved = after.risk < RISK_SUSPICIOUS
            plan.steps.append(MitigationStep(
                stage=stage, action=action, flows_before=len(current),
                attack_flows_before=int((current["label"] != NORMAL_LABEL).sum()),
                attack_flows_after=int((after_window["label"] != NORMAL_LABEL).sum()),
                risk_before=before.risk, risk_after=after.risk,
                cv_before=before.features["cv"], cv_after=after.features["cv"],
                rate_before=before.features["rate"], rate_after=after.features["rate"],
                volume_before=float(current.loc[current["_targeted"], "Rate"].sum()),
                volume_after=float(after_window.loc[after_window["_targeted"], "Rate"].sum()),
                resolved=resolved,
            ))
            current, before = after_window, after
            if resolved:
                break
        plan.final_window = current
        plan.resolved = plan.steps[-1].resolved
        return plan
