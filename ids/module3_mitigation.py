"""
MODULE 3: attack-specific mitigation, suggestions and verification.

Question answered: "What should we do about it, and did it work?"

1. Select the playbook for the attack family Module 2 reported (and the class-specific tip).
2. Decide whether the action may run automatically (confident + reversible) or needs an
   analyst's approval (low confidence, zero-day, or disruptive action).
3. Apply the action in a SIMULATED network controller: attack flows are filtered with the
   action's efficacy and legitimate traffic keeps flowing.
4. Verify: re-run Module 1 on the post-mitigation traffic. If the risk is still >= 0.35,
   escalate to the next, stronger action. This closes the loop described in the paper:
   "after the attack is mitigated, the IAT dispersion should return to normal".

The controller is a simulation: nothing here touches a real firewall. The example commands
show what each action would be on a Linux gateway (iptables / tc / sysctl).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ids.config import NORMAL_LABEL, RISK_SUSPICIOUS
from ids.module1_detection import FuzzyAnomalyDetector

AUTO_CONFIDENCE = 0.60   # below this an analyst must approve the action


@dataclass
class Action:
    name: str
    description: str
    efficacy: float            # share of attack flows the action removes (simulation)
    command: str               # what it would be on a Linux IoT gateway
    disruptive: bool = False   # may also hit legitimate users -> needs approval


PLAYBOOKS = {
    "DDoS": {
        "severity": "CRITICAL",
        "why": "Many sources are flooding the target; availability of the IoT service is at risk.",
        "actions": [
            Action("Per-source rate limiting + SYN cookies",
                   "Cap packets/s per source address and enable SYN cookies so half-open connections cannot exhaust memory.",
                   0.75, "iptables -A INPUT -m hashlimit --hashlimit-above 200/sec --hashlimit-mode srcip "
                         "--hashlimit-name ddos -j DROP\nsysctl -w net.ipv4.tcp_syncookies=1"),
            Action("Upstream filtering / blackhole the attack traffic",
                   "Ask the ISP / upstream router to drop the flood signature before it reaches the IoT network.",
                   0.97, "ip route add blackhole <victim-prefix>   # or BGP FlowSpec rule at the ISP", disruptive=True),
        ],
        "recommendations": [
            "Use an upstream DDoS scrubbing service or CDN in front of exposed IoT services.",
            "Apply ingress filtering (BCP 38) so spoofed sources are dropped at the edge.",
            "Keep spare capacity and auto-scaling for the gateway.",
        ],
    },
    "DoS": {
        "severity": "HIGH",
        "why": "A single source is flooding the target.",
        "actions": [
            Action("Rate-limit the offending source",
                   "Throttle the single attacking address to a normal packet rate.",
                   0.80, "iptables -A INPUT -s <attacker-ip> -m limit --limit 50/sec -j ACCEPT\n"
                         "iptables -A INPUT -s <attacker-ip> -j DROP"),
            Action("Block the source address",
                   "Drop all traffic from the attacking address for a fixed period.",
                   0.99, "iptables -I INPUT -s <attacker-ip> -j DROP   # auto-expire after 30 min"),
        ],
        "recommendations": [
            "Tune connection timeouts and backlog sizes on the targeted service.",
            "Add the source to a temporary deny-list and review if it is a compromised device.",
        ],
    },
    "Mirai": {
        "severity": "CRITICAL",
        "why": "Infected IoT devices (botnet) are generating floods and may infect others.",
        "actions": [
            Action("Quarantine infected devices",
                   "Move the source devices to an isolated quarantine VLAN with no internet access.",
                   0.90, "bridge vlan add dev <port> vid 999   # quarantine VLAN\n"
                         "iptables -A FORWARD -i vlan999 -j DROP"),
            Action("Block Telnet and C&C traffic network-wide",
                   "Close Telnet (23/2323) used by Mirai to spread and drop known C&C destinations.",
                   0.98, "iptables -A FORWARD -p tcp -m multiport --dports 23,2323 -j DROP\n"
                         "iptables -A FORWARD -d <c2-ip-list> -j DROP", disruptive=True),
        ],
        "recommendations": [
            "Reset every device's default username/password (Mirai spreads with ~60 factory logins).",
            "Re-flash firmware on infected devices: Mirai lives in memory, so reboot + patch.",
            "Disable Telnet/UPnP on all IoT devices.",
        ],
    },
    "Recon": {
        "severity": "MEDIUM",
        "why": "Someone is mapping the network: usually the first stage before an attack.",
        "actions": [
            Action("Throttle scanning and temporarily block the scanner",
                   "Limit new-connection attempts per source and block sources that probe many ports.",
                   0.85, "iptables -A INPUT -m recent --name scan --update --seconds 60 --hitcount 20 -j DROP\n"
                         "iptables -A INPUT -m recent --name scan --set"),
            Action("Redirect the scanner to a honeypot",
                   "Send the scanner to a decoy host to observe it without exposing real devices.",
                   0.97, "iptables -t nat -A PREROUTING -s <scanner-ip> -j DNAT --to <honeypot-ip>"),
        ],
        "recommendations": [
            "Close unused ports and services on IoT devices.",
            "Watch the scanner's address for follow-up attacks.",
            "Keep device firmware patched: vulnerability scans look for known CVEs.",
        ],
    },
    "Spoofing": {
        "severity": "HIGH",
        "why": "An attacker is impersonating a trusted host to intercept or redirect traffic.",
        "actions": [
            Action("Enforce ARP/DNS integrity",
                   "Pin the gateway's ARP entry / force the trusted DNS resolver and drop forged replies.",
                   0.85, "arp -s <gateway-ip> <gateway-mac>       # static ARP\n"
                         "iptables -A FORWARD -p udp --dport 53 ! -d <trusted-dns> -j DROP"),
            Action("Dynamic ARP inspection + isolate the spoofing port",
                   "Switch-level ARP inspection rejects forged ARP; the offending port is shut.",
                   0.97, "ip arp inspection vlan <id>   # on the managed switch\n"
                         "interface <port> shutdown", disruptive=True),
        ],
        "recommendations": [
            "Enable DNSSEC validation on the resolver.",
            "Use encrypted, authenticated protocols (TLS, SSH) so intercepted traffic is useless.",
        ],
    },
    "Web": {
        "severity": "HIGH",
        "why": "Application-layer attack against a web interface (injection, XSS, malware upload).",
        "actions": [
            Action("Activate WAF rules for the detected payload",
                   "Web application firewall drops requests matching injection / XSS / upload signatures.",
                   0.85, "SecRuleEngine On   # ModSecurity + OWASP Core Rule Set\n"
                         "SecRule ARGS \"@detectSQLi\" \"id:1001,deny,status:403\""),
            Action("Block the source and isolate the targeted host",
                   "Block the attacker and isolate the host in case it is already compromised.",
                   0.98, "iptables -I INPUT -s <attacker-ip> -j DROP", disruptive=True),
        ],
        "recommendations": [
            "Validate and parameterise all inputs (prepared statements for SQL).",
            "Patch the web interface / firmware of the IoT device.",
            "Scan the host for back-doors or uploaded files.",
        ],
    },
    "BruteForce": {
        "severity": "MEDIUM",
        "why": "Repeated login attempts are trying to guess a password.",
        "actions": [
            Action("Lock out the source after repeated failures",
                   "fail2ban-style ban of addresses with too many failed logins.",
                   0.90, "fail2ban-client set sshd banip <attacker-ip>"),
            Action("Block the source and require MFA",
                   "Permanent block and multi-factor authentication on the login service.",
                   0.99, "iptables -I INPUT -s <attacker-ip> -j DROP"),
        ],
        "recommendations": [
            "Use long unique passwords or key-based login; enable MFA.",
            "Move management interfaces off the public network.",
        ],
    },
    "Unknown": {
        "severity": "CRITICAL",
        "why": "Traffic is anomalous (Module 1) but matches no known attack (Module 2): possible zero-day.",
        "actions": [
            Action("Generic containment: rate-limit and capture",
                   "Apply a conservative rate limit to the anomalous flows and capture packets for analysis.",
                   0.70, "tc qdisc add dev eth0 root tbf rate 1mbit burst 32kbit latency 400ms\n"
                         "tcpdump -i eth0 -w zero_day_capture.pcap", disruptive=True),
            Action("Isolate the affected segment",
                   "Cut the affected network segment off until an analyst has investigated.",
                   0.97, "iptables -A FORWARD -i <segment> -j DROP", disruptive=True),
        ],
        "recommendations": [
            "Send the captured traffic to a security analyst / threat-intel service.",
            "Once understood, add the new attack as a class and retrain Module 2.",
        ],
    },
}

# Class-specific advice shown on top of the family playbook
CLASS_TIPS = {
    "DDoS-SYN_Flood": "SYN flood: enable SYN cookies and shorten SYN-RECEIVED timeout.",
    "DoS-SYN_Flood": "SYN flood: enable SYN cookies and shorten SYN-RECEIVED timeout.",
    "DDoS-ICMP_Flood": "ICMP flood: rate-limit ICMP echo requests (iptables -p icmp -m limit).",
    "DDoS-ICMP_Fragmentation": "Fragmented ICMP: drop fragmented ICMP packets at the edge.",
    "DDoS-UDP_Flood": "UDP flood: drop UDP to ports with no listening service.",
    "DoS-UDP_Flood": "UDP flood: drop UDP to ports with no listening service.",
    "DDoS-UDP_Fragmentation": "Fragmented UDP: drop non-initial fragments for UDP services.",
    "DDoS-ACK_Fragmentation": "ACK fragmentation: drop fragmented TCP ACKs without a session.",
    "DDoS-PSHACK_Flood": "PSH+ACK flood: use stateful filtering to drop packets with no session.",
    "DDoS-RSTFINFlood": "RST/FIN flood: stateful firewall drops RST/FIN for unknown connections.",
    "DDoS-TCP_Flood": "TCP flood: connection-rate limits per source.",
    "DoS-TCP_Flood": "TCP flood: connection-rate limits per source.",
    "DDoS-SynonymousIP_Flood": "Spoofed-source flood: ingress filtering (uRPF) drops forged sources.",
    "DDoS-HTTP_Flood": "HTTP flood: rate-limit requests per client and add a JS / CAPTCHA challenge.",
    "DoS-HTTP_Flood": "HTTP flood: rate-limit requests per client and add a JS / CAPTCHA challenge.",
    "DDoS-SlowLoris": "SlowLoris: reduce header timeout and limit connections per IP on the web server.",
    "Mirai-greeth_flood": "GRE-Ethernet flood: block GRE (protocol 47) where it is not used.",
    "Mirai-greip_flood": "GRE-IP flood: block GRE (protocol 47) where it is not used.",
    "Mirai-udpplain": "Plain UDP flood from bots: rate-limit UDP per source.",
    "Recon-PortScan": "Port scan: close unused ports; block sources hitting many ports.",
    "Recon-OSScan": "OS fingerprinting: normalise TTL/TCP options; block the scanner.",
    "Recon-HostDiscovery": "Host discovery: drop ICMP echo/ARP sweeps from untrusted segments.",
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
    resolved: bool


@dataclass
class MitigationPlan:
    family: str
    attack_class: str
    severity: str
    why: str
    class_tip: str
    actions: list
    recommendations: list
    auto_execute: bool
    approval_reason: str
    steps: list = field(default_factory=list)
    final_window: pd.DataFrame = None
    resolved: bool = False


class MitigationEngine:
    def __init__(self, detector: FuzzyAnomalyDetector, benign_pool: pd.DataFrame, seed: int = 7):
        self.detector = detector
        self.benign_pool = benign_pool
        self.rng = np.random.default_rng(seed)

    def plan(self, family: str, attack_class: str, confidence: float) -> MitigationPlan:
        pb = PLAYBOOKS.get(family, PLAYBOOKS["Unknown"])
        first = pb["actions"][0]
        reasons = []
        if family == "Unknown":
            reasons.append("zero-day: no known signature")
        if confidence < AUTO_CONFIDENCE:
            reasons.append(f"classifier confidence {confidence:.0%} < {AUTO_CONFIDENCE:.0%}")
        if first.disruptive:
            reasons.append("first action may affect legitimate users")
        return MitigationPlan(
            family=family, attack_class=attack_class, severity=pb["severity"], why=pb["why"],
            class_tip=CLASS_TIPS.get(attack_class, ""), actions=pb["actions"],
            recommendations=pb["recommendations"], auto_execute=not reasons,
            approval_reason="; ".join(reasons) or "high confidence, reversible action",
        )

    def _apply(self, window: pd.DataFrame, efficacy: float) -> pd.DataFrame:
        """
        Simulated controller: the action filters the flows Module 2 marked as attack
        ("_targeted"), each with probability = efficacy. Legitimate traffic keeps arriving,
        so filtered slots are refilled with benign flows. Ground-truth labels are NOT used.
        """
        targeted = window["_targeted"].to_numpy(dtype=bool)
        keep = ~targeted | (self.rng.random(len(window)) > efficacy)
        kept = window[keep]
        refill = self.benign_pool.sample(len(window) - len(kept), random_state=int(self.rng.integers(1e9)))
        refill = refill.assign(_targeted=False)
        return pd.concat([kept, refill]).sample(frac=1.0, random_state=int(self.rng.integers(1e9)))

    def execute_and_verify(self, plan: MitigationPlan, window: pd.DataFrame, targeted) -> MitigationPlan:
        """targeted: boolean per flow, True where Module 2 predicted an attack class."""
        current = window.assign(_targeted=np.asarray(targeted, dtype=bool))
        before = self.detector.analyse_window(current)
        for stage, action in enumerate(plan.actions, start=1):
            after_window = self._apply(current, action.efficacy)
            after = self.detector.analyse_window(after_window)
            resolved = after.risk < RISK_SUSPICIOUS
            plan.steps.append(MitigationStep(
                stage=stage, action=action, flows_before=len(current),
                attack_flows_before=int((current["label"] != NORMAL_LABEL).sum()),
                attack_flows_after=int((after_window["label"] != NORMAL_LABEL).sum()),
                risk_before=before.risk, risk_after=after.risk,
                cv_before=before.features["cv"], cv_after=after.features["cv"],
                rate_before=before.features["rate"], rate_after=after.features["rate"],
                resolved=resolved,
            ))
            current, before = after_window, after
            if resolved:
                break
        plan.final_window = current
        plan.resolved = plan.steps[-1].resolved
        return plan
