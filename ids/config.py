"""Shared paths, class taxonomy and constants for the three-module pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "ciciot2023"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "pipeline"
METRICS_DIR = ROOT / "results" / "pipeline"

TRAIN_SAMPLE = PROCESSED_DIR / "train_sample.parquet"
TEST_SAMPLE = PROCESSED_DIR / "test_sample.parquet"

RANDOM_SEED = 42
NORMAL_LABEL = "BenignTraffic"

# Module 1 sliding window
WINDOW_SIZE = 20
RISK_SUSPICIOUS = 0.35   # risk >= this -> hand over to Module 2
RISK_ATTACK = 0.65       # risk >= this -> confirmed attack

# Zero-day proxy: the whole Mirai family is never shown to Module 2 during training.
HELD_OUT_CLASSES = ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"]

# Every CICIoT2023 class mapped to its attack family. Module 3 picks a playbook per family.
FAMILY_OF = {
    "BenignTraffic": "Benign",
    # DDoS
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS", "DDoS-SlowLoris": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    # Mirai botnet
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    # Reconnaissance
    "Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "Recon-PingSweep": "Recon", "VulnerabilityScan": "Recon",
    # Spoofing
    "MITM-ArpSpoofing": "Spoofing", "DNS_Spoofing": "Spoofing",
    # Web-based
    "SqlInjection": "Web", "CommandInjection": "Web", "XSS": "Web",
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "Uploading_Attack": "Web",
    # Brute force
    "DictionaryBruteForce": "BruteForce",
}

FAMILY_DESCRIPTION = {
    "Benign": "Normal IoT device traffic",
    "DDoS": "Distributed denial of service: many sources flood the target",
    "DoS": "Denial of service: a single source floods the target",
    "Mirai": "Mirai botnet floods (GRE / UDP) launched from infected IoT devices",
    "Recon": "Reconnaissance: scanning hosts, ports, OS fingerprints and vulnerabilities",
    "Spoofing": "Identity spoofing / man-in-the-middle (ARP, DNS)",
    "Web": "Web/application-layer attacks (SQLi, XSS, command injection, malware upload)",
    "BruteForce": "Password guessing against login services",
    "Unknown": "Anomalous traffic that matches no known attack (possible zero-day)",
}
