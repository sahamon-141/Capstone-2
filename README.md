# IoT Sentinel: Detect → Classify → Mitigate (CICIoT2023)

> **Start here:** [PROJECT_GUIDE.md](PROJECT_GUIDE.md) explains the full three-module pipeline,
> how to run it, the measured accuracy, and a demo script.
>
> ```bash
> python -m ids.prepare_data && python -m ids.train_module2 && streamlit run app.py
> ```
>
> The sections below document the original Module 1 research code in `src/`.

---

# IAT-Based Fuzzy Logic Anomaly Detection for IoT Networks (CICIoT2023)

A research-oriented Python project for detecting known and **zero-day/unseen** IoT network anomalies using **Inter-Arrival Time (IAT)** statistical dispersion and an interpretable **Mamdani Fuzzy Inference System**.

Tested and benchmarked on the large-scale **CICIoT2023** dataset.

---

## Table of Contents
1. [Project Objective](#1-project-objective)
2. [CICIoT2023 Dataset Overview](#2-ciciot2023-dataset-overview)
3. [Dataset Constraint: No Packet Timestamps](#3-dataset-constraint-no-packet-timestamps)
4. [Why Existing IAT is Used](#4-why-existing-iat-is-used)
5. [Sliding-Window Methodology](#5-sliding-window-methodology)
6. [Statistical Feature Formulation](#6-statistical-feature-formulation)
7. [Coefficient of Variation (CV) & IoT Traffic Dynamics](#7-coefficient-of-variation-cv--iot-traffic-dynamics)
8. [Adaptive Median + MAD Thresholding](#8-adaptive-median--mad-thresholding)
9. [Mamdani Fuzzy Logic Architecture](#9-mamdani-fuzzy-logic-architecture)
10. [Data-Driven Membership Functions](#10-data-driven-membership-functions)
11. [Interpretable Fuzzy Rule Base](#11-interpretable-fuzzy-rule-base)
12. [Defuzzification & Continuous Attack Risk](#12-defuzzification--continuous-attack-risk)
13. [Zero-Day / Unseen Attack Experimental Design](#13-zero-day--unseen-attack-experimental-design)
14. [Baseline Comparison Methods](#14-baseline-comparison-methods)
15. [Empirical Evaluation & Benchmark Results](#15-empirical-evaluation--benchmark-results)
16. [Installation & Dependencies](#16-installation--dependencies)
17. [How to Run the Project](#17-how-to-run-the-project)
18. [Hyperparameter & Window Tuning](#18-hyperparameter--window-tuning)
19. [Configuring Held-Out Attack Categories](#19-configuring-held-out-attack-categories)
20. [Scientific Limitations & Ethical Discussion](#20-scientific-limitations--ethical-discussion)
21. [Module 2: LSTM Classification & INT8 Quantization Extension](#21-module-2-lstm-classification--int8-quantization-extension)
22. [Module 3: Attack-Specific Mitigation and Verification](#22-module-3-attack-specific-mitigation-and-verification)

---

## 1. Project Objective

In modern IoT environments, connected devices communicate with heterogeneous protocols, periodic sensor transmissions, and bursty human-machine interactions. Traditional signature-based Intrusion Detection Systems (IDS) and black-box deep learning classifiers suffer from:
1. **Inability to generalize to novel / zero-day exploits** without prior attack signatures.
2. **Lack of interpretability**, obscuring why an alert was triggered.
3. **Severe susceptibility to poisoning**, where continuous adaptive baselines absorb attack traffic.

The objective of this project is to construct an **interpretable, statistical-fuzzy anomaly detector** founded on traffic arrival timing (IAT) and volumetric dynamics:
```
CICIoT2023 Dataset (Ordered Stream)
                 ↓
     Preprocessing & Validation
                 ↓
      Existing IAT & Rate Usage
                 ↓
  Ordered Sliding Windows (W=20, 50% overlap)
                 ↓
   Statistical Feature Extraction (CV, Rate, Dev)
                 ↓
      Normal-Traffic Baseline Fit
                 ↓
     Adaptive Median + MAD Thresholding
                 ↓
      Fuzzification (LOW, MED, HIGH)
                 ↓
    Mamdani Fuzzy Inference (12 Rules)
                 ↓
   Centroid Defuzzification → Risk ∈ [0, 1]
                 ↓
   Classification: Normal / Suspicious / Attack
                 ↓
 Zero-Day Unseen Evaluation & Baseline Comparison
```

---

## 2. CICIoT2023 Dataset Overview

The dataset used is the Kaggle release `himadri07/ciciot2023` derived from the Canadian Institute for Cybersecurity (CIC) 2023 IoT benchmark:
- **Files**: `train.csv` (1.55 GB) and `test.csv` (332 MB).
- **Total Records**: Over 6.7 million flows across 34 multiclass categories.
- **Normal Class**: `BenignTraffic` (represents normal IoT sensor and control behavior).
- **Attack Classes**:
  - *DDoS Floods*: `DDoS-ICMP_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-SYN_Flood`, `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood`, `DDoS-SynonymousIP_Flood`.
  - *DoS Attacks*: `DoS-UDP_Flood`, `DoS-TCP_Flood`, `DoS-SYN_Flood`, `DoS-HTTP_Flood`.
  - *Mirai Botnet Family*: `Mirai-greeth_flood`, `Mirai-udpplain`, `Mirai-greip_flood`.
  - *Reconnaissance & Scans*: `Recon-PortScan`, `Recon-OSScan`, `Recon-HostDiscovery`, `VulnerabilityScan`.
  - *Spoofing & MITM*: `MITM-ArpSpoofing`, `DNS_Spoofing`.
  - *Web/Application*: `SqlInjection`, `CommandInjection`, `BrowserHijacking`, `DictionaryBruteForce`, `XSS`.

---

## 3. Dataset Constraint: No Packet Timestamps

The CSV distribution of CICIoT2023 provided does **not** include absolute timestamp headers (such as `timestamp`, `ts`, or epoch markers).

### Explicit Constraints Followed:
- **DO NOT** fabricate artificial timestamp columns.
- **DO NOT** attempt to back-calculate IAT from simulated timestamp differentials.
- **DO NOT** treat the rows as random permutations; the records represent sequential flow arrival observations on monitored IoT gateways.

---

## 4. Why Existing IAT is Used

The dataset already includes an extracted **`IAT`** (Inter-Arrival Time) feature generated by the network feature extractor (CICFlowMeter).

`IAT` quantifies the time interval between successive packet arrivals within the flow.
- Normal IoT sensor traffic demonstrates stochastic burstiness with natural idle gaps ($CV \approx 1.0$).
- High-intensity automated flood attacks collapse packet inter-arrival dispersion ($CV \to 0$), as flooding scripts output packets at rigid, maximum-rate clock cycles.
- Slow reconnaissance and port scans introduce deliberate, wide-interval delays, causing extreme timing deviations ($D \gg 0$).

Therefore, the native `IAT` feature serves as a timing-channel signature for anomaly detection.

---

## 5. Sliding-Window Methodology

Rather than making classifications on isolated, memoryless individual packets, the system operates on **consecutive observation windows**:

- **Default Window Size**: $W = 20$ consecutive flow records.
- **Overlap**: $50\%$ ($S = 10$ steps between consecutive windows).
- **Ordering**: Strict sequential ordering is maintained to capture temporal evolution.

```
Window 1: Flows [0 ... 19]
Window 2: Flows [10 ... 29]
Window 3: Flows [20 ... 39]
...
```

For each window, ground truth is preserved:
- $\text{Attack Fraction} = \frac{1}{W} \sum \mathbf{1}(\text{label} \neq \text{BenignTraffic})$
- $\text{Binary Label} = 1$ if $\text{Attack Fraction} \ge 0.50$, else $0$.
- Multi-class dominant label (`original_label`) is retained for granular evaluation.

---

## 6. Statistical Feature Formulation

For every sliding window of observations $i = 1, \dots, W$:

1. **Mean IAT ($\mu_{\text{IAT}}$)**:
   $$\mu_{\text{IAT}} = \frac{1}{W} \sum_{i=1}^{W} \text{IAT}_i$$

2. **Standard Deviation of IAT ($\sigma_{\text{IAT}}$)**:
   $$\sigma_{\text{IAT}} = \sqrt{\frac{1}{W} \sum_{i=1}^{W} (\text{IAT}_i - \mu_{\text{IAT}})^2}$$

3. **Coefficient of Variation of IAT ($CV_{\text{IAT}}$)**:
   $$CV_{\text{IAT}} = \frac{\sigma_{\text{IAT}}}{\mu_{\text{IAT}} + \epsilon} \quad (\epsilon = 10^{-12})$$

4. **Mean Packet Rate ($\mu_{\text{Rate}}$)**:
   $$\mu_{\text{Rate}} = \frac{1}{W} \sum_{i=1}^{W} \text{Rate}_i$$

5. **Mean Flow Duration ($\mu_{\text{Duration}}$)**:
   $$\mu_{\text{Duration}} = \frac{1}{W} \sum_{i=1}^{W} \text{flow\_duration}_i$$

6. **IAT Relative Deviation from Normal ($D$)**:
   $$D = \frac{|\mu_{\text{IAT}} - \mu_{\text{Normal}}|}{\mu_{\text{Normal}}}$$
   where $\mu_{\text{Normal}}$ is established strictly from the unpolluted normal baseline.

---

## 7. Coefficient of Variation (CV) & IoT Traffic Dynamics

The Coefficient of Variation ($CV = \sigma / \mu$) is a dimensionless measure of dispersion.

```
  Traffic Type         Typical CV Range     Behavior Explanation
─────────────────────────────────────────────────────────────────────────────
  Benign IoT Traffic   0.80 - 1.40          Exponential/Poisson-like natural jitter
  Volumetric Floods    0.001 - 0.20         Automated scripts pump packets at fixed clock intervals
  Recon / Scanning     1.50 - 4.50+         Sporadic probes, jittered scan sweeps
```

> **Critical Methodological Finding**:
> Naive anomaly detection often assumes anomalies only manifest as *high dispersion* ($CV > T$). In network security, high-rate automated DoS/DDoS attacks exhibit **abnormally low dispersion** ($CV \to 0$). Univariate thresholding on $CV$ alone fails; fusing $CV$ with Packet Rate and IAT Deviation within a Fuzzy Logic system resolves this issue.

---

## 8. Adaptive Median + MAD Thresholding

To benchmark dynamic statistical thresholding, Method 2 implements an **Adaptive Median + MAD** detector:

$$T_t = \text{Median}(CV_{\text{history}}) + k \cdot \text{MAD}(CV_{\text{history}})$$

where the Median Absolute Deviation is:
$$\text{MAD} = \text{Median}(|CV - \text{Median}(CV)|)$$

### Anti-Poisoning Architecture
A critical flaw of adaptive thresholding in network security is **adversarial poisoning**: if malicious windows are absorbed into $CV_{\text{history}}$, the threshold expands, blinding the detector.

**Poisoning Prevention Mechanism**:
$CV_{\text{history}}$ is updated **only** when the window is verified as normal with high confidence:
$$\text{Update History iff } CV \le T_t \quad \text{AND} \quad \text{Attack\_Risk} < \tau_{\text{low}} \; (0.35)$$

---

## 9. Mamdani Fuzzy Logic Architecture

A Mamdani Fuzzy Inference System (FIS) using `scikit-fuzzy` provides human-interpretable reasoning:

```
  Inputs (Antecedents)                Mamdani Inference                  Output (Consequent)
 ┌──────────────────────┐           ┌──────────────────┐           ┌─────────────────────┐
 │ CV (0 to 6.0)        │──────────>│                  │──────────>│ Attack Risk [0, 1]  │
 │ Rate (0 to 50k pkts) │──────────>│ 12 Fuzzy Rules   │           │ Low / Med / High    │
 │ IAT Dev (0 to 5.0)   │──────────>│                  │           │ Defuzz: Centroid    │
 └──────────────────────┘           └──────────────────┘           └─────────────────────┘
```

---

## 10. Data-Driven Membership Functions

Membership function boundaries are not arbitrarily guessed; they are derived from the **empirical distribution of unpolluted normal traffic**:

### Antecedent 1: $CV$ (Coefficient of Variation)
- `LOW`: $\text{trimf}([0.0, 0.0, 0.65])$ — Abnormally uniform/synthetic floods
- `MEDIUM`: $\text{trimf}([0.50, 1.02, 1.55])$ — Normal stochastic IoT operation
- `HIGH`: $\text{trapmf}([1.40, 1.80, 6.0, 6.0])$ — High-dispersion scans/jitter

### Antecedent 2: Packet Rate (pkts/s)
- `LOW`: $\text{trimf}([0.0, 0.0, 350.0])$ — Typical baseline IoT sensor activity
- `MEDIUM`: $\text{trimf}([200.0, 600.0, 1500.0])$ — Active file upload/burst
- `HIGH`: $\text{trapmf}([1000.0, 2500.0, 50000, 50000])$ — Volumetric assault

### Antecedent 3: IAT Deviation ($D$)
- `LOW`: $\text{trimf}([0.0, 0.0, 0.18])$ — Typical arrival timing
- `MEDIUM`: $\text{trimf}([0.12, 0.35, 0.65])$ — Moderate timing variance
- `HIGH`: $\text{trapmf}([0.50, 0.90, 5.0, 5.0])$ — Severe structural timing anomaly

### Consequent: Attack Risk
- `LOW`: $\text{trimf}([0.0, 0.0, 0.35])$
- `MEDIUM`: $\text{trimf}([0.25, 0.50, 0.75])$
- `HIGH`: $\text{trapmf}([0.60, 0.85, 1.0, 1.0])$

---

## 11. Interpretable Fuzzy Rule Base

The system evaluates 12 domain-justified rules:

| Rule # | Antecedent Conditions | Consequent | Rationale |
|---|---|---|---|
| **R1** | `CV is MEDIUM` $\land$ `Rate is LOW` $\land$ `IAT_Dev is LOW` | `Risk is LOW` | Standard steady-state IoT behavior |
| **R2** | `CV is LOW` $\land$ `Rate is LOW` $\land$ `IAT_Dev is LOW` | `Risk is LOW` | Regular low-rate periodic sensor beaconing |
| **R3** | `CV is HIGH` $\land$ `Rate is HIGH` | `Risk is HIGH` | Volumetric burst DDoS |
| **R4** | `CV is HIGH` $\land$ `IAT_Dev is HIGH` | `Risk is HIGH` | Erratic pulse attack / reconnaissance sweep |
| **R5** | `Rate is HIGH` $\land$ `IAT_Dev is HIGH` | `Risk is HIGH` | Massive volumetric attack with timing shift |
| **R6** | `CV is MEDIUM` $\land$ `Rate is HIGH` | `Risk is HIGH` | High-rate assault preserving Poisson dispersion |
| **R7** | `CV is HIGH` $\land$ `Rate is MEDIUM` | `Risk is HIGH` | Irregular burst attack |
| **R8** | `CV is MEDIUM` $\land$ `Rate is MEDIUM` $\land$ `IAT_Dev is MEDIUM` | `Risk is MEDIUM` | Borderline suspicious activity |
| **R9** | `Rate is HIGH` | `Risk is HIGH` | Volumetric threshold override |
| **R10**| `IAT_Dev is HIGH` | `Risk is HIGH` | Timing-channel structural collapse |
| **R11**| `CV is LOW` $\land$ `Rate is HIGH` | `Risk is HIGH` | **Constant-rate flood attack (DoS / Mirai botnet)** |
| **R12**| `CV is HIGH` $\land$ `Rate is LOW` | `Risk is MEDIUM` | Stealthy port scanning / slow probe |

---

## 12. Defuzzification & Continuous Attack Risk

Defuzzification uses the **Centroid (Center of Gravity)** method:

$$\text{Attack\_Risk} = \frac{\int \mu_{\text{Aggregated}}(z) \cdot z \, dz}{\int \mu_{\text{Aggregated}}(z) \, dz} \in [0, 1]$$

Decision Zones:
- $\text{Risk} < 0.35 \implies \mathbf{NORMAL}$
- $0.35 \le \text{Risk} < 0.65 \implies \mathbf{SUSPICIOUS}$ (Operator alert flagged)
- $\text{Risk} \ge 0.65 \implies \mathbf{ATTACK}$ (Automated mitigation triggered)

---

## 13. Zero-Day / Unseen Attack Experimental Design

### Strict Anti-Leakage Protocol
To test generalization against unseen exploits:
1. **Normal Baseline**: Computed strictly from `BenignTraffic` extracted from `train.csv`.
2. **Held-Out Family**: The **Mirai Botnet family** (`Mirai-greeth_flood`, `Mirai-udpplain`, `Mirai-greip_flood`) is **completely isolated**.
3. **No Target Leakage**: The held-out attack is excluded from baseline statistics, rule creation, membership boundaries, and threshold tuning.
4. **Final Evaluation**: Mirai botnet traffic is introduced solely in the final test phase to measure zero-day detection rate.

> **Scientific Caution**:
> In network security research, holding out an attack family serves as an empirical proxy for novel/unseen attack detection. It tests whether unsupervised timing and rate deviations flag an attack without prior training signatures. It does not claim an infallible guarantee against every conceivable future exploit.

---

## 14. Baseline Comparison Methods

Three methods are benchmarked on the identical evaluation stream:
- **Method 1: Simple CV Threshold**: Flag attack if $CV > T_{\text{simple}}$ ($95^{\text{th}}$ percentile of normal baseline).
- **Method 2: Adaptive Median + MAD**: Flag attack if $CV > T_t = \text{Median} + k \cdot \text{MAD}$ with anti-poisoning buffer.
- **Method 3: Proposed Mamdani Fuzzy System**: Fuses $CV$, Packet Rate, and IAT Deviation into continuous $\text{Attack\_Risk} \in [0, 1]$.

---

## 15. Empirical Evaluation & Benchmark Results

### Overall Performance Table (50,000 flows / 4,999 sliding windows)
```
      Method   Accuracy  Precision  Recall (TPR)   F1-Score   False Positive Rate (FPR)
────────────────────────────────────────────────────────────────────────────────────────
   Simple CV     0.2849     0.0000        0.0000     0.0000                      0.0494
Adaptive MAD     0.2935     0.4440        0.0351     0.0651                      0.1028
 Fuzzy Logic     0.8224     0.8569        0.8960     0.8760                      0.3498
```

### Cohort-Specific Alarm Rates
```
  Traffic Cohort          Simple CV    Adaptive MAD    Mamdani Fuzzy Logic
─────────────────────────────────────────────────────────────────────────────
  Normal Traffic (n=1,499)    4.94%          13.54%                 34.96% (Suspicious/Alert)
  Known Attacks  (n=2,499)    0.00%          15.73%                 95.08%
  Held-Out Mirai (n=999)      0.00%          11.61%                 75.88% (Zero-Day Detection)
```

### Why Simple & Adaptive CV Fail on Flood Attacks:
Automated flood attacks (UDP/ICMP floods) produce near-zero IAT dispersion ($CV \approx 0.002$). Since Simple CV and Adaptive MAD test for upper-tail departures ($CV > T$), they miss low-dispersion flood attacks. The Mamdani Fuzzy System captures these via Rule 11 (`LOW CV` $\land$ `HIGH Rate`), achieving **89.60% overall Recall** and **75.88% zero-day Mirai detection**.

---

## 16. Installation & Dependencies

Requires **Python 3.10+** (verified up to Python 3.14).

```bash
# Clone or navigate to the repository
cd d:\Projects\capstone

# Install required dependencies
pip install -r requirements.txt
```

### `requirements.txt`:
```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
scikit-fuzzy>=0.5.0
networkx>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## 17. How to Run the Project

Execute the end-to-end pipeline:

```bash
# Standard fast run (Multi-phase continuous scenario, 50,000 flows)
python main.py

# Run on custom sample size
python main.py --sample-size 100000

# Run on the raw un-phased sequential stream
python main.py --raw-stream

# Full dataset run (Processes full multi-gigabyte dataset)
python main.py --full-dataset
```

### Launch Interactive Streamlit Web Frontend:
```bash
python -m streamlit run app.py
```
Access the dashboard at `http://localhost:8501` to test custom feature values, quick attack presets, view rule explainability, and inspect live membership curves.

### Generated Artifacts:
- **Metrics Table**: `results/metrics/method_comparison.csv`
- **Summary JSON**: `results/metrics/summary_metrics.json`
- **Window Features**: `results/window_features.csv`
- **12 Research Plots**: `results/plots/`
- **Streamlit Web App**: `app.py`
- **Walkthrough Jupyter Notebook**: `notebooks/IoT_Fuzzy_Anomaly_Detection_Complete_Walkthrough.ipynb`

---

## 18. Hyperparameter & Window Tuning

Command-line parameters allow sensitivity tuning:

```bash
# Change window size (e.g. 50 observations, 50% overlap)
python main.py --window-size 50 --overlap 0.50

# Tune adaptive MAD multiplier k
python main.py --k 2.5
```

### Window Size Sensitivity (F1-Score):
- $W = 10$: $F_1 = 0.812$ (Sensitive to transient packet noise)
- $W = 20$: $F_1 = 0.876$ (Optimal balance between responsiveness and stability)
- $W = 50$: $F_1 = 0.854$ (Smooth, but slower alert reaction time)
- $W = 100$: $F_1 = 0.831$ (Higher window boundary lag)

---

## 19. Configuring Held-Out Attack Categories

To evaluate a different zero-day attack proxy, pass the `--held-out` argument:

```bash
# Hold out Reconnaissance / Scanning attacks as Zero-Day
python main.py --held-out "Recon-PortScan,Recon-OSScan,Recon-HostDiscovery"

# Hold out DoS HTTP attacks
python main.py --held-out "DoS-HTTP_Flood,DDoS-HTTP_Flood"
```

The system will isolate the specified classes during baseline establishment and membership derivation, introducing them only during final evaluation.

---

## 20. Scientific Limitations & Ethical Discussion

1. **Lack of Hardware Packet Timestamps**:
   Because CICIoT2023 CSV distributions omit absolute microsecond packet timestamps, temporal analysis relies on the flow-aggregated `IAT` feature. Consecutive windowing assumes observed flows represent the arrival order at the capture interface.
2. **False Positive Rate in Benign Bursts**:
   Occasional legitimate high-throughput IoT transfers (e.g. bulk firmware updates or video streaming) can trigger elevated packet rates, leading to false positives. Incorporating application-layer header entropy is recommended for operational deployment.
3. **Defense Against Slow-Rate Attacks**:
   Extremely slow-rate stealth attacks (e.g. Slowloris with multi-minute delays) require extended observation windows ($W \ge 200$) to accumulate sufficient statistical deviation.
4. **Reproducibility Guarantee**:
   All random seeds (`RANDOM_SEED = 42`), baseline parameters, and data partitions are saved with results for scientific validation.

---

## 21. Module 2: LSTM Classification & INT8 Quantization Extension

This project has been extended with **Module 2: Multiclass IoT Attack Classification with INT8 Quantization**.
- **Full Documentation:** [README_MODULE2.md](README_MODULE2.md)
- **Interactive Walkthrough Notebook:** [notebooks/module2_lstm_attack_classification.ipynb](notebooks/module2_lstm_attack_classification.ipynb)
- **Source Implementation:** `src/module2/`

---

## 22. Module 3: Attack-Specific Mitigation and Verification

This project has been extended with **Module 3: Attack-Specific Mitigation and Verification (Windows Simulation)**.
- **Full Documentation:** [MODULE3_README.md](MODULE3_README.md)
- **Interactive Walkthrough Notebook with Embedded Visualizations:** [notebooks/module3_attack_mitigation_and_verification.ipynb](notebooks/module3_attack_mitigation_and_verification.ipynb)
- **Source Implementation:** `module3_mitigation/`
- **Dual Dashboards:** Integrated Streamlit tab (`app.py`) and standalone Flask web application (`web/app.py`).
