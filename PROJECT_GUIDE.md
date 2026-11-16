# IoT Sentinel: Project Guide (for presentation)

A three-module intrusion detection and response system for IoT networks, built and evaluated on
**CICIoT2023**.

| Module | Question it answers | How |
|---|---|---|
| **1. Detection** | *Is this traffic window anomalous?* | Packet inter-arrival time (IAT) statistics + behaviour deviation → Mamdani fuzzy logic → risk 0–1 |
| **2. Classification** | *Which attack is it, or is it unknown?* | Random Forest (accurate) or INT8-quantised MLP (edge) + nearest-neighbour zero-day check |
| **3. Mitigation** | *What do we do, and did it work?* | Class → 1 of 10 policy families, or unknown → ladder chosen from the traffic shape → auto / analyst approval → simulated controller → re-run Module 1 to verify |

---

## 1. How to run it

```bash
pip install -r requirements.txt
python -m ids.prepare_data       # ~2 min: samples CICIoT2023 into data/processed/*.parquet
python -m ids.train_module2      # ~4 min: trains Module 2, calibrates Module 1, writes metrics
python -m ids.evaluate_module3   # ~3 min: evaluates Module 3 over all 33 attack classes
streamlit run app.py             # opens http://localhost:8501
python -m pytest tests/test_pipeline.py -q   # 8 end-to-end tests
```

(The raw dataset must be at `data/ciciot2023/train.csv` and `test.csv`.)

---

## 2. How the project works, start to end

### Step 0: Data (`ids/prepare_data.py`)
* CICIoT2023 contains traffic captured from 105 real IoT devices. Each CSV row is a **flow** with 46 numeric
  features (rate, TCP flags, packet sizes, protocol flags, **IAT**, …) and one of **34 labels**:
  benign + 33 attacks in 7 families (DDoS, DoS, Mirai, Recon, Spoofing, Web, BruteForce).
* `train.csv` is 1.6 GB, so it is streamed in chunks and sampled with a cap per class.
  This gives **153,135 training flows** (from `train.csv`) and **37,187 test flows** (from the separate `test.csv`).
* **Zero-day proxy:** all three Mirai classes are removed from Module 2's training. Module 2 only ever
  sees them at test time, which shows how the system handles an attack it has never seen.

### Step 1: Traffic window
Traffic is analysed in **sliding windows of 20 flows**. In the demo, a window is built from real test
flows: e.g. 80% attack flows plus 20% normal background traffic, as on a real network.

### Step 2: Module 1, anomaly detection (`ids/module1_detection.py`)
For each window:

| Feature | Formula | Why |
|---|---|---|
| Mean IAT μ | mean of the 20 IATs | typical gap between packets |
| Std IAT σ | standard deviation | how irregular the gaps are |
| **CV** | σ / μ | normal ≈ 1.0. **Floods → CV ≈ 0.001** (machine-regular). Scans → high |
| Packet rate R | **median** rate of the 20 flows | volume. The median stops one extreme flow from causing an alarm |
| IAT deviation D | \|μ − μ_normal\| / μ_normal | timing shift from the benign baseline |
| **Behaviour deviation B** (new) | median distance of the flows to the 5 nearest *benign* training flows ÷ threshold | catches attacks whose timing looks normal |

* B is learned from **benign traffic only**, so no attack labels are involved and Module 1 stays
  unsupervised. B = 1 is the 99th percentile of normal windows.
* **Fuzzy inference (Mamdani):**
  1. *Fuzzification*: each input gets LOW / MEDIUM / HIGH membership degrees (triangular/trapezoidal functions).
  2. *Rules*: 15 IF-THEN rules, e.g. *R12: IF CV is LOW AND rate is HIGH THEN risk HIGH* (constant-rate flood),
     *R14: IF B is HIGH THEN risk HIGH*. Rule strength = min of its IF-parts.
  3. *Aggregation*: each rule's output set is clipped at its strength, and the sets are merged with max.
  4. *Defuzzification*: the **centroid** gives one risk score in [0, 1].
* **Decision:** risk < 0.35 **NORMAL** (Modules 2 and 3 stay idle, saving compute) · 0.35–0.65 **SUSPICIOUS** ·
  ≥ 0.65 **ATTACK**. Suspicious and attack windows go to Module 2.
* The membership functions and the 13 timing rules are the same as the original `src/fuzzy_system.py`
  (a test checks that both give the same results). The sidebar option **"IAT timing only"** switches back
  to the original 3-input design so you can show the improvement live.

### Step 3: Module 2, attack classification (`ids/module2_classification.py`)
* **Random Forest** (60 trees, depth 20) on the 46 flow features. It classifies every flow in the window,
  and the window gets the **majority attack label**. Benign background flows are ignored when attacks make
  up at least 30% of the window.
* **INT8 edge model:** an MLP 46 → 128 → 64 → 31. After training it is **post-training quantised**:
  * inputs: per-feature affine INT8, `q = clip(round(r/s) + z, −128, 127)` with `s = (max − min)/255`
  * weights: symmetric per-layer INT8; activations: INT8 with calibrated scales
  * matrix products run as int8 × int8 → int32, as on a microcontroller
* **Zero-day check (open-set):** the median distance of the window's attack flows to the nearest *known*
  training flow is divided by a threshold calibrated on validation windows (97.5th percentile).
  A score > 1 labels the window **"Unknown (zero-day)"** instead of forcing it into a known class.
* If Module 2 finds only benign traffic, it overrules a Module 1 false alarm and nothing is mitigated.

### Step 4: Module 3, mitigation & verification (`ids/module3_mitigation.py`)
Module 3 merges the two earlier versions: the policy table and zero-day router of `module3_mitigation/`
with the flow-level verification of `ids/`. There is now only one Module 3.

1. **Routing.**
   * **Known attack:** the class Module 2 predicted is looked up in `module3_mitigation/policies.json`, which maps
     all 31 known classes to **10 policy families**, grouped by *how the attack is stopped*:

     | Policy family | Classes | Escalation ladder (least disruptive first) |
     |---|---|---|
     | Volumetric flood | UDP / ICMP / TCP / SynonymousIP floods | rate limit → block → upstream blackhole |
     | TCP state exhaustion | SYN, PSH-ACK, RST-FIN floods | SYN cookies → rate limit → block |
     | Fragmentation | ICMP / UDP / ACK fragmentation | drop fragments → rate limit → block |
     | Slow application | SlowLoris, HTTP floods | limit connections → block |
     | Reconnaissance | port/OS scan, host discovery, ping sweep, vuln. scan | block scanner → isolate segment |
     | Spoofing / MITM | ARP spoofing, DNS spoofing | ARP/DNS integrity → isolate segment |
     | Web application | SQLi, XSS, command injection, browser hijacking, upload | WAF rule → block |
     | Credential | dictionary brute force | throttle logins → block |
     | Malware / backdoor | backdoor malware | quarantine endpoint (admin only) |
     | Botnet | (only Mirai, which is held out, so unused) | rate limit → isolate device |

     Each policy also has an explanation, long-term recommendations, a class-specific tip
     (e.g. SlowLoris → shorter header timeout) and example Linux commands.
   * **Unknown (zero-day):** there is no class to look up, so the ladder is chosen from the **shape** of the flows
     Module 2 marked as attack. CV < 0.05 (or low CV at an above-normal rate) means a **constant-rate flood**
     (rate limit → block → blackhole). High CV at a low rate means a **slow probe** (block scanner → isolate).
     High rate with shifted timing means a **volumetric surge**. Anything else is an **unclassified anomaly**
     (rate limit → block → isolate).
2. **Auto vs. analyst approval:** the first action runs automatically only if the attack is known, Module 2's
   confidence reaches the policy's threshold (60–75%), the policy does not demand an administrator, and the
   action only *filters / limits*. **Blocking or isolating always needs an analyst.** A zero-day window may start
   with a reversible rate limit automatically, but a blocking stage waits for an analyst.
3. **Simulated controller:** the action filters the flows **Module 2 marked as attacks** (not the ground-truth
   labels), each with the action's assumed efficacy (filters 75–85%, blocks 95–98%). Normal traffic keeps flowing.
4. **Verification:** **Module 1 is re-run** on the post-mitigation traffic. If the risk is < 0.35, the attack is
   resolved and CV returns to ≈ 1.0, which is the paper's idea of "the IAT deviation returning to normal".
   Otherwise the next, stronger stage is applied.

> The controller is a simulation; nothing touches a real firewall. On a gateway, the same plan would call the
> listed commands.

---

## 3. Results (measured, from `results/pipeline/metrics.json`)

### Module 1: detection (1,360 test windows: 40 per class)
| Version | Recall | Precision | F1 | Balanced acc. | False-positive rate | Mirai (zero-day) caught |
|---|---|---|---|---|---|---|
| IAT timing only (original 3 inputs) | 15.9% | 99.1% | 27.4% | 55.5% | 5.0% | 33.3% |
| **IAT timing + behaviour B (final)** | **99.8%** | **99.8%** | **99.8%** | **96.1%** | **7.5%** | **100%** |

* Every attack family is detected in ≥ 98.5% of windows (Recon-OSScan is the lowest at 92.5%).
* Timing alone fails here because mixing in 20% normal background flows breaks the flood's perfectly
  regular timing. B detects the attack flows themselves.
* Note on the earlier paper numbers (82.2% accuracy, 89.6% recall): they came from a different, phased
  stream of consecutive rows, so they are not directly comparable. They are kept under
  *Model Performance → Earlier experiments*.

### Module 2: classification (31 known classes: benign + 30 attacks, 32,687 test flows)
| Model | Accuracy | Macro-F1 | Family accuracy | Size | Latency |
|---|---|---|---|---|---|
| **Random Forest (FP32)** | **90.9%** | 79.3% | **93.4%** | 118 MB | ≈ 15 µs / flow (batch) |
| MLP (FP32) | 78.0% | 63.2% | 83.6% | 63.6 KB | |
| **MLP (INT8, edge)** | 77.8% | 63.2% | 83.5% | **16.2 KB** | ≈ 21 µs / flow |

* INT8 makes the model **74.6% smaller** and agrees with FP32 on **98.5%** of predictions (accuracy drops only 0.1 pt).
* **Window level** (20 flows, 70% attack + 30% benign): **94.6% class accuracy, 95.4% family accuracy**.
* All flood classes have F1 ≥ 0.97. The weak classes are rare web attacks (SqlInjection 0.37, XSS 0.46,
  Uploading 0.32) and the Recon sub-types. They have very few samples and look alike at flow level.
  Their mistakes usually stay inside the right family, which is what selects the Module 3 playbook.

### Zero-day (Mirai, never trained)
| Check | Result |
|---|---|
| Module 1 flags Mirai windows | **100%** |
| Module 2 labels them "Unknown (zero-day)" | **100%** |
| Known attacks wrongly called "Unknown" | 4.4% |

In the earlier version, the LSTM with softmax-confidence rejection detected **0%** of Mirai and labelled
it as UDP floods. The nearest-neighbour open-set check fixes this.

### Module 3: mitigation (`python -m ids.evaluate_module3`, driven by Module 2's predictions)
645 windows: 15 for each of the 33 attack classes (80% attack + 20% benign flows) and 150 benign windows.

| Measure | Result |
|---|---|
| Windows reaching Module 3 | 491 |
| Known attacks routed to the correct policy family | **94.4%** |
| Mitigations verified (risk back below 0.35) | **95.7%** (85% with a single action) |
| Risk before → after | 0.73 → 0.17 (CV back to 0.99, normal ≈ 1.0) |
| Attack volume removed by the first action | 82.3% |
| Plans that could start automatically | 62.1% |
| Plans that needed a blocking action | 32.8% |
| Mirai windows: zero-day route / "constant-rate flood" shape / resolved | 100% / 82% / 97.8% |

Per family, the filtering policies (volumetric flood, TCP state, fragmentation, slow application) start
automatically and resolve 94–100% of their windows. Reconnaissance, spoofing, web, credential and malware
need an analyst because their first action blocks traffic, or because Module 2 is less confident on those classes.

### End-to-end ("Mixed day" stream, 10 seeds × 39 windows = 390 windows)
| Measure | Result |
|---|---|
| Attack windows detected by Module 1 | 158 / 160 (98.75%) |
| False alarms by Module 1 | 18 / 230 (7.8%); Module 2 cleared 13 of them |
| Windows that skipped Modules 2 and 3 | 54.9% |
| Mitigations verified | 159 / 163 (97.6%): 143 at stage 1, 16 at stage 2 |
| Route | 74.9% class policy, 25.1% zero-day |

---

## 4. What changed compared with the previous version

| Area | Before | Now |
|---|---|---|
| Structure | Module 1 in `src/`, LSTM in `src/module2/`, a generic Wi-Fi-oriented `module3_mitigation/`, one large unfocused `app.py` | One clean package `ids/` with one file per module + `pipeline.py` that chains them |
| Module 1 | 3 inputs (CV, mean rate, D). Mean rate was easily skewed by one extreme flow | Same fuzzy design + **behaviour deviation B** (unsupervised, benign-only) + robust median rate. Recall 15.9% → **99.8%** on mixed windows |
| Module 2 | LSTM 34.5% accuracy; RF 94% on only 87 test sequences | RF **90.9% / 94.6% (window)** on 32.7k test flows; new **INT8 MLP** (16 KB) with real integer inference |
| Zero-day | Softmax rejection: **0%** of Mirai detected | Nearest-neighbour open-set check: **100%** of Mirai detected, 4.4% false unknowns |
| Module 3 | Two separate versions: Wi-Fi-oriented policies with random "traffic reduction" numbers (later rebuilt for CICIoT2023 but run only on the old pipeline), and a simple 8-playbook version in the app | **One merged Module 3**: 10 CICIoT2023 policy families + traffic-shape zero-day routing, auto/approval logic, escalation, **verification by re-running the full Module 1**. 95.7% verified across all 33 attack classes |
| Frontend | One long page mixing everything | 5 pages: Overview, **Live Attack Lab** (4 numbered steps), Network Monitor, Model Performance, How It Works |
| Tests | Module 2/3 unit tests | + `tests/test_pipeline.py` end-to-end (fuzzy matches skfuzzy, benign stays idle, flood → classified → mitigated, Mirai → unknown, mitigation does not use ground truth) |

The old frontend is kept in `legacy/app_v1.py` and the Flask UI in `legacy/web_flask/` for reference.
The old `src/` code is untouched. `module3_mitigation/` is still the source of the policy table
(`policies.json`) and the zero-day router (`zero_day_containment.py`), which `ids/module3_mitigation.py` imports.
Its other parts (SQLite history, Wi-Fi scenarios, `run_module3.py`, `main.py` step 14) belong to the old
`src/` pipeline and are not used by the app.

---

## 5. Suggested demo script (≈ 10 minutes)

1. **Overview page**: explain the diagram: Detect → Classify → Mitigate → Verify loop.
2. **Live Attack Lab → `DDoS · DDoS-SYN_Flood`**: walk through the 4 steps:
   * Step 1: the window. Point at the IAT bars.
   * Step 2: CV ≈ 0.5 or lower, B ≫ 1, the fired rules, centroid risk ≈ 0.85 → ATTACK.
   * Step 3: RF predicts DDoS-SYN_Flood, top-5 probabilities, per-flow votes, zero-day score ≈ 0.
   * Step 4: CRITICAL playbook, SYN-cookie tip, automatic response. Click **Apply**, show the risk falling
     below 0.35 and CV returning to ≈ 1.
3. **`Benign · BenignTraffic`**: risk ≈ 0.15 → NORMAL. Modules 2 and 3 stay idle (saves compute).
4. **`Mirai · Mirai-greeth_flood`** (zero-day): Module 1 flags it, Module 2 reports **Unknown**,
   and Module 3 takes the **zero-day route**: it shows the traffic shape (usually "constant-rate flood") and starts
   with a reversible rate limit; blocking would need an analyst.
5. **`Web · SqlInjection`**: low confidence → analyst approval. Explain why the system is careful here.
6. Sidebar: switch Module 1 to **"IAT timing only"** on the same window to show why behaviour B was added.
   Switch Module 2 to **INT8** to show the edge model (≈ 1 ms per window).
7. **Network Monitor → Run stream**: the timeline and event log.
8. **Model Performance**: the numbers above.

## 6. Honest limitations (good to say before you are asked)
* The CSV has no raw packet timestamps; the dataset's `IAT` feature is used as given.
* Windows are assembled from sampled test flows because the Kaggle CSV rows are shuffled.
* The mitigation controller is simulated (efficacy values are assumptions, not measurements).
* Rare web-attack classes have little data, so their per-class F1 is low (family-level detection is still 100%).
* Module 1's false-positive rate is 7.5%. Module 2 overrules most of these, so they rarely trigger mitigation.
* The Random Forest file is large (118 MB). The INT8 MLP (16 KB) is the model meant for constrained devices.

## 7. File map
```
ids/config.py                 paths, class → family taxonomy, thresholds
ids/prepare_data.py           Step 0: sample CICIoT2023
ids/module1_detection.py      Module 1: IAT features, behaviour B, NumPy Mamdani fuzzy engine
ids/module2_classification.py Module 2: RF wrapper, INT8 quantiser + INT8 MLP, zero-day check
ids/module3_mitigation.py     Module 3: policy routing, zero-day shape routing, approval, simulated controller, verification
ids/evaluate_module3.py       Module 3 evaluation -> results/pipeline/module3_metrics.json
module3_mitigation/policies.json         the 10 policy families (used by ids/module3_mitigation.py)
module3_mitigation/zero_day_containment.py  the traffic-shape router (used by ids/module3_mitigation.py)
ids/pipeline.py               chains the modules; builds demo windows and streams
ids/train_module2.py          trains/evaluates everything, writes results/pipeline/
app.py                        Streamlit frontend (5 pages)
tests/test_pipeline.py        end-to-end tests
```
