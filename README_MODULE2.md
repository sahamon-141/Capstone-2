# Module 2: LSTM-Based Multiclass IoT Attack Classification with INT8 Quantization

## 1. Module 2 Objective

While **Module 1** functions as a high-speed, unsupervised first-line filter that detects traffic timing and volumetric anomalies via Mamdani Fuzzy Logic without prior signatures, it only makes binary decisions (Normal vs Anomaly/Attack Risk).

The objective of **Module 2** is to provide granular **Multiclass Attack Attribution and Threat Classification** using an **LSTM (Long Short-Term Memory) Neural Network**, and subsequently quantize the trained model to **INT8 precision** via TensorFlow Lite / LiteRT for resource-constrained edge deployment on IoT gateways and single-board computers (such as a Raspberry Pi).

---

## 2. How Module 2 Complements Module 1

Module 1 and Module 2 operate as a two-stage hierarchical defense architecture:

```
                          Incoming IoT Network Flows
                                      │
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │       MODULE 1: Fuzzy Logic Anomaly Filter       │
             │   - Sliding Window (W=20, 50% overlap)           │
             │   - Inter-Arrival Time (IAT) Dispersion (CV)     │
             │   - Packet Rate + Baseline Timing Deviation      │
             └──────────────────────────────────────────────────┘
                                      │
                                      │  Attack Risk Score ∈ [0, 1]
                                      ▼
                         Is Traffic Anomalous (Risk ≥ 0.35)?
                                     / \
                           NO       /   \       YES
                          ┌────────┘     └────────┐
                          ▼                       ▼
                  Traffic Allowed           Trigger Module 2
                (Module 2 remains IDLE       (Retrieve Flow Slice
                to conserve IoT CPU)        Ending at Window)
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │    MODULE 2: LSTM Classifier    │
                                 │   - Sequence Length L=50        │
                                 │   - 19 Flow Behavioral Features │
                                 │   - FP32 Keras or INT8 TFLite   │
                                 └─────────────────────────────────┘
                                                  │
                                                  ▼
                                 Specific Attack Classification:
                                 - Specific Class (e.g. DoS-UDP)
                                 - Softmax Confidence
                                 - Zero-Day Rejection Status
                                 - Latency: < 0.1 ms (INT8)
```

1. **Efficiency:** In typical IoT operations, >95% of traffic is benign. Module 1 handles this steady-state traffic in microsecond timeframes. Heavy sequential neural network inference is only triggered when Module 1 flags an anomaly.
2. **Specialization:** Module 1 excels at timing-channel anomaly detection (especially low-dispersion flood attacks), while Module 2 analyzes deep sequential correlations across 19 volumetric and protocol features.

---

## 3. Architecture & Data Flow

```
Raw CSV / Live Observation Stream
              │
              ▼
   Stratified Block Generator
   (Boundary-Safe Extraction)
              │
              ▼
   Leakage-Safe Preprocessor
   (StandardScaler + LabelEncoder fitted on Train only)
              │
              ▼
    Sequence Construction
    (N, 50, 19) — Sequences NEVER cross scenario or class boundaries
              │
       ┌──────┴──────────────────────────┐
       ▼                                 ▼
FP32 LSTM Network             Non-Sequential Baseline
(Keras, LSTM(64) -> Dense)    (Pure-NumPy Random Forest)
       │
       ▼
Full-Integer INT8 Quantization
(TFLiteConverter + Calibration Data)
       │
       ▼
Edge-Ready TFLite Artifact
(0.31 MB, XNNPACK acceleration)
```

---

## 4. Dataset and Actual Class Labels

The benchmark dataset is **CICIoT2023** (`himadri07/ciciot2023` on Kaggle), comprising **6,668,822 network flows** across 34 multiclass categories.

### Class Taxonomy & Partitioning

* **Benign Traffic (Normal):**
  * `BenignTraffic` (Class ID 0)
* **Known Attack Classes (Development & Training):**
  * *DDoS Floods:* `DDoS-ICMP_Flood`, `DDoS-UDP_Flood`, `DDoS-TCP_Flood`, `DDoS-SYN_Flood`, `DDoS-PSHACK_Flood`, `DDoS-RSTFINFlood`, `DDoS-SynonymousIP_Flood`, `DDoS-HTTP_Flood`, `DDoS-SlowLoris`, `DDoS-ICMP_Fragmentation`, `DDoS-UDP_Fragmentation`, `DDoS-ACK_Fragmentation`
  * *DoS Attacks:* `DoS-UDP_Flood`, `DoS-TCP_Flood`, `DoS-SYN_Flood`, `DoS-HTTP_Flood`
  * *Reconnaissance & Scans:* `Recon-PortScan`, `Recon-OSScan`, `Recon-HostDiscovery`, `Recon-PingSweep`, `VulnerabilityScan`
  * *Spoofing & MITM:* `MITM-ArpSpoofing`, `DNS_Spoofing`
  * *Web & Application Attacks:* `DictionaryBruteForce`, `SqlInjection`, `CommandInjection`, `BrowserHijacking`, `XSS`, `Backdoor_Malware`, `Uploading_Attack`
* **Held-Out Zero-Day Attack Family (Quarantined):**
  * `Mirai-greeth_flood`, `Mirai-udpplain`, `Mirai-greip_flood` (The entire Mirai botnet family is quarantined from model training, scaler fitting, and threshold tuning).

---

## 5. Sequence Construction and Ordering Assumptions

### Provenance Audit Findings
1. **Absence of Absolute Timestamps:** Neither `train.csv` nor `test.csv` contains a `ts`, `timestamp`, or epoch column.
2. **Kaggle Random Shuffling:** In `train.csv` and `test.csv`, consecutive rows are intermixed with dozens of disparate attack types due to random shuffling prior to upload.
3. **Scientifically Defensible Sequence Solution:**
   * Sequences must **never** concatenate unrelated rows from arbitrary shuffle order.
   * In Module 2, sequences are constructed within **homogeneous capture/scenario blocks** of identical attack or benign types.
   * `SequenceBuilder` enforces:
     * Sequence length: $L = 50$ consecutive observations.
     * Step size: $S = 25$ (train) or $S = 15$ (val/test).
     * Sequences never cross class block boundaries.
     * Partitioning into Train (70%), Validation (15%), and Test (15%) occurs on the continuous flow blocks **before** sequences are formed, preventing overlapping sequence leakage.

---

## 6. Selected Input Features (19 Features)

Candidate features were selected from the 47 available in CICIoT2023 to provide discriminative timing, volumetric, and transport state information while strictly excluding labels and identifiers:

| Category | Features | Description |
| :--- | :--- | :--- |
| **Timing & Rates** | `IAT`, `Rate`, `Srate`, `Drate`, `flow_duration`, `Duration` | Inter-arrival timing dispersion and packet delivery rates |
| **Volume & Size** | `Header_Length`, `Tot size`, `Tot sum`, `Number`, `AVG`, `Std`, `Min`, `Max` | Packet dimensions, payload totals, and statistical dispersion |
| **Protocol & State**| `Protocol Type`, `TCP`, `UDP`, `syn_count`, `ack_count` | Transport layer protocols and connection handshake states |

*Excluded:* `label`, `binary_label`, `label_encoded`, `scenario_id`, IP addresses, and post-label tags.

---

## 7. Preprocessing & Leakage Prevention

Implemented in `src/module2/preprocessor.py`:
1. **Numerical Imputation:** Missing values and NaNs are imputed using feature medians computed **strictly on the training split**.
2. **Infinite Values:** Clamped to the finite maximum of the training distribution.
3. **Scaling:** `StandardScaler` is fitted on cleaned training flows and applied to validation, test, and live inference streams.
4. **Label Mapping:** Deterministic bidirectional mapping saved to `models/module2_label_map.json`.

---

## 8. LSTM Model Architecture

```
Input Sequence: (Batch, 50, 19)
       │
       ▼
LSTM(64 units, unroll=True, return_sequences=False)
       │
       ▼
Dropout(0.20)
       │
       ▼
Dense(32 units, activation="relu")
       │
       ▼
Dense(num_classes=31, activation="softmax")
```

*Note on `unroll=True`:* Static unrolling of the 50 time steps eliminates dynamic `TensorArray` operations (`tf.TensorListReserve`), enabling native TensorFlow Lite INT8 quantization without external runtime dependencies.

---

## 9. Random Forest Baseline

To determine whether sequential recurrent modeling provides measurable benefits over non-sequential methods, a **Random Forest Baseline** (`BaselineRandomForest`) was implemented in pure NumPy (ensuring complete compatibility and immunity to OS DLL-blocking policies).

The Random Forest processes:
* Final observation in sequence ($T-1$)
* Sequence statistical aggregates ($\mu, \sigma$ across time steps)

---

## 10. Empirical Benchmark Results

Measured on untouched test sequences:

| Metric | Random Forest Baseline | FP32 LSTM (Keras) | INT8 LSTM (TFLite) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **0.9425** | 0.3448 | 0.3448 |
| **Macro-F1** | **0.9177** | 0.2047 | 0.2173 |
| **Weighted-F1** | **0.9302** | 0.2118 | 0.2248 |
| **CPU Latency (Mean)**| 0.196 ms | 51.070 ms | **0.087 ms** |
| **CPU Latency (P95)** | 0.230 ms | 55.096 ms | **0.099 ms** |
| **Model Size** | 2.68 MB | 0.31 MB | **0.31 MB** |
| **Prediction Agreement**| — | **85.06% Agreement with INT8** | — |

### Scientific Insights
1. **Inference Acceleration:** INT8 TFLite inference via the XNNPACK CPU delegate achieved a **~587x speedup** over native FP32 Keras (0.087 ms vs 51.07 ms per sequence), making it viable for high-throughput edge gateways.
2. **Sequential vs Non-Sequential Performance:** The Random Forest baseline outperformed the compact LSTM on overall accuracy (0.9425 vs 0.3448) when trained on short sample budgets, demonstrating that volumetric aggregates strongly discriminate CICIoT2023 attack classes. The LSTM requires larger training epoch budgets to optimize temporal transitions across all 31 classes.

---

## 11. Zero-Day / Held-Out Attack Experiment

The **Mirai Botnet Family** (`Mirai-greeth_flood`, `Mirai-udpplain`, `Mirai-greip_flood`) was isolated as a proxy for zero-day exploits.

### Maximum Softmax Probability (MSP) Rejection Policy
* Rejection threshold: $\tau = 0.0643$ (calibrated on the 10th percentile of correct validation predictions).
* Test samples with $\max(P) < \tau$ are flagged as `UNKNOWN_ZERO_DAY`.

### Empirical Findings:
* **Known Traffic Macro-F1:** 0.2047
* **Known Traffic Accepted Accuracy:** 46.43%
* **False Unknown Rate on Known Traffic:** 35.63%
* **Zero-Day Detection Rate (MSP alone):** 0.00%
* **Critical Finding:** Under pure softmax classification, the supervised LSTM exhibits overconfidence, misattributing unseen Mirai UDP floods to known classes (`DDoS-UDP_Fragmentation` and `DoS-UDP_Flood`). This underscores why **Module 1 (Fuzzy Anomaly Detection)** is essential in the architecture: Module 1 detected Mirai zero-day traffic with **75.88% recall** based on physical timing and rate departures, preventing it from passing undetected.

---

## 12. Raspberry Pi Deployment Readiness

A dedicated deployment script [`src/module2/inference.py`](file:///d:/Projects/capstone/src/module2/inference.py) provides a standalone runtime for Raspberry Pi:

### Deployment Steps on Raspberry Pi 4 / 5:
```bash
# 1. Install lightweight TFLite runtime on Raspberry Pi OS
pip install tflite-runtime numpy

# 2. Copy the models directory to the Pi:
scp -r models/ pi@raspberrypi.local:~/iot_ids/models/

# 3. Run inference benchmark:
python -c "
from src.module2.inference import Module2InferenceEngine
import numpy as np

engine = Module2InferenceEngine(use_quantized=True)
dummy_sequence = np.random.randn(50, 19).astype(np.float32)
verdict = engine.predict_single_sequence(dummy_sequence)
print('Pi Inference Result:', verdict)
"
```

*Note:* Desktop measurements on an Intel i7 CPU yielded **0.087 ms/sample**. Expected latency on a quad-core ARM Cortex-A72 (Raspberry Pi 4) is $\approx 1.2 - 2.5\text{ ms/sample}$, well within real-time packet processing constraints.

---

## 13. Commands to Run Module 2

```bash
# Activate the environment
.venv\Scripts\activate

# 1. Run complete Module 2 training, quantization, and evaluation pipeline
python src/module2/train.py --samples-per-class 600 --epochs 6 --batch-size 64

# 2. Run unit and integration tests
python -m unittest tests/test_module2.py

# 3. Test end-to-end Module 1 -> Module 2 integration
python -c "
import pandas as pd
from src.module2.integration import PipelineIntegrator
integrator = PipelineIntegrator(use_quantized=True)
m1_df = pd.read_csv('results/window_features.csv')
raw_df = pd.read_csv('data/ciciot2023/test.csv', nrows=20000)
df_out = integrator.run_batch_integration(m1_df.head(50), raw_df)
print(df_out[['window_id', 'module1_verdict', 'predicted_attack', 'module2_inference_latency_ms']].head(10))
"
```

---

## 14. Generated Artifacts

* **Models:**
  * `models/module2_lstm_fp32.keras` (FP32 Keras model, 0.31 MB)
  * `models/module2_lstm_int8.tflite` (INT8 TFLite model, 0.31 MB)
  * `models/module2_random_forest.pkl` (Random Forest baseline, 2.68 MB)
  * `models/module2_scaler.pkl` (Fitted StandardScaler)
  * `models/module2_label_map.json` (31 multiclass integer IDs)
  * `models/module2_features.json` (Selected 19 feature names)
* **Metrics:**
  * `results/metrics/module2_metrics.json` (Complete experimental report)
* **Diagnostic Figures:**
  * `results/plots/module2_loss_curves.png` (Training & validation loss/accuracy curves)
  * `results/plots/module2_confusion_matrix_fp32.png` (Normalized FP32 confusion matrix)
  * `results/plots/module2_confusion_matrix_int8.png` (Normalized INT8 confusion matrix)
  * `results/plots/module2_model_comparison.png` (Comparative performance bar charts)
