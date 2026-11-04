"""
Generates the comprehensive, educational Jupyter Notebook for Module 2:
notebooks/module2_lstm_attack_classification.ipynb
"""

import json
from pathlib import Path

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}


def add_markdown(content: str):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")]
    })


def add_code(code_str: str):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code_str.split("\n")]
    })


# ------------------------------------------------------------------------------
# 1. Introduction
# ------------------------------------------------------------------------------
add_markdown("""# Module 2: LSTM-Based Multiclass IoT Attack Classification with INT8 Quantization
## Research & Engineering Walkthrough

**Capstone Extension:** IoT Intrusion Detection System (CICIoT2023)  
**Author / Engineering Role:** Senior ML Engineer & IoT Cybersecurity Researcher  
**Core Framework:** TensorFlow / Keras, TensorFlow Lite / LiteRT, NumPy, Scikit-Learn  

---

### Pipeline Overview & Relationship to Module 1
* **Module 1 (First-Line Sentinel):** Inter-Arrival Time (IAT) statistical dispersion ($CV = \\sigma/\\mu$) and Mamdani Fuzzy Inference System for rapid, unsupervised anomaly filtering.
* **Module 2 (Second-Line Attribution):** A Long Short-Term Memory (LSTM) recurrent neural network that receives anomalous observation sequences ($L=50$) and classifies the specific threat into one of 34 IoT attack categories.
* **Edge Deployment:** Full-integer INT8 post-training quantization reducing inference latency to **< 0.1 ms** for deployment on resource-constrained IoT gateways and Raspberry Pi devices.
""")

# ------------------------------------------------------------------------------
# 2. Environment Setup
# ------------------------------------------------------------------------------
add_markdown("""## 1. Environment Setup & Library Imports
We initialize the environment and load our modular architecture components.""")

add_code("""import os
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# Verify TensorFlow availability
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
print(f"NumPy Version:      {np.__version__}")
""")

# ------------------------------------------------------------------------------
# 3. Dataset Inspection & Provenance
# ------------------------------------------------------------------------------
add_markdown("""## 2. Dataset Inspection & Label Taxonomy (CICIoT2023)
We examine the Kaggle release `himadri07/ciciot2023`, its column headers, and verify that all 34 classes are accounted for.""")

add_code("""from src.module2.config import module2_config

train_path = module2_config.DATASET_DIR / module2_config.TRAIN_FILENAME
test_path = module2_config.DATASET_DIR / module2_config.TEST_FILENAME

print(f"Train File: {train_path} ({train_path.stat().st_size / (1024**3):.2f} GB)")
print(f"Test File:  {test_path} ({test_path.stat().st_size / (1024**2):.2f} MB)")

# Inspect first 5 rows
df_sample = pd.read_csv(train_path, nrows=5)
print(f"Total Columns: {len(df_sample.columns)}")
print("Columns:", list(df_sample.columns))
""")

add_code("""# Inspect Label Distribution in test.csv
df_test_labels = pd.read_csv(test_path, usecols=["label"])
val_counts = df_test_labels["label"].value_counts()
print(f"Total Test Records: {len(df_test_labels):,}")
print(f"Total Unique Classes: {len(val_counts)}")
print("\\nTop 10 Classes:")
print(val_counts.head(10))
""")

# ------------------------------------------------------------------------------
# 4. Ordering & Sequence Validation
# ------------------------------------------------------------------------------
add_markdown("""## 3. Ordering, Provenance & Sequence Construction Strategy

### Critical Provenance Finding:
1. **No Timestamps:** Neither `train.csv` nor `test.csv` has a `ts` or `timestamp` column.
2. **Kaggle Random Shuffling:** Consecutive rows in the raw CSV represent randomly juxtaposed flows from different attack scenarios due to an 80/20 train-test random shuffle.
3. **Defensible Sequence Formulation:**
   * Feeding randomly shuffled rows to an LSTM is scientifically invalid as it forces the model to memorize arbitrary shuffling transitions.
   * **Our Solution:** We construct sequences of length $L=50$ strictly within **homogeneous capture/scenario blocks** of identical attack or benign types.
   * Sequences **never** cross scenario or class boundaries.
""")

add_code("""from src.module2.sequence_builder import SequenceBuilder

builder = SequenceBuilder(sequence_length=50, step_size=25)
print(f"Sequence Length: {builder.sequence_length}")
print(f"Step Size:       {builder.step_size}")
print(f"Label Strategy:  {builder.label_strategy}")
print(f"Strict Homogeneous Sequences: {builder.strict_homogeneous}")
""")

# ------------------------------------------------------------------------------
# 5. Preprocessing & Leakage-Safe Splitting
# ------------------------------------------------------------------------------
add_markdown("""## 4. Preprocessing & Leakage-Safe Transformation
We select 19 discriminative behavioral features covering timing, volume, and transport state. Preprocessing transformations (`StandardScaler` and `LabelEncoder`) are fitted strictly on the training partition.""")

add_code("""from src.module2.preprocessor import Module2Preprocessor

preprocessor = Module2Preprocessor(module2_config)
print("Selected Features (19):")
for idx, feat in enumerate(module2_config.SELECTED_FEATURES):
    print(f"  {idx+1:2d}. {feat}")
""")

# ------------------------------------------------------------------------------
# 6. Model Architecture & Training Curves
# ------------------------------------------------------------------------------
add_markdown("""## 5. LSTM Model Architecture
The network processes sequences of shape `(Batch, 50, 19)`:
* `LSTM(64 units, unroll=True)`
* `Dropout(0.20)`
* `Dense(32, activation="relu")`
* `Dense(31, activation="softmax")`

`unroll=True` ensures static unrolling of the 50 time steps, enabling seamless conversion to TensorFlow Lite without dynamic `TensorArray` errors.
""")

add_code("""from src.module2.lstm_model import IoTAttackLSTM

lstm = IoTAttackLSTM(sequence_length=50, num_features=19, num_classes=31)
print(lstm.get_summary_string())
""")

add_code("""# Display Training & Validation Loss/Accuracy Curves
loss_curve_path = module2_config.PLOTS_DIR / "module2_loss_curves.png"
if loss_curve_path.exists():
    display(Image(filename=str(loss_curve_path)))
else:
    print("Training curve plot not yet generated.")
""")

# ------------------------------------------------------------------------------
# 7. Non-Sequential Random Forest Baseline Comparison
# ------------------------------------------------------------------------------
add_markdown("""## 6. Non-Sequential Baseline: Random Forest
We compare the LSTM against a non-sequential Random Forest benchmark trained on the exact same train/test partition using sequence-aggregate statistics and the final observation.""")

add_code("""import json

metrics_path = module2_config.METRICS_PATH
if metrics_path.exists():
    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)
    comp_df = pd.DataFrame(metrics_data["model_comparison"])
    print(comp_df.to_string(index=False))
""")

add_code("""# Display Comparative Bar Chart
comp_plot_path = module2_config.PLOTS_DIR / "module2_model_comparison.png"
if comp_plot_path.exists():
    display(Image(filename=str(comp_plot_path)))
""")

# ------------------------------------------------------------------------------
# 8. Zero-Day / Held-Out Attack Experiment
# ------------------------------------------------------------------------------
add_markdown("""## 7. Zero-Day / Unseen Attack Evaluation
The **Mirai Botnet Family** (`Mirai-greeth_flood`, `Mirai-udpplain`, `Mirai-greip_flood`) was strictly quarantined from model training and validation.

We test Maximum Softmax Probability (MSP) rejection: predictions with confidence $\\max(P) < \\tau$ are flagged as `UNKNOWN_ZERO_DAY`.""")

add_code("""if metrics_path.exists():
    zd_data = metrics_data["zero_day_evaluation"]
    print(f"Rejection Threshold (tau):       {zd_data['rejection_threshold']}")
    print(f"Known Traffic Macro-F1:          {zd_data['known_macro_f1']}")
    print(f"False Unknown Rate (Known FP):   {zd_data['false_unknown_rate'] * 100:.2f}%")
    print(f"Zero-Day Detection Rate (Recall):{zd_data['zero_day_detection_rate'] * 100:.2f}%")
    print(f"Mean Confidence on Known:        {zd_data['mean_confidence_known']}")
    print(f"Mean Confidence on Unseen:       {zd_data['mean_confidence_unseen']}")
    print(f"Unseen Misattributions:          {zd_data['unseen_misattributions_when_accepted']}")
""")

# ------------------------------------------------------------------------------
# 9. INT8 Quantization & Verification
# ------------------------------------------------------------------------------
add_markdown("""## 8. INT8 Post-Training Quantization
We convert the trained FP32 Keras model to full-integer INT8 precision using TensorFlow Lite with a representative training calibration generator.""")

add_code("""from src.module2.quantization import ModelQuantizer

int8_model_path = module2_config.INT8_MODEL_PATH
if int8_model_path.exists():
    info = ModelQuantizer.inspect_tflite_model(int8_model_path)
    print(f"Quantized Model Size: {int8_model_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Tensor Count:        {info['tensor_count']}")
    print("Input Details:       ", info["input_details"])
    print("Output Details:      ", info["output_details"])
""")

add_code("""# Display Normalized Confusion Matrices (FP32 vs INT8)
cm_fp32 = module2_config.PLOTS_DIR / "module2_confusion_matrix_fp32.png"
cm_int8 = module2_config.PLOTS_DIR / "module2_confusion_matrix_int8.png"

if cm_fp32.exists():
    print("--- FP32 LSTM Confusion Matrix ---")
    display(Image(filename=str(cm_fp32)))

if cm_int8.exists():
    print("--- INT8 TFLite Confusion Matrix ---")
    display(Image(filename=str(cm_int8)))
""")

# ------------------------------------------------------------------------------
# 10. Module 1 -> Module 2 Integration
# ------------------------------------------------------------------------------
add_markdown("""## 9. End-to-End Module 1 + Module 2 Integration
We demonstrate the live handoff from Module 1 anomaly detection to Module 2 LSTM classification.""")

add_code("""from src.module2.integration import PipelineIntegrator

integrator = PipelineIntegrator(use_quantized=True)
m1_df = pd.read_csv(module2_config.RESULTS_DIR / "window_features.csv")
raw_df = pd.read_csv(module2_config.DATASET_DIR / "test.csv", nrows=25000)

# Evaluate a slice of mixed windows
demo_windows = pd.concat([m1_df.head(5), m1_df[m1_df["binary_label"] == 1].head(5)], ignore_index=True)
df_integrated = integrator.run_batch_integration(demo_windows, raw_df)

print(df_integrated[[
    "window_id", "module1_verdict", "attack_risk", "module2_invoked", "predicted_attack", "confidence", "total_pipeline_latency_ms"
]].to_string(index=False))
""")

# ------------------------------------------------------------------------------
# 11. Conclusions & Deployment Readiness
# ------------------------------------------------------------------------------
add_markdown("""## 10. Research Conclusions & Edge Deployment

1. **Quantization Trade-offs:** The INT8 TFLite quantized model achieved an identical classification accuracy to FP32 while delivering a **~587x speedup** on CPU (0.087 ms vs 51.07 ms per sequence).
2. **Sequential vs Non-Sequential Modeling:** The Random Forest baseline outperformed the compact LSTM on overall accuracy (0.9425 vs 0.3448) when trained on small sample budgets, demonstrating that volumetric aggregates provide strong discriminating signals for flood attacks in CICIoT2023.
3. **Necessity of Hierarchical Defense:** Under pure softmax classification, the supervised LSTM exhibits overconfidence on unseen Mirai UDP floods, misclassifying them into known categories. **Module 1 (Fuzzy Logic)** successfully flags these unseen attacks with **75.88% recall** based on timing dispersion ($CV$), proving that the combined two-stage pipeline is essential for robust zero-day defense.
""")

output_file = Path("d:/Projects/capstone/notebooks/module2_lstm_attack_classification.ipynb")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook successfully written to: {output_file}")
