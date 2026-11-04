"""
Configuration module for Module 2: LSTM-Based IoT Multiclass Attack Classification
with INT8 Quantization.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Module2Config:
    # --------------------------------------------------------------------------
    # 1. Directory & File Paths
    # --------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATASET_DIR: Path = BASE_DIR / "data" / "ciciot2023"
    TRAIN_FILENAME: str = "train.csv"
    TEST_FILENAME: str = "test.csv"

    MODELS_DIR: Path = BASE_DIR / "models"
    RESULTS_DIR: Path = BASE_DIR / "results"
    PLOTS_DIR: Path = RESULTS_DIR / "plots"
    METRICS_DIR: Path = RESULTS_DIR / "metrics"

    # Saved artifacts
    FP32_MODEL_PATH: Path = MODELS_DIR / "module2_lstm_fp32.keras"
    INT8_MODEL_PATH: Path = MODELS_DIR / "module2_lstm_int8.tflite"
    SCALER_PATH: Path = MODELS_DIR / "module2_scaler.pkl"
    LABEL_MAP_PATH: Path = MODELS_DIR / "module2_label_map.json"
    FEATURES_PATH: Path = MODELS_DIR / "module2_features.json"
    METRICS_PATH: Path = METRICS_DIR / "module2_metrics.json"

    # --------------------------------------------------------------------------
    # 2. Selected Input Features (18 features covering timing, volume, & state)
    # --------------------------------------------------------------------------
    # Strict exclusion: label, binary target, scenario IDs, post-hoc tags
    SELECTED_FEATURES: List[str] = field(
        default_factory=lambda: [
            # Timing & Rates
            "IAT",
            "Rate",
            "Srate",
            "Drate",
            "flow_duration",
            "Duration",
            # Volume & Packet Size Statistics
            "Header_Length",
            "Tot size",
            "Tot sum",
            "Number",
            "AVG",
            "Std",
            "Min",
            "Max",
            # Protocols & Transport Flags
            "Protocol Type",
            "TCP",
            "UDP",
            "syn_count",
            "ack_count",
        ]
    )

    # --------------------------------------------------------------------------
    # 3. Label & Attack Family Settings
    # --------------------------------------------------------------------------
    NORMAL_LABEL: str = "BenignTraffic"

    # Held-out attack family for Zero-Day evaluation (identical to Module 1)
    HELD_OUT_ATTACKS: List[str] = field(
        default_factory=lambda: [
            "Mirai-greeth_flood",
            "Mirai-udpplain",
            "Mirai-greip_flood",
        ]
    )

    # --------------------------------------------------------------------------
    # 4. Sequence Construction
    # --------------------------------------------------------------------------
    DEFAULT_SEQUENCE_LENGTH: int = 50
    SEQUENCE_EXPERIMENT_LENGTHS: List[int] = field(
        default_factory=lambda: [10, 20, 50, 100]
    )
    # Step size when sliding within a homogeneous block
    SEQUENCE_STEP: int = 25
    # Label strategy: 'last' observation in sequence receives target label
    LABEL_STRATEGY: str = "last"
    # Filter out mixed sequences across class transitions
    STRICT_HOMOGENEOUS_SEQUENCES: bool = True

    # --------------------------------------------------------------------------
    # 5. Data Sampling & Splits
    # --------------------------------------------------------------------------
    # Subsampling cap per class for fast, balanced, and tractable training
    SAMPLES_PER_CLASS_TRAIN: int = 2500
    SAMPLES_PER_CLASS_VAL: int = 500
    SAMPLES_PER_CLASS_TEST: int = 500

    TRAIN_SPLIT_RATIO: float = 0.70
    VAL_SPLIT_RATIO: float = 0.15
    TEST_SPLIT_RATIO: float = 0.15

    # --------------------------------------------------------------------------
    # 6. LSTM Architecture Hyperparameters
    # --------------------------------------------------------------------------
    LSTM_UNITS: int = 64
    DROPOUT_RATE: float = 0.20
    DENSE_UNITS: int = 32
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 64
    MAX_EPOCHS: int = 25
    PATIENCE: int = 5

    # --------------------------------------------------------------------------
    # 7. Random Forest Baseline Hyperparameters
    # --------------------------------------------------------------------------
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 15

    # --------------------------------------------------------------------------
    # 8. Zero-Day Uncertainty Rejection Policy
    # --------------------------------------------------------------------------
    # Softmax confidence rejection threshold (max probability < threshold -> UNKNOWN)
    REJECTION_THRESHOLD_PERCENTILE: float = 0.10  # Tune on validation data (e.g. 10th percentile of correct predictions)

    # --------------------------------------------------------------------------
    # 9. Reproducibility
    # --------------------------------------------------------------------------
    RANDOM_SEED: int = 42

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)


module2_config = Module2Config()
