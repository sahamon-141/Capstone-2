"""
Configuration module for IAT-Based Fuzzy Logic IoT Network Anomaly Detection.

This module centralizes all file paths, feature lists, sliding-window parameters,
adaptive threshold parameters, fuzzy inference boundaries, zero-day experiment settings,
and execution flags.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ProjectConfig:
    # --------------------------------------------------------------------------
    # 1. Paths & File Management
    # --------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATASET_DIR: Path = BASE_DIR / "data" / "ciciot2023"
    TRAIN_FILENAME: str = "train.csv"
    TEST_FILENAME: str = "test.csv"
    
    RESULTS_DIR: Path = BASE_DIR / "results"
    PLOTS_DIR: Path = RESULTS_DIR / "plots"
    METRICS_DIR: Path = RESULTS_DIR / "metrics"
    WINDOW_FEATURES_PATH: Path = RESULTS_DIR / "window_features.csv"
    
    # --------------------------------------------------------------------------
    # 2. Dataset Labels & Features
    # --------------------------------------------------------------------------
    NORMAL_LABEL: str = "BenignTraffic"
    
    # Core raw features extracted from the 46 features available in CICIoT2023
    CORE_RAW_FEATURES: List[str] = field(
        default_factory=lambda: ["IAT", "Rate", "Number", "flow_duration"]
    )
    
    # --------------------------------------------------------------------------
    # 3. Sliding Window Parameters
    # --------------------------------------------------------------------------
    # Consecutive observation windowing (not reconstructed timestamps)
    WINDOW_SIZE: int = 20
    OVERLAP: float = 0.50
    
    @property
    def STEP_SIZE(self) -> int:
        return max(1, int(self.WINDOW_SIZE * (1.0 - self.OVERLAP)))
    
    # Optional experiment to analyze impact of window sizes
    WINDOW_EXPERIMENT_SIZES: List[int] = field(
        default_factory=lambda: [10, 20, 50, 100]
    )
    
    # Window attack classification threshold (ratio of attack samples in window)
    WINDOW_ATTACK_THRESHOLD: float = 0.50

    # --------------------------------------------------------------------------
    # 4. Adaptive Thresholding (Median + MAD) Parameters
    # --------------------------------------------------------------------------
    MAD_K: float = 3.0
    K_EXPERIMENT_LIST: List[float] = field(
        default_factory=lambda: [2.0, 2.5, 3.0, 3.5, 4.0]
    )
    # Maximum risk score or CV ratio to allow normal history adaptation
    HISTORY_UPDATE_CONFIDENCE_THRESHOLD: float = 0.35
    ADAPTATION_BUFFER_SIZE: int = 500  # Sliding memory buffer for history

    # --------------------------------------------------------------------------
    # 5. Fuzzy Logic System Parameters
    # --------------------------------------------------------------------------
    # Decision boundaries for converting continuous Attack Risk [0, 1] to categories
    FUZZY_LOW_THRESHOLD: float = 0.35   # Risk < 0.35 -> NORMAL
    FUZZY_HIGH_THRESHOLD: float = 0.65  # Risk >= 0.65 -> ATTACK; [0.35, 0.65) -> SUSPICIOUS
    
    # Fuzzy input membership universes and percentiles (derived from normal baseline)
    FUZZY_RESOLUTION: int = 200

    # --------------------------------------------------------------------------
    # 6. Zero-Day / Unseen Attack Experiment Configuration
    # --------------------------------------------------------------------------
    # Held-out attack categories completely excluded from training & tuning
    # Options: Mirai family ('Mirai-greeth_flood', 'Mirai-udpplain', 'Mirai-greip_flood')
    # or Recon family ('Recon-PortScan', 'Recon-OSScan', 'Recon-HostDiscovery')
    HELD_OUT_ATTACKS: List[str] = field(
        default_factory=lambda: [
            "Mirai-greeth_flood",
            "Mirai-udpplain",
            "Mirai-greip_flood"
        ]
    )

    # --------------------------------------------------------------------------
    # 7. Performance & Reproducibility
    # --------------------------------------------------------------------------
    SAMPLE_MODE: bool = True
    SAMPLE_SIZE: int = 100000      # 100k flows gives ~10,000 sliding windows
    CHUNK_SIZE: int = 50000        # Chunk size for streaming large files
    RANDOM_SEED: int = 42

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        self.DATASET_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Default global instance
config = ProjectConfig()
