"""
LSTM Model Architecture and Training Module for Module 2.

Implements configurable Keras LSTM network for IoT attack classification:
Input (T, F) -> LSTM(64) -> Dropout(0.2) -> Dense(32, relu) -> Dense(C, softmax)
Includes early stopping, checkpointing, and class-weight balancing.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from src.module2.config import Module2Config, module2_config


class IoTAttackLSTM:
    """Configurable LSTM classifier for temporal IoT flow sequences."""

    def __init__(
        self,
        sequence_length: int = 50,
        num_features: int = 18,
        num_classes: int = 34,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.20,
        learning_rate: float = 0.001,
        cfg: Module2Config = module2_config,
    ):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.num_classes = num_classes
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.cfg = cfg
        self.model = None
        self.history = None

    def build_model(self):
        """Builds and compiles the Keras sequential LSTM model."""
        import tensorflow as tf
        from tensorflow.keras import layers, models, optimizers

        # Set reproducible seeds
        tf.random.set_seed(self.cfg.RANDOM_SEED)
        np.random.seed(self.cfg.RANDOM_SEED)

        model = models.Sequential([
            layers.Input(shape=(self.sequence_length, self.num_features), name="sequence_input"),
            layers.LSTM(self.lstm_units, return_sequences=False, unroll=True, name="lstm_layer"),
            layers.Dropout(self.dropout_rate, name="dropout_layer"),
            layers.Dense(self.dense_units, activation="relu", name="dense_hidden"),
            layers.Dense(self.num_classes, activation="softmax", name="multiclass_output"),
        ])

        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def get_summary_string(self) -> str:
        """Returns string representation of model summary."""
        if self.model is None:
            self.build_model()
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        use_class_weights: bool = True,
        save_best: bool = True,
        model_save_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Trains LSTM model on X_train, validating on X_val.
        Does NOT touch test data.
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        if self.model is None:
            self.build_model()

        b_size = batch_size or self.cfg.BATCH_SIZE
        max_eps = epochs or self.cfg.MAX_EPOCHS
        out_path = model_save_path or self.cfg.FP32_MODEL_PATH
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.PATIENCE,
                restore_best_weights=True,
                verbose=1,
            )
        ]

        if save_best:
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(out_path),
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=1,
                )
            )

        # Class weights for handling class imbalance
        cw_dict = None
        if use_class_weights:
            unique_classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight="balanced",
                classes=unique_classes,
                y=y_train,
            )
            cw_dict = {int(c): float(w) for c, w in zip(unique_classes, weights)}

        print(f"\n--- Starting LSTM Training ({len(X_train):,} train, {len(X_val):,} val sequences) ---")
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=b_size,
            epochs=max_eps,
            callbacks=callbacks,
            class_weight=cw_dict,
            verbose=1,
        )

        return self.history.history

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Outputs softmax probability distributions over multiclass categories."""
        if self.model is None:
            raise ValueError("Model is not built or trained.")
        return self.model.predict(X, verbose=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Outputs predicted class IDs (argmax of softmax probabilities)."""
        probs = self.predict_probabilities(X)
        return np.argmax(probs, axis=1)

    def plot_training_curves(self, save_path: Optional[Path] = None) -> Path:
        """Plots training vs validation loss and accuracy curves."""
        if self.history is None:
            raise ValueError("No training history available to plot.")

        dest = save_path or (self.cfg.PLOTS_DIR / "module2_loss_curves.png")
        dest.parent.mkdir(parents=True, exist_ok=True)

        h = self.history.history
        epochs = range(1, len(h["loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        ax1.plot(epochs, h["loss"], "b-o", label="Training Loss", linewidth=2)
        if "val_loss" in h:
            ax1.plot(epochs, h["val_loss"], "r--s", label="Validation Loss", linewidth=2)
        ax1.set_title("Module 2: Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Epoch", fontsize=11)
        ax1.set_ylabel("Loss", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="upper right")

        # Accuracy curve
        ax2.plot(epochs, h["accuracy"], "g-o", label="Training Accuracy", linewidth=2)
        if "val_accuracy" in h:
            ax2.plot(epochs, h["val_accuracy"], "m--s", label="Validation Accuracy", linewidth=2)
        ax2.set_title("Module 2: Training & Validation Accuracy", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Epoch", fontsize=11)
        ax2.set_ylabel("Accuracy", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="lower right")

        fig.tight_layout()
        fig.savefig(dest, dpi=300)
        plt.close(fig)
        print(f"Training curves saved to {dest}")
        return dest
