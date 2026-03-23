"""
Early stopping and training-curve plots (extracted from main_TrainValTest for standalone file_ver2).
"""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


class EarlyStoppingAfterEpoch(tf.keras.callbacks.Callback):
    """Early stopping that only starts monitoring after start_epoch (for TF < 2.11)."""

    def __init__(self, monitor="val_accuracy", patience=80, start_epoch=100, mode="max", **kwargs):
        super().__init__(**kwargs)
        self.monitor = monitor
        self.patience = patience
        self.start_epoch = start_epoch
        self.mode = mode
        self.best = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or epoch + 1 < self.start_epoch:
            return
        current = logs.get(self.monitor)
        if current is None:
            return
        improved = self.best is None or (
            current > self.best if self.mode == "max" else current < self.best
        )
        if improved:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                print(
                    f"\nEarlyStopping: no improvement in {self.monitor} for {self.patience} epochs "
                    f"(after epoch {epoch + 1})."
                )


def _make_early_stopping(start_epoch=100, patience=80):
    try:
        return EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            verbose=1,
            mode="max",
            restore_best_weights=False,
            start_from_epoch=start_epoch,
        )
    except TypeError:
        return EarlyStoppingAfterEpoch(
            monitor="val_accuracy", patience=patience, start_epoch=start_epoch, mode="max"
        )


def save_training_curves(history, results_dir, prefix="training"):
    """Save training curves (loss and accuracy plots)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], "b-", label="Train Loss", alpha=0.8)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history["accuracy"], "b-", label="Train Accuracy", alpha=0.8)
    ax2.plot(epochs, history["val_accuracy"], "r-", label="Val Accuracy", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{prefix}_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path
