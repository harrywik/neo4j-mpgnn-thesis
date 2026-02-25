from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .measurements_summary import pair_durations, get_validation_accuracies


def plot_phase_summary(csv_path: Path, df: pd.DataFrame) -> None:
    sampling_durations = pair_durations(df, "start_batch_fetch", "end_batch_fetch")
    training_durations = pair_durations(df, "start_batch_processing", "end_batch_processing")

    if len(sampling_durations) or len(training_durations):
        sampling_mean = float(sampling_durations.mean()) if len(sampling_durations) else 0.0
        training_mean = float(training_durations.mean()) if len(training_durations) else 0.0

        labels = ["sampling", "training"]
        means = [sampling_mean, training_mean]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, means, color=["#4C78A8", "#F58518"])
        ax.set_title("Phase durations (mean)")
        ax.set_ylabel("seconds")

        fig.tight_layout()
        plot_path = csv_path.with_name("phase_summary.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


def plot_validation_convergence(csv_path: Path, df: pd.DataFrame) -> None:
    val_accs = get_validation_accuracies(df).copy()
    if len(val_accs):
        val_accs = val_accs.reset_index(drop=True)
        epochs = list(range(1, len(val_accs) + 1))
        acc_values = pd.to_numeric(val_accs["Value"], errors="coerce")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epochs, acc_values, marker="o", linestyle="-")
        ax.set_title("Validation accuracy convergence")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(epochs)
        fig.tight_layout()
        conv_path = csv_path.with_name("validation_convergence.png")
        fig.savefig(conv_path, dpi=150)
        plt.close(fig)
