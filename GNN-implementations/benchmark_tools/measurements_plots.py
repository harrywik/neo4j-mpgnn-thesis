from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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


def plot_cpu_utilization(csv_path: Path, df: pd.DataFrame) -> None:
    cpu = df[df["Event"] == "cpu_utilization_percentage"][["Time", "Value"]].copy()
    ram = df[df["Event"] == "ram_usage_mb"][["Time", "Value"]].copy()
    if cpu.empty:
        return

    program_start = df.loc[df["Event"] == "program_start", "Time"]
    program_end = df.loc[df["Event"] == "program_end", "Time"]
    if len(program_start):
        t0 = float(program_start.iloc[0])
    else:
        t0 = float(df["Time"].min())

    if len(program_end):
        t1 = float(program_end.iloc[-1])
    else:
        t1 = float(df["Time"].max())

    if t1 <= t0:
        return

    cpu["rel_time"] = pd.to_numeric(cpu["Time"], errors="coerce") - t0
    cpu = cpu.dropna(subset=["rel_time", "Value"]).sort_values("rel_time")
    if cpu.empty:
        return

    x = cpu["rel_time"].to_numpy()
    y = pd.to_numeric(cpu["Value"], errors="coerce").to_numpy()
    if len(x) == 0:
        return

    # Interpolate to a uniform grid from 0 to duration
    duration = t1 - t0
    grid = np.linspace(0.0, duration, num=200)
    y_interp = np.interp(grid, x, y)

    fig, ax = plt.subplots(figsize=(6, 4))

    def _shade(event_start: str, event_end: str, color: str, alpha: float) -> None:
        starts = df.loc[df["Event"] == event_start, "Time"].to_list()
        ends = df.loc[df["Event"] == event_end, "Time"].to_list()
        n = min(len(starts), len(ends))
        for s, e in zip(starts[:n], ends[:n]):
            ax.axvspan(s - t0, e - t0, color=color, alpha=alpha, lw=0, zorder=0)

    # Sampling = light blue background
    _shade("start_batch_fetch", "end_batch_fetch", color="#DCEEFF", alpha=0.4)
    # Training = white background (subtle overlay to preserve contrast)
    _shade("start_batch_processing", "end_batch_processing", color="#FFFFFF", alpha=0.2)

    ax.plot(grid, y_interp, linestyle="-", zorder=2, label="CPU %")
    ax.set_title("CPU/RAM utilization over time")
    ax.set_xlabel("seconds")
    ax.set_ylabel("cpu utilization (%)")
    ax.set_xlim(0.0, duration)

    # RAM on secondary axis
    if not ram.empty:
        ram["rel_time"] = pd.to_numeric(ram["Time"], errors="coerce") - t0
        ram = ram.dropna(subset=["rel_time", "Value"]).sort_values("rel_time")
        if not ram.empty:
            rx = ram["rel_time"].to_numpy()
            ry = pd.to_numeric(ram["Value"], errors="coerce").to_numpy()
            if len(rx):
                rgrid = np.linspace(0.0, duration, num=200)
                r_interp = np.interp(rgrid, rx, ry)
                ax2 = ax.twinx()
                ax2.plot(rgrid, r_interp, linestyle="--", color="#2CA02C", label="RAM (MB)")
                ax2.set_ylabel("ram usage (MB)")

    # Background legend/comment
    ax.text(
        0.01,
        0.98,
        "Background: light blue = sampling, white = training",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="none"),
        zorder=3,
    )

    fig.tight_layout()
    plot_path = csv_path.with_name("cpu_utilization.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
