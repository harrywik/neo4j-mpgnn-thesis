from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
        epochs = np.arange(1, len(val_accs) + 1, dtype=float)
        acc_values = pd.to_numeric(val_accs["Value"], errors="coerce")
        mask = acc_values.notna()
        if not mask.any():
            return

        epochs = epochs[mask.to_numpy()]
        acc_values = acc_values[mask].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(6, 4))

        if len(epochs) >= 2:
            interp_epochs = np.linspace(float(epochs.min()), float(epochs.max()), num=max(200, len(epochs) * 10))
            interp_acc = np.interp(interp_epochs, epochs, acc_values)
            ax.plot(interp_epochs, interp_acc, linestyle="-", color="#4C78A8", linewidth=1.8)

        ax.scatter(epochs, acc_values, color="#4C78A8", s=18, zorder=3)
        ax.set_title("Validation accuracy convergence")
        ax.set_xlabel("epoch")
        ax.set_ylabel("validation accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(float(epochs.min()), float(epochs.max()))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        fig.tight_layout()
        conv_path = csv_path.with_name("validation_convergence.png")
        fig.savefig(conv_path, dpi=150)
        plt.close(fig)


def plot_validation_convergence_time(csv_path: Path, df: pd.DataFrame) -> None:
    val_accs = get_validation_accuracies(df).copy()
    if val_accs.empty:
        return

    val_accs = val_accs.reset_index(drop=True)
    times = pd.to_numeric(val_accs["Time"], errors="coerce")
    acc_values = pd.to_numeric(val_accs["Value"], errors="coerce")
    mask = times.notna() & acc_values.notna()
    if not mask.any():
        return

    times = times[mask]
    acc_values = acc_values[mask]

    epoch_start = df.loc[df["Event"] == "epoch_start", "Time"]
    if len(epoch_start):
        t0 = float(epoch_start.iloc[0])
    else:
        t0 = float(times.min())

    times = times - t0
    times = times[times >= 0]
    acc_values = acc_values.loc[times.index]
    if times.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, acc_values, marker="o", linestyle="-")
    ax.set_title("Validation accuracy over time")
    ax.set_xlabel("seconds")
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([float(t) for t in times.to_list()])
    fig.tight_layout()
    conv_path = csv_path.with_name("validation_convergence_time.png")
    fig.savefig(conv_path, dpi=150)
    plt.close(fig)


def plot_subphase_latency(csv_path: Path, summary: dict) -> None:
    """Bar chart of per-batch sub-phase latencies from a single run.

    Reads the scalar fields written by the micro-timers in
    ``BaseLineGS`` / ``NoCacheFeatureStore`` and produces a horizontal bar
    chart saved as ``subphase_latency.png`` next to the CSV.
    """
    metrics = summary.get("metrics", {})

    segments = [
        ("network_baseline_ms",    "Network RTT baseline",          "#888888"),
        ("topo_first_record_ms",   "Topology: server exec + RTT",   "#4C78A8"),
        ("topo_transfer_ms",       "Topology: data transfer",       "#7BAFD4"),
        ("topo_etl_ms",            "Topology: Python ETL",          "#AED4F0"),
        ("feat_x_first_record_ms", "Feature-x: server exec + RTT", "#F58518"),
        ("feat_x_transfer_ms",     "Feature-x: data transfer",      "#F7A850"),
        ("feat_x_etl_ms",          "Feature-x: Python ETL",         "#FAC980"),
        ("feat_y_first_record_ms", "Feature-y: server exec + RTT", "#54A24B"),
        ("feat_y_transfer_ms",     "Feature-y: data transfer",      "#7EC47A"),
        ("feat_y_etl_ms",          "Feature-y: Python ETL",         "#AADFAA"),
    ]

    labels, values, colors = [], [], []
    for key, label, color in segments:
        v = metrics.get(key)
        if v is not None:
            labels.append(label)
            values.append(float(v))
            colors.append(color)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55)))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} ms",
            va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch")
    ax.set_title("Sub-phase latency breakdown")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(csv_path.with_name("subphase_latency.png"), dpi=150)
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
