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

        color = "#E45756"
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(epochs, acc_values, color=color, linewidth=1.8)
        ax.scatter(epochs, acc_values, color=color, s=18, zorder=3)

        # Mark best epoch with a dashed vertical line
        best_idx = int(np.argmax(acc_values))
        best_epoch = float(epochs[best_idx])
        ax.axvline(x=best_epoch, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            best_epoch, 0.02, f"{int(round(best_epoch))}",
            color=color, fontsize=7, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(),
        )

        ax.set_title("Validation accuracy vs epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.set_ylim(0.0, 1.0)
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

    times = (times - t0).to_numpy(dtype=float)
    acc_values = acc_values.to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(acc_values) & (times >= 0)
    times = times[valid]
    acc_values = acc_values[valid]
    if len(times) == 0:
        return

    color = "#E45756"
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(times, acc_values, color=color, linewidth=1.8)
    ax.scatter(times, acc_values, color=color, s=18, zorder=3)

    # Mark best epoch with a dashed vertical line
    best_idx = int(np.argmax(acc_values))
    ax.axvline(x=times[best_idx], color=color, linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        times[best_idx], 0.02, f"{times[best_idx]:.1f}s",
        color=color, fontsize=7, ha="center", va="bottom",
        transform=ax.get_xaxis_transform(),
    )

    ax.set_title("Validation accuracy vs wall time")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
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

    topo_segments = [
        ("topo_fetch_ms",               "Topology: total fetch (wall)",              "#2171B5"),
        ("sampler_avg_db_exec_time_ms", "Topology: DB execution",                    "#4C78A8"),
        ("network_baseline_ms",         "Topology: driver + protocol overhead",      "#7BAFD4"),
        ("topo_etl_ms",                 "Topology: Python ETL",                      "#AED4F0"),
    ]
    # x and y share one query, so all timings cover both in a single round-trip.
    # The profiler attributes the combined query to "feat_x" internally.
    feat_segments = [
        ("feat_x_avg_client_wall_time_ms", "Features ([float], int): total fetch",         "#D94F00"),
        ("feat_x_avg_db_exec_time_ms",     "Features ([float], int): DB execution",               "#F58518"),
        ("feat_y_avg_db_exec_time_ms",     "Features ([float], int): DB execution (y)",           "#54A24B"),
        ("feat_x_avg_driver_overhead_ms",  "Features ([float], int): non-exec time", "#F7A850"),
        ("feat_x_etl_ms",                  "Features ([float], int): Python ETL",                 "#FAC980"),
    ]

    def _resolve(segs):
        rows = []
        for key, label, color in segs:
            v = metrics.get(key)
            if v is not None:
                rows.append((label, float(v), color))
        return sorted(rows, key=lambda r: r[1], reverse=True)

    topo_rows = _resolve(topo_segments)
    feat_rows = _resolve(feat_segments)
    all_rows = topo_rows + feat_rows

    if not all_rows:
        return

    labels = [r[0] for r in all_rows]
    values = [r[1] for r in all_rows]
    colors = [r[2] for r in all_rows]

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
