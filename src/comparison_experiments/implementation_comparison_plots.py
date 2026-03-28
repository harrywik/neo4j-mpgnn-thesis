"""Comparison plots for the implementation comparison experiment.

All plotting functions accept:
    impl_data : dict[str, dict]
        Mapping from implementation name to the aggregation dict returned by
        ``aggregate_runs()``.  Keys present in each dict:
            epoch_mean, epoch_std,
            time_grid, time_mean, time_std,
            scalar_means, scalar_stds, n_runs
    output_dir : Path
        Directory where PNG files are written.

Entry point:
    plot_all_comparisons(impl_dirs, output_dir, nbr_runs)
        Calls ``aggregate_runs`` for each implementation directory and then
        generates all comparison plots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Color / style helpers  (same palette as comparison_plots.py)
# ---------------------------------------------------------------------------

_COLORS = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756",
    "#B279A2", "#9D755D", "#72B7B2", "#F2CF5B",
]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _lighten(color: str, amount: float = 0.55) -> tuple:
    r, g, b, a = mcolors.to_rgba(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b), a)


def _ci95(std: float, n: int) -> float:
    """95 % CI half-width from a standard deviation and sample count."""
    return 1.96 * std / (n ** 0.5) if n > 1 else 0.0


# ---------------------------------------------------------------------------
# Validation convergence
# ---------------------------------------------------------------------------

def plot_comparison_validation_convergence(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """One line + 95 % CI band per implementation vs epoch number."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(impl_data.items()):
        y = data["epoch_mean"]
        yerr = data["epoch_std"]
        if len(y) == 0:
            continue
        x = np.arange(1, len(y) + 1)
        color = _color(i)
        ax.plot(x, y, label=name, color=color, linewidth=1.8)
        n = max(data["n_runs"], 1)
        ci = 1.96 * yerr / np.sqrt(n)
        ax.fill_between(x, y - ci, y + ci, alpha=0.25, color=_lighten(color))

    ax.set_title("Validation accuracy vs epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_validation_convergence.png", dpi=150)
    plt.close(fig)


def plot_comparison_validation_convergence_time(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """One line + 95 % CI band per implementation vs wall time (seconds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(impl_data.items()):
        tg = data["time_grid"]
        y = data["time_mean"]
        yerr = data["time_std"]
        if len(tg) == 0:
            continue
        color = _color(i)
        ax.plot(tg, y, label=name, color=color, linewidth=1.8)
        n = max(data["n_runs"], 1)
        ci = 1.96 * yerr / np.sqrt(n)
        ax.fill_between(tg, y - ci, y + ci, alpha=0.25, color=_lighten(color))

    ax.set_title("Validation accuracy vs wall time")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_validation_convergence_time.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase summary  (sampling vs training time per batch)
# ---------------------------------------------------------------------------

def plot_comparison_phase_summary(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """Grouped bars: sampling_mean_s and training_mean_s per batch per implementation."""
    names = list(impl_data.keys())
    sampling_means, sampling_cis = [], []
    training_means, training_cis = [], []

    for name, data in impl_data.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        n = max(data["n_runs"], 1)
        sampling_means.append(float(sm.get("sampling_mean_s") or 0.0))
        sampling_cis.append(_ci95(float(ss.get("sampling_mean_s") or 0.0), n))
        training_means.append(float(sm.get("training_mean_s") or 0.0))
        training_cis.append(_ci95(float(ss.get("training_mean_s") or 0.0), n))

    if not any(sampling_means) and not any(training_means):
        return

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 4))

    bars1 = ax.bar(x - width / 2, sampling_means, width, yerr=sampling_cis,
                   label="Sampling", color="#4C78A8", capsize=4,
                   error_kw={"ecolor": "#2b4b78", "capsize": 4})
    bars2 = ax.bar(x + width / 2, training_means, width, yerr=training_cis,
                   label="Training", color="#F58518", capsize=4,
                   error_kw={"ecolor": "#a05010", "capsize": 4})

    for bar, val in list(zip(bars1, sampling_means)) + list(zip(bars2, training_means)):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3g}s", ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_title("Mean phase time per batch (± 95 % CI)")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_phase_summary.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------

def plot_comparison_throughput(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """Bar + 95 % CI per implementation: throughput in samples / second."""
    names, means, cis = [], [], []
    for name, data in impl_data.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        n = max(data["n_runs"], 1)
        v = sm.get("throughput_samples_per_s")
        if v is not None:
            names.append(name)
            means.append(float(v))
            cis.append(_ci95(float(ss.get("throughput_samples_per_s") or 0.0), n))

    if not names:
        return

    _vertical_bar_chart(
        names, means, cis,
        title="Training throughput (± 95 % CI)",
        ylabel="Samples / second",
        out_path=output_dir / "comparison_throughput.png",
    )


# ---------------------------------------------------------------------------
# Best validation accuracy
# ---------------------------------------------------------------------------

def plot_comparison_best_accuracy(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """Bar + 95 % CI per implementation: best validation accuracy."""
    names, means, cis = [], [], []
    for name, data in impl_data.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        n = max(data["n_runs"], 1)
        v = sm.get("best_validation_accuracy")
        if v is not None:
            names.append(name)
            means.append(float(v))
            cis.append(_ci95(float(ss.get("best_validation_accuracy") or 0.0), n))

    if not names:
        return

    _vertical_bar_chart(
        names, means, cis,
        title="Best validation accuracy (± 95 % CI)",
        ylabel="Accuracy",
        out_path=output_dir / "comparison_best_accuracy.png",
        ylim=(0.0, 1.0),
        fmt=".3f",
    )


# ---------------------------------------------------------------------------
# Sub-phase latency
# ---------------------------------------------------------------------------

_SUBPHASE_SEGMENTS = [
    ("topo_fetch_ms",    "Topology: fetch (wall)"),
    ("topo_etl_ms",      "Topology: Python ETL"),
    ("feat_x_etl_ms",    "Features: Python ETL"),
    ("feat_x_transfer_ms", "Features: transfer + driver"),
]


def plot_comparison_subphase_latency(
    impl_data: dict[str, dict], output_dir: Path
) -> None:
    """Grouped horizontal bars: one colour per implementation, one group per sub-phase segment."""
    impl_names = list(impl_data.keys())

    # Determine which segments have at least one non-null value
    segments_with_data = []
    for seg_key, seg_label in _SUBPHASE_SEGMENTS:
        vals = [
            (float(data["scalar_means"][seg_key])
             if data["scalar_means"].get(seg_key) is not None else None)
            for data in impl_data.values()
        ]
        if any(v is not None for v in vals):
            segments_with_data.append((seg_key, seg_label, vals))

    if not segments_with_data:
        return

    n_segs = len(segments_with_data)
    n_impls = len(impl_names)
    bar_h = 0.8 / n_impls
    y_base = np.arange(n_segs, dtype=float)

    fig, ax = plt.subplots(figsize=(8, max(3, n_segs * 1.1 + 1.5)))
    for impl_idx, (name, data) in enumerate(impl_data.items()):
        color = _color(impl_idx)
        offset = (impl_idx - (n_impls - 1) / 2.0) * bar_h
        vals = [row[2][impl_idx] for row in segments_with_data]
        vals_plot = [v if v is not None else 0.0 for v in vals]
        bars = ax.barh(y_base + offset, vals_plot, bar_h * 0.9, label=name, color=color)
        for bar, v in zip(bars, vals):
            if v is not None and v > 0:
                ax.text(
                    bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {v:.2f}", va="center", ha="left", fontsize=7, color=color,
                )

    ax.set_yticks(y_base)
    ax.set_yticklabels([row[1] for row in segments_with_data], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean ms per batch")
    ax.set_title("Sub-phase latency comparison")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_subphase_latency.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CPU utilisation  (reads measurements.json directly — not in aggregate_runs)
# ---------------------------------------------------------------------------

def _collect_cpu_from_runs(impl_dir: Path, nbr_runs: int) -> tuple[float, float] | None:
    """Return (mean_python_cpu_pct, ci_95) from per-run measurements.json, or None."""
    vals: list[float] = []
    for run_idx in range(nbr_runs):
        run_dir = impl_dir / f"run_{run_idx}"
        mj = run_dir / "measurements.json"
        if not mj.exists():
            continue
        try:
            with open(mj) as f:
                data = json.load(f)
            v = data.get("metrics", {}).get("avg_cpu_utilization")
            if v is not None:
                vals.append(float(v))
        except Exception:
            pass
    if not vals:
        return None
    mean = float(np.mean(vals))
    ci = 1.96 * float(np.std(vals, ddof=1)) / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return mean, ci


def plot_comparison_cpu_bar(
    impl_dirs: dict[str, Path], nbr_runs: int, output_dir: Path
) -> None:
    """Bar + 95 % CI per implementation: average Python CPU utilisation (%)."""
    names, means, cis = [], [], []
    for name, impl_dir in impl_dirs.items():
        result = _collect_cpu_from_runs(impl_dir, nbr_runs)
        if result is not None:
            names.append(name)
            means.append(result[0])
            cis.append(result[1])

    if not names:
        return

    _vertical_bar_chart(
        names, means, cis,
        title="Average Python CPU utilisation (± 95 % CI)",
        ylabel="CPU %",
        out_path=output_dir / "comparison_cpu_bar.png",
        fmt=".1f",
    )


_CPU_EVENTS = [
    "python_cpu_coarse",
    "python_cpu_sampling",
    "python_cpu_etl",
    "python_cpu_training",
]


def _load_cpu_trace(csv_path: Path):
    """Return CPU trace data for a single run, or None on failure.

    Merges all python_cpu_* events, smooths them with a short rolling window,
    and returns time relative to the first epoch_start together with the raw
    coarse samples for marker overlays.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    cpu = df[df["Event"].isin(_CPU_EVENTS)][["Time", "Value"]].copy()
    if cpu.empty:
        return None

    epoch_starts = df.loc[df["Event"] == "epoch_start", "Time"]
    t0 = float(epoch_starts.iloc[0]) if len(epoch_starts) else float(df["Time"].min())

    cpu = cpu.sort_values("Time")
    rel_t = (cpu["Time"] - t0).to_numpy(dtype=float)
    vals = cpu["Value"].to_numpy(dtype=float)

    coarse = df[df["Event"] == "python_cpu_coarse"][["Time", "Value"]].copy()
    coarse_rel_t = (coarse["Time"].to_numpy(dtype=float) - t0) if not coarse.empty else np.array([])
    coarse_vals = coarse["Value"].to_numpy(dtype=float) if not coarse.empty else np.array([])

    # Rolling-mean smoothing: window spans roughly 0.4 s worth of samples.
    if len(vals) > 4:
        total_span = rel_t[-1] - rel_t[0]
        avg_gap = total_span / max(len(vals) - 1, 1)
        win = max(5, int(round(0.4 / avg_gap))) if avg_gap > 0 else 5
        win = min(win, max(5, len(vals) // 3))
        smoothed = pd.Series(vals).rolling(win, center=True, min_periods=1).mean().to_numpy()
    else:
        smoothed = vals

    return rel_t, vals, smoothed, coarse_rel_t, coarse_vals


def plot_comparison_cpu_timeline(
    impl_dirs: dict[str, Path], output_dir: Path
) -> None:
    """CPU utilisation over time from run_0 — one clean panel per implementation.

    Each panel uses local y-scaling so small CPU variations remain visible.
    The plot shows a faint raw trace, a smoothed trend line, and markers for
    the coarse 5-second samples.
    """

    traces = {}
    for name, impl_dir in impl_dirs.items():
        csv_path = impl_dir / "run_0" / "measurements.csv"
        if not csv_path.exists():
            continue
        result = _load_cpu_trace(csv_path)
        if result is not None:
            traces[name] = result

    if not traces:
        return

    n = len(traces)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.9 * n), squeeze=False)

    for idx, (ax_row, (name, trace)) in enumerate(zip(axes, traces.items())):
        ax = ax_row[0]
        color = _color(idx)
        rel_t, raw_vals, smoothed, coarse_rel_t, coarse_vals = trace

        ax.plot(rel_t, raw_vals, color=color, linewidth=0.8, alpha=0.18, zorder=1)
        ax.plot(rel_t, smoothed, color=color, linewidth=2.0, zorder=2)

        if len(coarse_rel_t):
            ax.scatter(
                coarse_rel_t,
                coarse_vals,
                s=28,
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                alpha=0.95,
                zorder=3,
            )

        mean_cpu = float(np.mean(raw_vals))
        peak_cpu = float(np.max(raw_vals))
        ax.axhline(mean_cpu, color=color, linewidth=1.0, linestyle="--", alpha=0.8, zorder=0)

        q_low = float(np.percentile(raw_vals, 2))
        q_high = float(np.percentile(raw_vals, 98))
        spread = max(q_high - q_low, 4.0)
        pad = max(spread * 0.2, 2.0)
        y_min = max(0.0, q_low - pad)
        y_max = q_high + pad

        ax.set_title(f"{name}  |  mean={mean_cpu:.1f}%  peak={peak_cpu:.1f}%", fontsize=10, fontweight="bold", pad=4)
        ax.set_ylabel("CPU %", fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(max(rel_t[0] - 0.1, 0.0), rel_t[-1] + 0.1)
        ax.grid(True, axis="y", linestyle="--", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == n - 1:
            ax.set_xlabel("Elapsed seconds")

    fig.suptitle("Python CPU utilisation — run 0 per implementation", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_cpu_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _vertical_bar_chart(
    names: list[str],
    means: list[float],
    cis: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: tuple[float, float] | None = None,
    fmt: str = ".3g",
) -> None:
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.8), 4))
    for i, (xi, mean, ci) in enumerate(zip(x, means, cis)):
        color = _color(i)
        ax.bar(xi, mean, yerr=ci, color=color, capsize=4, width=0.55,
               error_kw={"ecolor": "#333333", "capsize": 4})
        ax.text(xi, mean, f"{mean:{fmt}}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------------------

def plot_all_comparisons(
    impl_dirs: dict[str, Path],
    output_dir: Path,
    nbr_runs: int,
) -> None:
    """Collect aggregate data for each implementation and generate all comparison plots.

    Parameters
    ----------
    impl_dirs:
        Mapping from implementation name to the directory that contains
        ``run_0/``, ``run_1/``, … subdirectories.
    output_dir:
        Directory where comparison PNGs are written (usually the parent of
        all impl dirs).
    nbr_runs:
        Number of runs per implementation (passed to ``aggregate_runs``).
    """
    from comparison_experiments.sampler_comparison.comparison_plots import aggregate_runs

    impl_data: dict[str, dict] = {}
    for name, impl_dir in impl_dirs.items():
        agg = aggregate_runs(impl_dir, nbr_runs)
        if agg["n_runs"] == 0:
            print(f"  Warning: no valid runs found for '{name}' in {impl_dir}")
            continue
        impl_data[name] = agg

    if not impl_data:
        print("No valid implementation data found, skipping comparison plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating comparison plots → {output_dir}")

    plot_comparison_validation_convergence(impl_data, output_dir)
    plot_comparison_validation_convergence_time(impl_data, output_dir)
    plot_comparison_phase_summary(impl_data, output_dir)
    plot_comparison_throughput(impl_data, output_dir)
    plot_comparison_best_accuracy(impl_data, output_dir)
    plot_comparison_subphase_latency(impl_data, output_dir)
    plot_comparison_cpu_bar(impl_dirs, nbr_runs, output_dir)
    plot_comparison_cpu_timeline(impl_dirs, output_dir)

    print(f"  Comparison plots written to: {output_dir}")
