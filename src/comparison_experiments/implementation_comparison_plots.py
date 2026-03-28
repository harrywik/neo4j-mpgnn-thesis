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
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
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

_C_DRIVER = "#D9D9D9"
_C_ETL = "#F0F0F0"


def _avg_query_profile_globals(impl_dir: Path, nbr_runs: int) -> tuple[dict[str, float], dict[str, float]]:
    sampler_vals: dict[str, list[float]] = defaultdict(list)
    feat_vals: dict[str, list[float]] = defaultdict(list)

    for run_idx in range(nbr_runs):
        run_dir = impl_dir / f"run_{run_idx}"
        qp_path = run_dir / "query_profile.json"
        if not qp_path.exists():
            continue
        try:
            with open(qp_path) as f:
                qp = json.load(f)
        except Exception:
            continue

        for key, value in qp.get("sampler", {}).get("global", {}).items():
            if isinstance(value, (int, float)):
                sampler_vals[key].append(float(value))
        for key, value in qp.get("feat_x", {}).get("global", {}).items():
            if isinstance(value, (int, float)):
                feat_vals[key].append(float(value))

    sampler_avg = {k: float(np.mean(v)) for k, v in sampler_vals.items() if v}
    feat_avg = {k: float(np.mean(v)) for k, v in feat_vals.items() if v}
    return sampler_avg, feat_avg


def _build_avg_end_to_end_slices(
    impl_dir: Path,
    agg: dict[str, Any],
    nbr_runs: int,
) -> list[tuple[str, list[tuple[str, float, str]]]]:
    sampler_avg, feat_avg = _avg_query_profile_globals(impl_dir, nbr_runs)
    metrics = agg["scalar_means"]

    topo_slices: list[tuple[str, float, str]] = []
    topo_server = sampler_avg.get("avg_result_consumed_after_ms")
    topo_startup = sampler_avg.get("avg_query_startup_ms", 0.0)
    topo_recv = sampler_avg.get("avg_client_recv_ms")
    topo_etl = metrics.get("topo_etl_ms")
    if topo_server and topo_server > 0:
        t_first = float(topo_startup) if topo_startup else 0.0
        t_rest = float(topo_server) - t_first
        if t_first > 0:
            topo_slices.append(("DB time-to-first-row", t_first, "#08519C"))
        if t_rest > 0:
            topo_slices.append(("DB server latency (topology)", t_rest, "#2171B5"))
    if topo_recv and topo_recv > 0:
        topo_slices.append(("Network + driver recv", float(topo_recv), _C_DRIVER))
    if topo_etl and topo_etl > 0:
        topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))
    if not topo_slices:
        topo_wall = sampler_avg.get("avg_client_wall_time_ms") or metrics.get("topo_fetch_ms")
        if topo_wall and topo_wall > 0:
            topo_slices.append(("Total fetch (wall)", float(topo_wall), "#2171B5"))
        if topo_etl and topo_etl > 0:
            topo_slices.append(("Python ETL", float(topo_etl), _C_ETL))

    feat_slices: list[tuple[str, float, str]] = []
    feat_server = feat_avg.get("avg_result_consumed_after_ms")
    feat_startup = feat_avg.get("avg_query_startup_ms", 0.0)
    feat_recv = feat_avg.get("avg_client_recv_ms")
    feat_etl = metrics.get("feat_x_etl_ms")
    if feat_server and feat_server > 0:
        t_first = float(feat_startup) if feat_startup else 0.0
        t_rest = float(feat_server) - t_first
        if t_first > 0:
            feat_slices.append(("DB time-to-first-row", t_first, "#C44E00"))
        if t_rest > 0:
            feat_slices.append(("DB server latency (features)", t_rest, "#F58518"))
    if feat_recv and feat_recv > 0:
        feat_slices.append(("Network + driver recv", float(feat_recv), _C_DRIVER))
    if feat_etl and feat_etl > 0:
        feat_slices.append(("Python ETL", float(feat_etl), _C_ETL))
    if not feat_slices:
        feat_wall = feat_avg.get("avg_client_wall_time_ms")
        if feat_wall and feat_wall > 0:
            feat_slices.append(("Total fetch (wall)", float(feat_wall), "#F58518"))
        if feat_etl and feat_etl > 0:
            feat_slices.append(("Python ETL", float(feat_etl), _C_ETL))

    rows: list[tuple[str, list[tuple[str, float, str]]]] = []
    if topo_slices:
        rows.append(("Topology", topo_slices))
    if feat_slices:
        rows.append(("Features", feat_slices))
    return rows


def plot_comparison_subphase_latency(
    impl_data: dict[str, dict],
    impl_dirs: dict[str, Path],
    nbr_runs: int,
    output_dir: Path,
) -> None:
    """Averaged end-to-end-style latency breakdown for all implementations."""
    grouped_rows: list[tuple[str, list[tuple[str, list[tuple[str, float, str]]]]]] = []
    for impl_name, agg in impl_data.items():
        impl_rows = _build_avg_end_to_end_slices(impl_dirs[impl_name], agg, nbr_runs)
        if impl_rows:
            grouped_rows.append((impl_name, impl_rows))

    if not grouped_rows:
        return

    n_rows = sum(len(impl_rows) for _, impl_rows in grouped_rows)
    bar_height = 0.52
    group_gap = 0.58

    fig, ax = plt.subplots(figsize=(11, max(3.0, n_rows * 0.9 + len(grouped_rows) * 0.35)))
    yticks: list[float] = []
    ylabels: list[str] = []
    legend_handles: dict[str, tuple[str, str]] = {}
    max_total = 0.0
    current_y = 0.0

    for impl_name, impl_rows in grouped_rows:
        for stage_idx, (stage_name, slices) in enumerate(impl_rows):
            row_y = current_y + stage_idx * bar_height
            left = 0.0
            total = sum(width for _, width, _ in slices)
            max_total = max(max_total, total)
            for seg_label, width, color in slices:
                if width <= 0:
                    continue
                ax.barh(row_y, width, left=left, height=bar_height, color=color,
                        edgecolor="white", linewidth=0.5)
                if total > 0 and width / total > 0.08:
                    ax.text(
                        left + width / 2,
                        row_y,
                        f"{width:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if color not in (_C_DRIVER, _C_ETL) else "#333333",
                        fontweight="bold",
                    )
                left += width
                legend_handles[seg_label] = (seg_label, color)

            ax.text(left, row_y, f" {left:.1f} ms", ha="left", va="center", fontsize=8)
            yticks.append(row_y)
            ylabels.append(f"{impl_name} | {stage_name}")

        current_y += len(impl_rows) * bar_height + group_gap

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("mean ms per batch (avg across runs)")
    ax.set_title("End-to-end latency breakdown by implementation")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    if max_total > 0:
        ax.set_xlim(0, max_total * 1.22)

    legend_patches = [Patch(facecolor=color, label=label, edgecolor="white")
                      for label, color in legend_handles.values()]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=7, framealpha=0.85, ncol=1)
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
    plot_comparison_subphase_latency(impl_data, impl_dirs, nbr_runs, output_dir)
    plot_comparison_cpu_bar(impl_dirs, nbr_runs, output_dir)
    plot_comparison_cpu_timeline(impl_dirs, output_dir)

    print(f"  Comparison plots written to: {output_dir}")
