"""Comparison plots for the sampler comparison experiment.

After all sampler runs have completed, ``plot_comparison`` reads the
per-run ``validation_accuracies.csv`` and ``measurements.json`` files,
aggregates them across runs (mean ± std), and writes multi-sampler
comparison plots and a summary JSON to the experiment directory.
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
import pandas as pd

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_data(run_dir: Path) -> dict[str, Any] | None:
    """Load per-run data from a single run directory.

    Returns a dict with:
        ``val_accs``  — DataFrame with columns [Time, Value] (one row per epoch)
        ``summary``   — dict loaded from measurements.json
        ``df``        — raw measurements DataFrame

    Returns ``None`` if required files are missing.
    """
    val_acc_path = run_dir / "validation_accuracies.csv"
    json_path = run_dir / "measurements.json"
    csv_path = run_dir / "measurements.csv"

    if not val_acc_path.exists() or not json_path.exists():
        return None

    val_accs = pd.read_csv(val_acc_path)
    with open(json_path) as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    return {"val_accs": val_accs, "summary": summary, "df": df}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _epoch_series(val_accs: pd.DataFrame) -> np.ndarray:
    """Return a 1-D array of validation accuracy values (one per epoch)."""
    vals = pd.to_numeric(val_accs["Value"], errors="coerce").to_numpy(dtype=float)
    return vals


def _time_series(val_accs: pd.DataFrame, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (times_s, accuracies) normalized so t=0 is the first epoch_start."""
    times = pd.to_numeric(val_accs["Time"], errors="coerce").to_numpy(dtype=float)
    accs = pd.to_numeric(val_accs["Value"], errors="coerce").to_numpy(dtype=float)

    epoch_starts = df.loc[df["Event"] == "epoch_start", "Time"]
    t0 = float(epoch_starts.iloc[0]) if len(epoch_starts) else float(times[0])
    times = times - t0
    mask = np.isfinite(times) & np.isfinite(accs) & (times >= 0)
    return times[mask], accs[mask]


def aggregate_runs(
    sampler_dir: Path, num_runs: int
) -> dict[str, Any]:
    """Aggregate across all runs for one sampler.

    Returns:
        epoch_mean, epoch_std  — shape (min_epochs,)
        time_grid              — shape (300,) shared time axis in seconds
        time_mean, time_std    — shape (300,) interpolated accuracy stats
        scalar_means, scalar_stds — dicts of aggregated scalar metrics
    """
    epoch_arrays: list[np.ndarray] = []
    time_curves: list[tuple[np.ndarray, np.ndarray]] = []
    scalars: dict[str, list[float]] = {}

    for run_idx in range(num_runs):
        run_dir = sampler_dir / f"run_{run_idx}"
        data = load_run_data(run_dir)
        if data is None:
            continue

        va = data["val_accs"]
        df = data["df"]
        summary = data["summary"]
        metrics = summary.get("metrics", {})
        run_info = summary.get("run", {})

        epoch_arrays.append(_epoch_series(va))
        time_curves.append(_time_series(va, df))

        # Collect scalars from measurements.json
        def _collect(key: str, value: Any) -> None:
            if isinstance(value, (int, float)) and value is not None:
                scalars.setdefault(key, []).append(float(value))

        _collect("best_validation_accuracy", metrics.get("best_validation_accuracy"))
        _collect("final_validation_accuracy", metrics.get("final_validation_accuracy"))
        _collect("throughput_samples_per_s", metrics.get("throughput_samples_per_s"))
        _collect("time_to_best_accuracy_s", metrics.get("time_to_best_accuracy_s"))
        _collect("training_time_s", run_info.get("training_time_s"))
        _collect("avg_batch_nodes", metrics.get("avg_batch_nodes"))
        _collect("avg_batch_edges", metrics.get("avg_batch_edges"))
        _collect("remote_feature_total_s", metrics.get("remote_feature_total_s"))

        sp = metrics.get("sampling_phase_time_s", {}) or {}
        tp = metrics.get("training_phase_time_s", {}) or {}
        _collect("sampling_mean_s", sp.get("mean_s"))
        _collect("training_mean_s", tp.get("mean_s"))

        # Derived: sampling overhead fraction
        s_mean = sp.get("mean_s")
        t_mean = tp.get("mean_s")
        if s_mean is not None and t_mean is not None and (s_mean + t_mean) > 0:
            _collect("sampling_overhead_fraction", s_mean / (s_mean + t_mean))

        # Derived: batch size CV
        bn_vals = df.loc[df["Event"] == "batch_nbr_nodes_total", "Value"]
        bn_vals = pd.to_numeric(bn_vals, errors="coerce").dropna()
        if len(bn_vals) > 1 and bn_vals.mean() > 0:
            _collect("batch_nodes_cv", float(bn_vals.std() / bn_vals.mean()))

        # Derived: epochs to best accuracy
        va_vals = pd.to_numeric(va["Value"], errors="coerce").to_numpy(dtype=float)
        if len(va_vals):
            _collect("epochs_to_best_accuracy", float(np.argmax(va_vals) + 1))

        # Derived: feature fetch overhead fraction
        rft = metrics.get("remote_feature_total_s")
        tt = run_info.get("training_time_s")
        if rft is not None and tt is not None and tt > 0:
            _collect("feature_fetch_overhead_fraction", rft / tt)

        # Sub-phase micro-timer scalars (mean ms per batch/call)
        for key in (
            "topo_query_sent_ms", "topo_first_record_ms",
            "topo_transfer_ms", "topo_etl_ms",
            "feat_x_query_sent_ms", "feat_x_first_record_ms",
            "feat_x_transfer_ms", "feat_x_etl_ms",
            "feat_y_query_sent_ms", "feat_y_first_record_ms",
            "feat_y_transfer_ms", "feat_y_etl_ms",
            "network_baseline_ms",
        ):
            _collect(key, metrics.get(key))

        # Derived: estimated server exec times (first_record - RTT baseline)
        rtt = metrics.get("network_baseline_ms")
        if rtt is not None:
            topo_fr = metrics.get("topo_first_record_ms")
            if topo_fr is not None:
                _collect("topo_server_exec_ms", max(0.0, topo_fr - rtt))
            fx_fr = metrics.get("feat_x_first_record_ms")
            fy_fr = metrics.get("feat_y_first_record_ms")
            if fx_fr is not None and fy_fr is not None:
                _collect("feat_server_exec_ms", max(0.0, fx_fr + fy_fr - 2 * rtt))

    # Epoch-based aggregation (trim to shortest run)
    if epoch_arrays:
        min_len = min(len(a) for a in epoch_arrays)
        stacked = np.stack([a[:min_len] for a in epoch_arrays])
        epoch_mean = stacked.mean(axis=0)
        epoch_std = stacked.std(axis=0)
    else:
        epoch_mean = epoch_std = np.array([])

    # Time-based aggregation (interpolate onto shared grid)
    if time_curves:
        max_t = max(t[-1] for t, _ in time_curves if len(t))
        time_grid = np.linspace(0.0, max_t, 300)
        interp_curves = np.stack(
            [np.interp(time_grid, t, a, left=np.nan, right=np.nan)
             for t, a in time_curves]
        )
        time_mean = np.nanmean(interp_curves, axis=0)
        time_std = np.nanstd(interp_curves, axis=0)
    else:
        time_grid = np.array([])
        time_mean = time_std = np.array([])

    scalar_means = {k: float(np.mean(v)) for k, v in scalars.items()}
    scalar_stds = {k: float(np.std(v)) for k, v in scalars.items()}

    return {
        "epoch_mean": epoch_mean,
        "epoch_std": epoch_std,
        "time_grid": time_grid,
        "time_mean": time_mean,
        "time_std": time_std,
        "scalar_means": scalar_means,
        "scalar_stds": scalar_stds,
        "n_runs": len(epoch_arrays),
    }


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D"]


def _color(i: int) -> str:
    return _COLORS[i % len(_COLORS)]


def _lighten(color: str, amount: float = 0.55) -> tuple:
    """Return a lighter version of *color* by blending it toward white.

    ``amount=0`` returns the original color; ``amount=1`` returns white.
    """
    r, g, b, a = mcolors.to_rgba(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b), a)


def plot_accuracy_vs_epochs(
    agg: dict[str, dict], out_dir: Path
) -> None:
    """Plot mean validation accuracy per epoch with best-epoch markers."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(agg.items()):
        y = data["epoch_mean"]
        yerr = data["epoch_std"]
        if len(y) == 0:
            continue
        x = np.arange(1, len(y) + 1)
        color = _color(i)
        ax.plot(x, y, label=name, color=color, linewidth=1.8)
        n = max(data["n_runs"], 1)
        ci = 1.96 * yerr / np.sqrt(n)
        ax.fill_between(x, y - ci, y + ci, alpha=0.35, color=_lighten(color))

        best_epoch = data["scalar_means"].get("epochs_to_best_accuracy")
        if best_epoch is not None:
            epoch_int = int(round(best_epoch))
            ax.axvline(x=best_epoch, color=color, linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(
                best_epoch, 0.02, f"{epoch_int}",
                color=color, fontsize=7, ha="center", va="bottom",
                transform=ax.get_xaxis_transform(),
            )

    ax.set_title("Validation accuracy vs epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_accuracy_vs_epochs.png", dpi=150)
    plt.close(fig)


def plot_accuracy_vs_time(
    agg: dict[str, dict], out_dir: Path
) -> None:
    """Plot mean validation accuracy vs wall time, one line per sampler."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (name, data) in enumerate(agg.items()):
        tg = data["time_grid"]
        y = data["time_mean"]
        yerr = data["time_std"]
        if len(tg) == 0:
            continue
        color = _color(i)
        ax.plot(tg, y, label=name, color=color, linewidth=1.8)
        n = max(data["n_runs"], 1)
        ci = 1.96 * yerr / np.sqrt(n)
        ax.fill_between(tg, y - ci, y + ci, alpha=0.35, color=_lighten(color))

    ax.set_title("Validation accuracy vs wall time")
    ax.set_xlabel("Elapsed seconds")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_accuracy_vs_time.png", dpi=150)
    plt.close(fig)


def _bar_chart(
    names: list[str],
    means: list[float],
    stds: list[float],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4))
    x = np.arange(len(names))
    for i, (xi, mean, std) in enumerate(zip(x, means, stds)):
        color = _color(i)
        bar = ax.bar(xi, mean, yerr=std, color=color, capsize=5, width=0.5,
                     error_kw={"ecolor": _lighten(color, 0.4), "elinewidth": 1.5})
        ax.text(
            xi,
            mean,
            f"{mean:.3g}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sampling_time(agg: dict[str, dict], out_dir: Path) -> None:
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "sampling_mean_s" in sm:
            names.append(name)
            means.append(sm["sampling_mean_s"] * 1000)  # ms
            stds.append(ss.get("sampling_mean_s", 0.0) * 1000)
    _bar_chart(
        names, means, stds,
        "Mean sampling time per batch",
        "milliseconds",
        out_dir / "comparison_sampling_time.png",
    )


def plot_batch_size(agg: dict[str, dict], out_dir: Path) -> None:
    """Grouped bar chart: avg_batch_nodes and avg_batch_edges per sampler."""
    names = list(agg.keys())
    nodes_mean = [agg[n]["scalar_means"].get("avg_batch_nodes", 0.0) for n in names]
    nodes_std = [agg[n]["scalar_stds"].get("avg_batch_nodes", 0.0) for n in names]
    edges_mean = [agg[n]["scalar_means"].get("avg_batch_edges", 0.0) for n in names]
    edges_std = [agg[n]["scalar_stds"].get("avg_batch_edges", 0.0) for n in names]

    x = np.arange(len(names))
    w = 0.35
    _nodes_color = "#4C78A8"
    _edges_color = "#F58518"
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.8), 4))
    bars_n = ax.bar(x - w / 2, nodes_mean, w, yerr=nodes_std, label="Nodes", capsize=4,
                    color=_nodes_color,
                    error_kw={"ecolor": _lighten(_nodes_color, 0.4), "elinewidth": 1.5})
    bars_e = ax.bar(x + w / 2, edges_mean, w, yerr=edges_std, label="Edges", capsize=4,
                    color=_edges_color,
                    error_kw={"ecolor": _lighten(_edges_color, 0.4), "elinewidth": 1.5})
    for bar, val in zip(bars_n, nodes_mean):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3g}",
            ha="center", va="bottom", fontsize=8,
        )
    for bar, val in zip(bars_e, edges_mean):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3g}",
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_title("Mean batch size (nodes and edges)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_batch_size.png", dpi=150)
    plt.close(fig)


def plot_epoch_to_best_acc(agg: dict[str, dict], out_dir: Path) -> None:
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "epochs_to_best_accuracy" in sm:
            names.append(name)
            means.append(sm["epochs_to_best_accuracy"])
            stds.append(ss.get("epochs_to_best_accuracy", 0.0))
    _bar_chart(
        names, means, stds,
        "Mean epoch to best validation accuracy",
        "epoch",
        out_dir / "comparison_time_to_best_acc.png",
    )


def plot_best_val_acc(agg: dict[str, dict], out_dir: Path) -> None:
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "best_validation_accuracy" in sm:
            names.append(name)
            means.append(sm["best_validation_accuracy"])
            stds.append(ss.get("best_validation_accuracy", 0.0))
    _bar_chart(
        names, means, stds,
        "Mean best validation accuracy",
        "accuracy",
        out_dir / "comparison_best_val_acc.png",
    )


def plot_throughput(agg: dict[str, dict], out_dir: Path) -> None:
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "throughput_samples_per_s" in sm:
            names.append(name)
            means.append(sm["throughput_samples_per_s"])
            stds.append(ss.get("throughput_samples_per_s", 0.0))
    _bar_chart(
        names, means, stds,
        "Training throughput",
        "samples / second",
        out_dir / "comparison_throughput.png",
    )


# ---------------------------------------------------------------------------
# Latency breakdown plots
# ---------------------------------------------------------------------------

# Ordered segment definitions for the stacked bar chart.
# Each entry: (scalar_key, display_label, colour)
_LATENCY_SEGMENTS = [
    ("topo_first_record_ms",  "Topo: server+RTT",   "#4C78A8"),
    ("topo_transfer_ms",      "Topo: transfer",      "#7BAFD4"),
    ("topo_etl_ms",           "Topo: ETL",           "#AED4F0"),
    ("feat_x_first_record_ms","Feat-x: server+RTT",  "#F58518"),
    ("feat_x_transfer_ms",    "Feat-x: transfer",    "#F7A850"),
    ("feat_x_etl_ms",         "Feat-x: ETL",         "#FAC980"),
    ("feat_y_first_record_ms","Feat-y: server+RTT",  "#54A24B"),
    ("feat_y_transfer_ms",    "Feat-y: transfer",    "#7EC47A"),
    ("feat_y_etl_ms",         "Feat-y: ETL",         "#AADFAA"),
    ("training_mean_ms",      "GNN training",        "#E45756"),
]


def _get_training_mean_ms(sm: dict) -> float | None:
    """Return mean training-phase time in ms from scalar_means."""
    v = sm.get("training_mean_s")
    return v * 1000 if v is not None else None


def plot_latency_breakdown(agg: dict[str, dict], out_dir: Path) -> None:
    """Stacked bar chart: mean batch latency split into sub-phases per sampler."""
    names = list(agg.keys())
    if not names:
        return

    # Build a matrix: rows = segments, cols = samplers
    seg_keys = [s[0] for s in _LATENCY_SEGMENTS]
    seg_labels = [s[1] for s in _LATENCY_SEGMENTS]
    seg_colors = [s[2] for s in _LATENCY_SEGMENTS]

    values: list[list[float]] = []  # [n_segments][n_samplers]
    for key, _, _ in _LATENCY_SEGMENTS:
        row = []
        for name in names:
            sm = agg[name]["scalar_means"]
            if key == "training_mean_ms":
                v = _get_training_mean_ms(sm)
            else:
                v = sm.get(key)
            row.append(v if v is not None else 0.0)
        values.append(row)

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2), 5))

    bottoms = np.zeros(len(names))
    for seg_vals, label, color in zip(values, seg_labels, seg_colors):
        seg_arr = np.array(seg_vals)
        bars = ax.bar(x, seg_arr, bottom=bottoms, label=label, color=color, width=0.5)
        # Label each segment if tall enough to be readable
        for xi, (bot, val) in enumerate(zip(bottoms, seg_arr)):
            if val > 1.0:
                ax.text(xi, bot + val / 2, f"{val:.1f}", ha="center", va="center",
                        fontsize=6.5, color="white" if val > 4 else "black")
        bottoms += seg_arr

    # Horizontal dashed line for the network RTT baseline
    rtt_values = [agg[n]["scalar_means"].get("network_baseline_ms") for n in names]
    rtt_values = [v for v in rtt_values if v is not None]
    if rtt_values:
        rtt_mean = float(np.mean(rtt_values))
        ax.axhline(rtt_mean, color="black", linestyle="--", linewidth=1.2,
                   label=f"RTT baseline ({rtt_mean:.1f} ms)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.set_title("Mean batch latency breakdown")
    ax.set_ylabel("milliseconds")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "comparison_latency_breakdown.png", dpi=150)
    plt.close(fig)


def plot_rtt_baseline(agg: dict[str, dict], out_dir: Path) -> None:
    """Bar chart: network RTT baseline (RETURN 1 round-trip) per sampler."""
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "network_baseline_ms" in sm:
            names.append(name)
            means.append(sm["network_baseline_ms"])
            stds.append(ss.get("network_baseline_ms", 0.0))
    if not names:
        return
    _bar_chart(
        names, means, stds,
        "Network RTT baseline (RETURN 1)",
        "milliseconds",
        out_dir / "comparison_rtt_baseline.png",
    )


def plot_topology_server_exec(agg: dict[str, dict], out_dir: Path) -> None:
    """Bar chart: estimated server execution time for the topology Cypher query."""
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "topo_server_exec_ms" in sm:
            names.append(name)
            means.append(sm["topo_server_exec_ms"])
            stds.append(ss.get("topo_server_exec_ms", 0.0))
    if not names:
        return
    _bar_chart(
        names, means, stds,
        "Est. topology query server execution",
        "milliseconds",
        out_dir / "comparison_topology_server_exec.png",
    )


def plot_feature_server_exec(agg: dict[str, dict], out_dir: Path) -> None:
    """Bar chart: estimated server execution time for feature fetch queries (x+y)."""
    names, means, stds = [], [], []
    for name, data in agg.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        if "feat_server_exec_ms" in sm:
            names.append(name)
            means.append(sm["feat_server_exec_ms"])
            stds.append(ss.get("feat_server_exec_ms", 0.0))
    if not names:
        return
    _bar_chart(
        names, means, stds,
        "Est. feature fetch server execution (x + y)",
        "milliseconds",
        out_dir / "comparison_feature_server_exec.png",
    )


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------

def write_summary(
    agg: dict[str, dict], out_dir: Path, num_runs: int
) -> None:
    summary: dict[str, Any] = {}
    for name, data in agg.items():
        summary[name] = {
            "n_runs_completed": data["n_runs"],
            "num_epochs_in_plot": int(len(data["epoch_mean"])),
            "scalar_means": data["scalar_means"],
            "scalar_stds": data["scalar_stds"],
        }
    out_path = out_dir / "comparison_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def plot_comparison(
    experiment_dir: Path,
    sampler_names: list[str],
    num_runs: int,
) -> None:
    """Aggregate all runs and generate all comparison plots.

    Args:
        experiment_dir: Top-level directory for this comparison run
            (e.g. ``experiment_results/sampler_comparison/comparison_0/``).
        sampler_names: Ordered list of sampler labels to include.
        num_runs: Number of runs per sampler.
    """
    agg: dict[str, dict] = {}
    for name in sampler_names:
        sampler_dir = experiment_dir / name
        if not sampler_dir.exists():
            print(f"  WARNING: {sampler_dir} not found — skipping {name}.")
            continue
        data = aggregate_runs(sampler_dir, num_runs)
        if data["n_runs"] == 0:
            print(f"  WARNING: No completed runs found for {name} — skipping.")
            continue
        agg[name] = data
        print(f"  Aggregated {data['n_runs']} runs for {name}.")

    if not agg:
        print("No data to plot.")
        return

    plot_accuracy_vs_epochs(agg, experiment_dir)
    plot_accuracy_vs_time(agg, experiment_dir)
    plot_sampling_time(agg, experiment_dir)
    plot_batch_size(agg, experiment_dir)
    plot_epoch_to_best_acc(agg, experiment_dir)
    plot_best_val_acc(agg, experiment_dir)
    plot_latency_breakdown(agg, experiment_dir)
    plot_rtt_baseline(agg, experiment_dir)
    plot_topology_server_exec(agg, experiment_dir)
    plot_feature_server_exec(agg, experiment_dir)
    write_summary(agg, experiment_dir, num_runs)

    print(f"  Plots written to {experiment_dir}")
