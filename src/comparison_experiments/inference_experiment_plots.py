"""inference_experiment_plots.py — plots for the inference strategy comparison.

Can be used as a module (called from inference_experiment.py) or as a
standalone script to regenerate plots from a saved results JSON:

    python -m comparison_experiments.inference_experiment_plots \
        results/inference_comparison/cora_GCN_20260403_182940.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Colour / style  (same palette as the rest of the codebase)
# ---------------------------------------------------------------------------

_STRATEGY_COLORS = {
    "neighborhood_sampling": "#4C78A8",  # blue
    "in_db_java":            "#F58518",  # orange
    "in_db_cypher":          "#54A24B",  # green
    "in_db_cypher_opt":      "#E45756",  # red
    "full_graph":            "#B279A2",  # purple
    "full_graph_direct":     "#72B7B2",  # teal
}
_FALLBACK_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]

_STRATEGY_LABELS = {
    "full_graph":            "Full-graph in RAM (cold)",
    "full_graph_direct":     "Full-graph in RAM (hot)",
    "neighborhood_sampling": "Neighborhood sampling",
    "in_db_java":            "In-DB Java",
}

_STRATEGY_MARKERS = {
    "full_graph":            "o",
    "full_graph_direct":     "s",
    "neighborhood_sampling": "^",
    "in_db_java":            "D",
    "in_db_cypher":          "v",
    "in_db_cypher_opt":      "P",
}

_MARKERS = ["o", "s", "^", "D", "v", "P"]

# Integer formatter for the N (seed-node count) x-axis so ticks render as
# "1, 2, 4, ..." rather than "1.0, 2.0, 4.0, ...".
_N_FMT = mticker.FuncFormatter(lambda x, _: str(int(x)))


def _color(strat_or_idx) -> str:
    if isinstance(strat_or_idx, str):
        return _STRATEGY_COLORS.get(strat_or_idx, _FALLBACK_COLORS[0])
    return _FALLBACK_COLORS[strat_or_idx % len(_FALLBACK_COLORS)]


def _marker(strat_or_idx) -> str:
    if isinstance(strat_or_idx, str):
        return _STRATEGY_MARKERS.get(strat_or_idx, _MARKERS[0])
    return _MARKERS[strat_or_idx % len(_MARKERS)]


def _lighten(color: str, amount: float = 0.5) -> tuple:
    r, g, b, a = mcolors.to_rgba(color)
    return (1 - amount * (1 - r), 1 - amount * (1 - g), 1 - amount * (1 - b), a)


def _label(strategy: str) -> str:
    return _STRATEGY_LABELS.get(strategy, strategy)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract(results: dict, metric: str) -> dict[str, tuple[list, list, list]]:
    """Return {strategy: (node_counts, means, ci95s)} for a metric."""
    strategies: dict[str, tuple[list, list, list]] = {}
    for n_str, by_strat in sorted(results.items(), key=lambda x: int(x[0])):
        N = int(n_str)
        for strat, agg in by_strat.items():
            entry = agg.get(metric, {})
            mean = entry.get("mean")
            ci95 = entry.get("ci95")
            if mean is None:
                continue
            if strat not in strategies:
                strategies[strat] = ([], [], [])
            strategies[strat][0].append(N)
            strategies[strat][1].append(mean)
            strategies[strat][2].append(ci95 if ci95 is not None else 0.0)
    return strategies




# ---------------------------------------------------------------------------
# 1. Throughput vs N  (log-log)
# ---------------------------------------------------------------------------

def plot_throughput_scaling(results: dict, output_dir: Path) -> None:
    data = _extract(results, "throughput_nodes_per_s")
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat, (ns, means, cis) in data.items():
        ns = np.array(ns); means = np.array(means); cis = np.array(cis)
        color = _color(strat)
        ax.plot(ns, means, label=_label(strat), color=color,
                linewidth=2, marker=_marker(strat), markersize=6)
        ax.fill_between(ns, np.maximum(means - cis, 1e-3), means + cis,
                        alpha=0.18, color=_lighten(color))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_N_FMT)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Throughput (nodes / s)", fontsize=11)
    ax.set_title("Inference throughput vs number of nodes (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_throughput_scaling.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. ms / node vs N
# ---------------------------------------------------------------------------

def _plot_strat(ax, strat, ns, means, cis, ms_scale=1.0, log_y=False):
    """Draw one strategy line."""
    ns = np.array(ns); means = np.array(means) * ms_scale; cis = np.array(cis) * ms_scale
    color = _color(strat)
    ax.plot(ns, means, label=_label(strat), color=color,
            linewidth=2, linestyle="-", marker=_marker(strat), markersize=6)
    lower = np.maximum(means - cis, means * 0.5) if log_y else means - cis
    ax.fill_between(ns, lower, means + cis, alpha=0.10, color=_lighten(color))
    return means


def plot_latency_per_node(results: dict, output_dir: Path) -> None:
    data = _extract(results, "total_time_s")
    fig, ax = plt.subplots(figsize=(8, 5))

    all_means_ms: list[float] = []
    for strat, (ns, means, cis) in data.items():
        m = _plot_strat(ax, strat, ns, means, cis, ms_scale=1000.0, log_y=True)
        all_means_ms.extend(m.tolist())

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_N_FMT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    if all_means_ms:
        ax.set_ylim(bottom=min(all_means_ms) * 0.4)
    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Total inference time (ms)", fontsize=11)
    ax.set_title("Inference latency vs N (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_latency_per_node.png", dpi=150)
    plt.close(fig)


def plot_latency_per_node_linear(results: dict, output_dir: Path) -> None:
    data = _extract(results, "total_time_s")
    fig, ax = plt.subplots(figsize=(8, 5))

    for strat, (ns, means, cis) in data.items():
        _plot_strat(ax, strat, ns, means, cis, ms_scale=1000.0)

    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Total inference time (ms)", fontsize=11)
    ax.set_title("Inference latency vs N (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_latency_per_node_linear.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Accuracy bar chart at the largest N
# ---------------------------------------------------------------------------

def plot_accuracy(results: dict, output_dir: Path, config: dict | None = None) -> None:
    """Bar chart of mean accuracy (± 95 % CI) at the largest measured N.

    The 95 % CI is computed as the Wald binomial CI over the predicted
    nodes themselves (n = N · nbr_runs), rather than across run replicates.
    """
    all_ns = sorted(int(n) for n in results)
    if not all_ns:
        return
    max_n = all_ns[-1]
    max_n_str = str(max_n)

    # Total number of node predictions at max_n: N · nbr_runs.
    nbr_runs_at_max = 1
    if config is not None:
        node_counts = config.get("node_counts") or []
        nbr_runs_list = config.get("nbr_runs") or []
        for nc, nr in zip(node_counts, nbr_runs_list):
            if int(nc) == max_n:
                nbr_runs_at_max = int(nr)
                break
    n_total = max_n * nbr_runs_at_max

    by_strat = results[max_n_str]
    # Collect only real strategies (skip internal _agreement keys)
    strategies = [s for s in by_strat if not s.startswith("_")]

    means, cis, labels, colors = [], [], [], []
    for strat in strategies:
        entry = by_strat[strat].get("accuracy", {})
        mean = entry.get("mean")
        if mean is None:
            continue
        # Wald 95 % CI on a proportion: 1.96 · sqrt(p(1-p)/n)
        ci = 1.96 * np.sqrt(mean * (1 - mean) / n_total) if n_total > 0 else 0.0
        means.append(mean * 100)
        cis.append(ci * 100)
        labels.append(strat)
        colors.append(_color(strat))

    if not means:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(means) * 1.4), 5))
    x = np.arange(len(means))
    bars = ax.bar(
        x, means, yerr=cis, capsize=5,
        color=colors, edgecolor="#333333", linewidth=0.6,
        error_kw={"ecolor": "#333333", "capsize": 5, "elinewidth": 1.3},
    )

    # Value labels above each bar
    for bar, mean, ci in zip(bars, means, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 0.8,
            f"{mean:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # y-axis: start just below the minimum so bars don't look cut off
    y_min = max(0.0, min(means) - max(cis) - 8)
    ax.set_ylim(y_min, min(100, max(means) + max(cis) + 8))

    ax.set_xticks(x)
    ax.set_xticklabels([_label(s) for s in labels], fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title(f"Inference accuracy at N = {max_n} (± 95 % CI)", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_accuracy.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Batch P50 / P95 latency vs N  (batched strategies only)
# ---------------------------------------------------------------------------

def plot_batch_latency(results: dict, output_dir: Path) -> None:
    p50_data = _extract(results, "p50_batch_ms")
    p95_data = _extract(results, "p95_batch_ms")

    # only strategies that have batch latency data
    batched = {s for s, (ns, ms, _) in p50_data.items() if any(m is not None for m in ms)}
    if not batched:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for strat in batched:
        color = _color(strat)
        if strat in p50_data:
            ns, means, cis = p50_data[strat]
            ns = np.array(ns); means = np.array(means); cis = np.array(cis)
            ax.plot(ns, means, label=f"{_label(strat)} P50",
                    color=color, linewidth=2, marker=_marker(strat), markersize=6)
            ax.fill_between(ns, np.maximum(means - cis, 0), means + cis,
                            alpha=0.15, color=_lighten(color))
        if strat in p95_data:
            ns, means, cis = p95_data[strat]
            ns = np.array(ns); means = np.array(means); cis = np.array(cis)
            ax.plot(ns, means, label=f"{_label(strat)} P95",
                    color=color, linewidth=1.5, linestyle="--",
                    marker=_marker(strat), markersize=5)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(_N_FMT)
    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Batch latency (ms)", fontsize=11)
    ax.set_title("Per-batch latency P50 / P95 vs N (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_batch_latency.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Grouped bar: throughput at fixed N values
# ---------------------------------------------------------------------------

def plot_throughput_bars(results: dict, output_dir: Path) -> None:
    """Grouped bar chart at a handful of representative N values."""
    all_ns = sorted(int(n) for n in results)
    # Pick ~4 representative points spread across the range
    if len(all_ns) <= 4:
        selected_ns = all_ns
    else:
        idxs = np.round(np.linspace(0, len(all_ns) - 1, 4)).astype(int)
        selected_ns = [all_ns[i] for i in idxs]

    strategies = list(next(iter(results.values())).keys())
    n_strats = len(strategies)
    n_groups = len(selected_ns)
    width = 0.7 / n_strats

    fig, ax = plt.subplots(figsize=(max(7, n_groups * 2.2), 5))
    x = np.arange(n_groups)

    for si, strat in enumerate(strategies):
        means, cis = [], []
        for N in selected_ns:
            entry = results.get(str(N), {}).get(strat, {}).get("throughput_nodes_per_s", {})
            means.append(entry.get("mean") or 0.0)
            cis.append(entry.get("ci95") or 0.0)
        offset = (si - n_strats / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=cis,
                      label=_label(strat), color=_color(si), capsize=4,
                      error_kw={"ecolor": "#333333", "capsize": 4})
        for bar, mean in zip(bars, means):
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{mean:,.0f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in selected_ns], fontsize=10)
    ax.set_ylabel("Throughput (nodes / s)", fontsize=11)
    ax.set_title("Inference throughput by strategy (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_throughput_bars.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. Crossover: throughput ratio in_db_java / neighborhood_sampling
# ---------------------------------------------------------------------------

def plot_speedup(results: dict, output_dir: Path) -> None:
    """Speedup of each strategy vs neighborhood_sampling (baseline = 1)."""
    strat_data = _extract(results, "throughput_nodes_per_s")
    if "neighborhood_sampling" not in strat_data:
        return

    baseline_ns, baseline_means, _ = strat_data["neighborhood_sampling"]
    baseline_map = dict(zip(baseline_ns, baseline_means))

    fig, ax = plt.subplots(figsize=(8, 4))
    for strat, (ns, means, _) in strat_data.items():
        ratios = []
        valid_ns = []
        for N, mean in zip(ns, means):
            b = baseline_map.get(N)
            if b and b > 0:
                ratios.append(mean / b)
                valid_ns.append(N)
        if not valid_ns:
            continue
        ax.plot(valid_ns, ratios, label=_label(strat), color=_color(strat),
                linewidth=2, marker=_marker(strat), markersize=6)

    ax.axhline(1.0, color="#888888", linewidth=1.2, linestyle="--", label="Baseline (neighborhood_sampling)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(_N_FMT)
    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Throughput ratio vs neighborhood_sampling", fontsize=11)
    ax.set_title("Relative speedup over neighborhood sampling", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_speedup.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. ms / node vs N  (scaling efficiency)
# ---------------------------------------------------------------------------

def plot_ms_per_node_scaling(results: dict, output_dir: Path) -> None:
    """Line chart of ms-per-node vs N for each strategy (log-log).

    A flat line means perfectly linear scaling; an upward slope indicates
    super-linear overhead as batch size grows (e.g. larger subgraphs).
    """
    data = _extract(results, "ms_per_node")
    fig, ax = plt.subplots(figsize=(8, 5))

    all_means: list[float] = []
    for strat, (ns, means, cis) in data.items():
        m = _plot_strat(ax, strat, ns, means, cis, log_y=True)
        all_means.extend(m.tolist())

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_N_FMT)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    if all_means:
        ax.set_ylim(bottom=min(all_means) * 0.4)
    ax.set_xlabel("N (number of seed nodes)", fontsize=11)
    ax.set_ylabel("Latency per node (ms / node)", fontsize=11)
    ax.set_title("Inference latency per node vs N (± 95 % CI)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "inference_ms_per_node.png", dpi=150)
    plt.close(fig)

def plot_all(results_json_path: str | Path, output_dir: Path | None = None) -> Path:
    """Generate all inference experiment plots from a results JSON file.

    Parameters
    ----------
    results_json_path:
        Path to the JSON file written by inference_experiment.py.
    output_dir:
        Where to write PNGs. Defaults to the same directory as the JSON file.

    Returns
    -------
    Path to the output directory.
    """
    results_json_path = Path(results_json_path)
    with open(results_json_path) as f:
        data = json.load(f)

    results = data["results"]
    config = data.get("config")
    if output_dir is None:
        output_dir = results_json_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = results_json_path.stem
    print(f"Generating inference experiment plots → {output_dir}/")

    plot_latency_per_node(results, output_dir)
    plot_latency_per_node_linear(results, output_dir)
    plot_accuracy(results, output_dir, config=config)
    plot_ms_per_node_scaling(results, output_dir)

    plots = [
        "inference_latency_per_node.png",
        "inference_latency_per_node_linear.png",
        "inference_accuracy.png",
        "inference_ms_per_node.png",
    ]
    for p in plots:
        print(f"  {p}")

    return output_dir


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m comparison_experiments.inference_experiment_plots <results.json> [output_dir]")
        sys.exit(1)
    json_path = sys.argv[1]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    plot_all(json_path, out)
