"""Combined dataset comparison plots for one fixed implementation.

Entry point:
    plot_all_dataset_comparisons(dataset_dirs, output_dir, nbr_runs)

Only writes:
    - dataset_comparison_phase_summary.png
    - dataset_comparison_throughput.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ci95(std: float, n: int) -> float:
    return 1.96 * std / (n ** 0.5) if n > 1 else 0.0


def plot_dataset_phase_summary(dataset_data: dict[str, dict], output_dir: Path) -> None:
    """Grouped bars: sampling_mean_s and training_mean_s per dataset."""
    datasets = list(dataset_data.keys())
    sampling_means, sampling_cis = [], []
    training_means, training_cis = [], []

    for _, data in dataset_data.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        n = max(data["n_runs"], 1)
        sampling_means.append(float(sm.get("sampling_mean_s") or 0.0))
        sampling_cis.append(_ci95(float(ss.get("sampling_mean_s") or 0.0), n))
        training_means.append(float(sm.get("training_mean_s") or 0.0))
        training_cis.append(_ci95(float(ss.get("training_mean_s") or 0.0), n))

    if not any(sampling_means) and not any(training_means):
        return

    x = np.arange(len(datasets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 2), 4.2))

    bars1 = ax.bar(
        x - width / 2,
        sampling_means,
        width,
        yerr=sampling_cis,
        label="Sampling",
        color="#4C78A8",
        capsize=4,
        error_kw={"ecolor": "#2b4b78", "capsize": 4},
    )
    bars2 = ax.bar(
        x + width / 2,
        training_means,
        width,
        yerr=training_cis,
        label="Training",
        color="#F58518",
        capsize=4,
        error_kw={"ecolor": "#a05010", "capsize": 4},
    )

    for bar, val in list(zip(bars1, sampling_means)) + list(zip(bars2, training_means)):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3g}s",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=9)
    ax.set_title("Mean phase time per batch by dataset (± 95 % CI)")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_comparison_phase_summary.png", dpi=150)
    plt.close(fig)


def plot_dataset_throughput(dataset_data: dict[str, dict], output_dir: Path) -> None:
    """Bar chart for throughput_samples_per_s by dataset with 95 % CI."""
    datasets, means, cis = [], [], []
    for dataset_name, data in dataset_data.items():
        sm = data["scalar_means"]
        ss = data["scalar_stds"]
        n = max(data["n_runs"], 1)
        v = sm.get("throughput_samples_per_s")
        if v is None:
            continue
        datasets.append(dataset_name)
        means.append(float(v))
        cis.append(_ci95(float(ss.get("throughput_samples_per_s") or 0.0), n))

    if not datasets:
        return

    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.8), 4.2))

    bars = ax.bar(
        x,
        means,
        yerr=cis,
        color="#54A24B",
        width=0.55,
        capsize=4,
        error_kw={"ecolor": "#2f6d31", "capsize": 4},
    )

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{mean:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=9)
    ax.set_title("Training throughput by dataset (± 95 % CI)")
    ax.set_ylabel("Samples / second")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "dataset_comparison_throughput.png", dpi=150)
    plt.close(fig)


def plot_all_dataset_comparisons(
    dataset_dirs: dict[str, Path], output_dir: Path, nbr_runs: int
) -> None:
    """Aggregate runs for each dataset and generate combined dataset comparison plots."""
    from comparison_experiments.sampler_comparison.comparison_plots import aggregate_runs

    dataset_data: dict[str, dict] = {}
    for dataset_name, dataset_dir in dataset_dirs.items():
        agg = aggregate_runs(dataset_dir, nbr_runs)
        if agg["n_runs"] == 0:
            print(f"  Warning: no valid runs found for dataset '{dataset_name}' in {dataset_dir}")
            continue
        dataset_data[dataset_name] = agg

    if not dataset_data:
        print("No valid dataset data found, skipping dataset comparison plots.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating dataset comparison plots -> {output_dir}")

    plot_dataset_phase_summary(dataset_data, output_dir)
    plot_dataset_throughput(dataset_data, output_dir)

    print(f"  Dataset comparison plots written to: {output_dir}")
