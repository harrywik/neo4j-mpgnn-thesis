"""Generate query-profile plots for one or more run directories.

Reads ``query_profile.json`` (and optionally ``train_profile.txt`` /
``measurements.json``) from each run directory and writes all plots into
``query_profile_plots/``.

Usage
-----
# single dataset directory:
python -m benchmarking_tools.plot_query_profiles \\
    --run_dir experiment_results/report_files/rq_bottlenecks/baseline_neo4j/papers100M

# parent directory — every immediate sub-directory that has a query_profile.json
# is processed in one go:
python -m benchmarking_tools.plot_query_profiles \\
    --run_dir experiment_results/report_files/rq_bottlenecks/baseline_neo4j
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarking_tools.measurements_plots import (
    plot_all_operator_profiles,
    plot_end_to_end_latency,
)


def generate_query_profile_plots(run_dir: Path) -> None:
    """Generate all query-profile plots for *run_dir* into ``query_profile_plots/``."""
    profile_path = run_dir / "query_profile.json"
    if not profile_path.exists():
        print(f"  No query_profile.json in {run_dir}, skipping.")
        return

    out_dir = run_dir / "query_profile_plots"
    out_dir.mkdir(exist_ok=True)

    print(f"  Generating query profile plots for {run_dir.name}...")

    # Per-operator plots (no-op when operators arrays are empty).
    plot_all_operator_profiles(profile_path, output_dir=out_dir)

    # End-to-end latency waterfall — reads global timing from query_profile.json.
    train_profile_path = run_dir / "train_profile.txt"

    # Build a minimal summary dict so the function can find subphase metrics.
    # Prefer measurements.json if available; fall back to the embedded
    # subphase_metrics block inside query_profile.json itself.
    summary: dict = {}
    measurements_json = run_dir / "measurements.json"
    if measurements_json.exists():
        with open(measurements_json) as f:
            summary = json.load(f)
    else:
        with open(profile_path) as f:
            qp = json.load(f)
        subphase = qp.get("subphase_metrics", {})
        if subphase:
            summary = {"metrics": subphase}

    plot_end_to_end_latency(profile_path, train_profile_path, summary, output_dir=out_dir)
    print(f"  Done → {out_dir}")


def _resolve_dirs(path: Path) -> list[Path]:
    if (path / "query_profile.json").exists():
        return [path]
    sub_dirs = sorted(
        p for p in path.iterdir()
        if p.is_dir() and (p / "query_profile.json").exists()
    )
    if not sub_dirs:
        raise FileNotFoundError(
            f"No query_profile.json found in {path} or any of its immediate sub-directories."
        )
    return sub_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help=(
            "Path to a run directory containing query_profile.json, or a parent "
            "directory whose immediate sub-directories each contain one."
        ),
    )
    args = parser.parse_args()

    run_dirs = _resolve_dirs(args.run_dir.resolve())
    for rd in run_dirs:
        generate_query_profile_plots(rd)
