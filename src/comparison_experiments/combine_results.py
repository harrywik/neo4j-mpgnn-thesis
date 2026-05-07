"""combine_results.py — regenerate comparison plots from existing result directories.

Usage
-----
    # Multi-run directories (produced by compare_implementations):
    python -m comparison_experiments.combine_results \\
        --dirs baseline_neo4j=experiment_results/results/run_1/baseline_neo4j \\
               multsampler=experiment_results/results/run_2/multsampler \\
        --output_dir experiment_results/results/combined \\
        --nbr_runs 3

    # Single-run directories (bare result folders with measurements.json directly inside):
    python -m comparison_experiments.combine_results \\
        --dirs experiment_results/report_files/baseline_neo4j/arxiv \\
               experiment_results/report_files/multsampler/arxiv \\
        --output_dir experiment_results/results/combined

Each entry in --dirs can be:
  - A bare PATH       — impl name is inferred from the directory name
  - An IMPL=PATH pair — impl name is taken from IMPL

Single-run directories (those containing ``measurements.json`` directly, without a
``run_0/`` sub-directory) are automatically wrapped in a temporary directory with a
``run_0`` symlink so the aggregation and plotting functions work transparently.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

# Ensure src/ is on the path when executed as a script.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comparison_experiments.implementation_comparison_plots import plot_all_comparisons


def _is_single_run_dir(path: Path) -> bool:
    """Return True if the directory is a bare single-run result dir."""
    return (path / "measurements.json").exists() and not (path / "run_0").exists()


def _wrap_single_run(path: Path, stack: ExitStack) -> Path:
    """Create a temp dir with a run_0 symlink pointing to *path* and return it."""
    tmp = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    (tmp / "run_0").symlink_to(path.resolve())
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate comparison plots from existing per-implementation result directories."
    )
    parser.add_argument(
        "--dirs", nargs="+", required=True,
        metavar="PATH_or_IMPL=PATH",
        help=(
            "One entry per implementation: either a bare PATH (impl name inferred from "
            "the directory name) or an IMPL=PATH pair. PATH can be a multi-run directory "
            "(containing run_0/, run_1/, …) or a single-run result directory. "
            "Example: experiment_results/report_files/baseline_neo4j/arxiv "
            "         baseline_neo4j=experiment_results/results/run_1/baseline_neo4j"
        ),
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory where the combined comparison plots will be written.",
    )
    parser.add_argument(
        "--nbr_runs", type=int, default=3,
        help="Number of runs per multi-run implementation to aggregate (default: 3). "
             "Single-run directories always use 1 run regardless of this value.",
    )
    args = parser.parse_args()

    raw_dirs: dict[str, Path] = {}
    for entry in args.dirs:
        if "=" in entry:
            name, path_str = entry.split("=", 1)
        else:
            p = Path(entry)
            name = p.name
            path_str = entry
        p = Path(path_str)
        if not p.exists():
            parser.error(f"Directory does not exist: {p}")
        raw_dirs[name] = p

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        impl_dirs: dict[str, Path] = {}
        for name, p in raw_dirs.items():
            if _is_single_run_dir(p):
                print(f"  {name}: single-run directory, wrapping as run_0")
                impl_dirs[name] = _wrap_single_run(p, stack)
            else:
                impl_dirs[name] = p

        print(f"Combining {len(impl_dirs)} implementation(s): {', '.join(impl_dirs)}")
        print(f"Output directory: {output_dir}")

        plot_all_comparisons(impl_dirs, output_dir, args.nbr_runs)

    print(f"Done. Plots written to: {output_dir}")


if __name__ == "__main__":
    main()
