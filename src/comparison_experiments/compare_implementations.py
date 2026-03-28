"""compare_implementations.py — run multiple implementations and compare results.

Usage
-----
    python -m comparison_experiments.compare_implementations \\
        --dataset cora \\
        --implementations baseline_neo4j multsampler baseline_pyg \\
        --nbr_runs 3

Each implementation is run ``nbr_runs`` times.  All results land under a shared
``experiment_results/results/run_N_YYYY-MM-DD/`` directory, with one sub-directory
per implementation that contains ``run_0/``, ``run_1/``, … sub-dirs inside it.
After all runs complete, comparison plots with every implementation on the same
axes are written to the parent directory.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on the path when executed as a script.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from training.Main import (
    load_configs,
    _build_common_kwargs,
    _create_multi_run_parent_dir,
    _make_graph_store,
    _make_sampler,
    _make_feature_store,
    _make_model,
    _run_standard,
    _run_in_memory,
    _run_saint_neo4j,
    _run_saint_pyg,
    _run_distributed,
    _log_config,
)
from benchmarking_tools import Measurer, QueryProfileAccumulator
from comparison_experiments.implementation_comparison_plots import plot_all_comparisons


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_one(
    dataset_cfg: dict,
    impl_cfg: dict,
    run_dir: Path,
    driver,
    common_kwargs: dict | None,
) -> None:
    """Execute one training run, writing results to *run_dir*."""
    profile = dataset_cfg.get("profile", False)
    in_memory = impl_cfg.get("in_memory", False)
    trainer_name = impl_cfg["trainer"]

    profile_accumulator = QueryProfileAccumulator() if profile else None
    measurer = Measurer(
        dataset_cfg,
        profile_accumulator=profile_accumulator,
        output_dir=run_dir,
    )
    model = _make_model(impl_cfg, dataset_cfg)

    if in_memory:
        if trainer_name == "GraphSAINTTrainer":
            _run_saint_pyg(dataset_cfg, impl_cfg, measurer, model)
        else:
            _run_in_memory(dataset_cfg, impl_cfg, measurer, model)
        return

    graph_store = _make_graph_store(impl_cfg, common_kwargs, driver, measurer, profile_accumulator)

    sampler = None
    if impl_cfg.get("sampler"):
        sampler = _make_sampler(impl_cfg, graph_store, dataset_cfg, measurer)

    feature_store = None
    if impl_cfg.get("feature_store"):
        feature_store = _make_feature_store(
            impl_cfg, common_kwargs, driver, measurer, profile_accumulator, sampler
        )

    _log_config(measurer, impl_cfg, dataset_cfg, sampler, model)

    if trainer_name == "GraphSAINTTrainer":
        _run_saint_neo4j(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, model)
    elif trainer_name == "DistributedTrainer":
        _run_distributed(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model)
    else:
        _run_standard(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple GNN implementations and produce comparison plots."
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (must match a file in src/configs/datasets/)",
    )
    parser.add_argument(
        "--implementations", nargs="+", required=True,
        metavar="IMPL",
        help="One or more implementation names (must each match a file in "
             "src/configs/implementations/), e.g. baseline_neo4j multsampler baseline_pyg",
    )
    parser.add_argument(
        "--nbr_runs", type=int, default=3,
        help="Number of independent training runs per implementation (default: 3)",
    )
    args = parser.parse_args()

    uri      = os.environ.get("URI", "")
    user     = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    parent_dir = _create_multi_run_parent_dir()
    print(f"Results directory: {parent_dir}")

    impl_dirs: dict[str, Path] = {}

    for impl_name in args.implementations:
        dataset_cfg, impl_cfg = load_configs(args.dataset, impl_name)
        # Store the name so _log_config can write it to the config result.
        impl_cfg["_name"] = impl_name

        impl_dir = parent_dir / impl_name
        impl_dir.mkdir(parents=True, exist_ok=True)
        impl_dirs[impl_name] = impl_dir

        in_memory = impl_cfg.get("in_memory", False)

        # One shared driver per implementation; fresh driver between implementations.
        driver = None
        common_kwargs = None
        if not in_memory:
            from Neo4jConnection import Neo4jConnection
            driver = Neo4jConnection(uri, user, password).get_driver()
            common_kwargs = _build_common_kwargs(dataset_cfg, uri, user, password)

        print(f"\n=== Implementation: {impl_name} ===")
        for run_i in range(args.nbr_runs):
            print(f"  Run {run_i + 1}/{args.nbr_runs}")
            run_dir = impl_dir / f"run_{run_i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _run_one(dataset_cfg, impl_cfg, run_dir, driver, common_kwargs)

        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    print("\nAll runs complete.  Generating comparison plots …")
    plot_all_comparisons(impl_dirs, parent_dir, args.nbr_runs)
    print(f"Done.  Results in: {parent_dir}")


if __name__ == "__main__":
    main()
