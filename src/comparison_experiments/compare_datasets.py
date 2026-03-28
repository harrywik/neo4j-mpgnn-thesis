"""compare_datasets.py — run one implementation over multiple datasets.

Usage
-----
    python -m comparison_experiments.compare_datasets \
        --implementation baseline_pyg \
        --datasets cora arxiv products \
        --nbr_runs 3

Runs one fixed implementation for each dataset, executes nbr_runs per dataset,
and writes combined plots across datasets for:
    - throughput
    - phase summary (sampling vs training)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmarking_tools import Measurer, QueryProfileAccumulator
from comparison_experiments.dataset_comparison_plots import plot_all_dataset_comparisons
from training.Main import (
    _build_common_kwargs,
    _create_multi_run_parent_dir,
    _log_config,
    _make_feature_store,
    _make_graph_store,
    _make_model,
    _make_sampler,
    _run_distributed,
    _run_in_memory,
    _run_saint_neo4j,
    _run_saint_pyg,
    _run_standard,
    load_configs,
)


def _run_one(
    dataset_cfg: dict,
    impl_cfg: dict,
    run_dir: Path,
    driver,
    common_kwargs: dict | None,
) -> None:
    """Execute one training run for a fixed dataset+implementation pair."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one implementation over multiple datasets and compare results."
    )
    parser.add_argument(
        "--implementation",
        required=True,
        help="Implementation name (must match src/configs/implementations/<name>.json)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        metavar="DATASET",
        help="One or more dataset names (must match src/configs/datasets/<name>.json)",
    )
    parser.add_argument(
        "--nbr_runs",
        type=int,
        default=3,
        help="Number of independent runs per dataset (default: 3)",
    )
    args = parser.parse_args()

    uri = os.environ.get("URI", "")
    user = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    parent_dir = _create_multi_run_parent_dir()
    print(f"Results directory: {parent_dir}")

    dataset_dirs: dict[str, Path] = {}

    for dataset_name in args.datasets:
        dataset_cfg, impl_cfg = load_configs(dataset_name, args.implementation)
        impl_cfg["_name"] = args.implementation

        dataset_dir = parent_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_dirs[dataset_name] = dataset_dir

        in_memory = impl_cfg.get("in_memory", False)

        driver = None
        common_kwargs = None
        if not in_memory:
            from Neo4jConnection import Neo4jConnection

            driver = Neo4jConnection(uri, user, password).get_driver()
            common_kwargs = _build_common_kwargs(dataset_cfg, uri, user, password)

        print(f"\n=== Dataset: {dataset_name} | implementation: {args.implementation} ===")
        for run_i in range(args.nbr_runs):
            print(f"  Run {run_i + 1}/{args.nbr_runs}")
            run_dir = dataset_dir / f"run_{run_i}"
            run_dir.mkdir(parents=True, exist_ok=True)
            _run_one(dataset_cfg, impl_cfg, run_dir, driver, common_kwargs)

        if driver is not None:
            try:
                driver.close()
            except Exception:
                pass

    print("\nAll runs complete. Generating dataset comparison plots ...")
    plot_all_dataset_comparisons(dataset_dirs, parent_dir, args.nbr_runs)
    print(f"Done. Results in: {parent_dir}")


if __name__ == "__main__":
    main()
