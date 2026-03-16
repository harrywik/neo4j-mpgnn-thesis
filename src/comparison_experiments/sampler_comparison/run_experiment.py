"""Sampler comparison experiment.

Runs each sampler listed in config.json for ``num_runs`` independent trials
using an identical model, dataset, and hyper-parameters.  Results are written
to ``experiment_results/sampler_comparison/comparison_N/`` and comparison
plots are generated after all runs complete.

Usage (from repo root):
    python src/experiments/sampler_comparison/run_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from dotenv import load_dotenv
from torch_geometric.datasets import Planetoid

# ---------------------------------------------------------------------------
# Path bootstrap — allow running as a script or via make
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neo4j_pyg.models import GCN
from neo4j_pyg.feature_stores import NoCacheFeatureStore
from neo4j_pyg.graph_stores import BaseLineGS
from neo4j_pyg.samplers import (
    Neo4jNeighborSampler,
    OldNeighborSampler,
)
from Neo4jConnection import Neo4jConnection
from training.Training import Trainer, put_nodeLoader_args_map
from comparison_experiments.sampler_comparison.experiment_measurer import ExperimentMeasurer
from comparison_experiments.sampler_comparison.comparison_plots import plot_comparison

load_dotenv()

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Describes how to build one sampler run."""
    data_key: str                          # "neo4j" or "pyg"
    make_sampler: Optional[Callable]       # None for the PyG path


SAMPLER_REGISTRY: dict[str, RunSpec] = {
    "Neo4jNeighborSampler": RunSpec(
        "neo4j",
        lambda gs, nn: Neo4jNeighborSampler(gs, num_neighbors=nn),
    ),
    "OldNeighborSampler": RunSpec(
        "neo4j",
        lambda gs, nn: OldNeighborSampler(gs, num_neighbors=nn),
    ),
    "PyGEquivalentSampler": RunSpec(
        "neo4j",
        lambda gs, nn: PyGEquivalentSampler(gs, num_neighbors=nn),
    ),
    "PyGNeighborLoader": RunSpec("pyg", None),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_comparison_dir(base: Path) -> Path:
    """Return the next auto-incremented comparison_N directory."""
    existing = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("comparison_"):
                try:
                    existing.append(int(p.name.split("_", 1)[1]))
                except ValueError:
                    pass
    next_id = (max(existing) + 1) if existing else 0
    out = base / f"comparison_{next_id}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _make_model(model_args: dict) -> GCN:
    return GCN(**model_args)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config_path = Path(__file__).parent / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    num_runs: int = config["num_runs"]
    max_epochs: int = config["max_epochs"]
    batch_size: int = config["batch_size"]
    num_neighbors: list[int] = config["num_neighbors"]
    sampler_names: list[str] = config["samplers"]
    drop_last: bool = config.get("drop_last")

    model_args = {
        "in_dim": 1433,
        "hidden_dim1": config.get("hidden_dim1"),
        "hidden_dim2": config.get("hidden_dim2"),
        "nbr_classes": 7,
        "init_weights": config.get("initialise_weights"),
    }

    # Resolve output directory
    repo_root = SRC_DIR.parent
    experiment_dir = _next_comparison_dir(
        repo_root / "experiment_results" / "sampler_comparison"
    )
    print(f"Results will be written to: {experiment_dir}")

    # ------------------------------------------------------------------
    # One-time data setup
    # ------------------------------------------------------------------
    neo4j_feature_store = None
    neo4j_graph_store = None
    pyg_graph = None
    pyg_train_indices = None

    needs_neo4j = any(
        SAMPLER_REGISTRY[n].data_key == "neo4j" for n in sampler_names
        if n in SAMPLER_REGISTRY
    )
    needs_pyg = any(
        SAMPLER_REGISTRY[n].data_key == "pyg" for n in sampler_names
        if n in SAMPLER_REGISTRY
    )

    if needs_neo4j:
        uri = os.environ["URI"]
        user = os.environ["USERNAME"]
        password = os.environ["PASSWORD"]
        driver = Neo4jConnection(uri, user, password).get_driver()

        neo4j_feature_store = NoCacheFeatureStore(
            driver,
            database_name="neo4j",
            dataset_name="cora",
            feature_property="embedding",
            nodeid_property="id",
            split_property_name="split",
            split_property_type="str",
            target_property="subject",
            feature_property_type="byte[]",
        )
        neo4j_graph_store = BaseLineGS(
            driver,
            database_name="neo4j",
            dataset_name="cora",
            feature_property="embedding",
            nodeid_property="id",
            split_property_name="split",
            split_property_type="str",
            target_property="subject",
        )

    if needs_pyg:
        dataset = Planetoid(root="data/Planetoid", name="Cora")
        pyg_graph = dataset[0]
        pyg_train_indices = torch.where(pyg_graph.train_mask.reshape(-1))[0]

    # ------------------------------------------------------------------
    # Orchestration loop
    # ------------------------------------------------------------------
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=config.get("shuffle"),
    )

    for sampler_name in sampler_names:
        if sampler_name not in SAMPLER_REGISTRY:
            print(f"WARNING: {sampler_name!r} not in SAMPLER_REGISTRY — skipping.")
            continue

        spec = SAMPLER_REGISTRY[sampler_name]
        print(f"\n=== Sampler: {sampler_name} ===")

        for run_idx in range(num_runs):
            print(f"  Run {run_idx + 1}/{num_runs} ...", flush=True)

            run_dir = experiment_dir / sampler_name / f"run_{run_idx}"
            measurer = ExperimentMeasurer(run_dir, config)
            measurer.write_to_configresult("sampler", sampler_name)
            measurer.write_to_configresult("run_idx", run_idx)
            measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
            measurer.write_to_configresult("num_neighbors", num_neighbors)
            # Attach the per-run measurer to the shared graph/feature stores so
            # that BaseLineGS.sample_from_nodes and NoCacheFeatureStore._get_tensor
            # can log sub-phase timings without holding a direct reference.
            if neo4j_graph_store is not None:
                neo4j_graph_store.measurer = measurer
            if neo4j_feature_store is not None:
                neo4j_feature_store.measurer = measurer
                # Re-log the RTT baseline measured at feature store construction.
                rtt = getattr(neo4j_feature_store, "_network_baseline_ms", None)
                if rtt is not None:
                    measurer.log_event("network_baseline_ms", rtt)

            model = _make_model(model_args)

            if spec.data_key == "neo4j":
                sampler = spec.make_sampler(neo4j_graph_store, num_neighbors)
                trainer = Trainer(
                    model=model,
                    data=(neo4j_feature_store, neo4j_graph_store),
                    sampler=sampler,
                    measurer=measurer,
                    patience=config.get("patience"),
                    min_delta=config.get("min_delta"),
                    batch_size=batch_size,
                    lr=config.get("lr"),
                    nodeloader_args=nodeloader_args,
                    drop_last=drop_last,
                )
            else:  # "pyg"
                trainer = Trainer(
                    model=model,
                    data=pyg_graph,
                    train_indices=pyg_train_indices,
                    num_neighbors=num_neighbors,
                    measurer=measurer,
                    patience=config.get("patience"),
                    min_delta=config.get("min_delta"),
                    batch_size=batch_size,
                    lr=config.get("lr"),
                    drop_last=drop_last,
                )

            trainer.train(max_epochs=max_epochs)
            # Trainer.train() calls measurer.close() + summarize() internally,
            # writing per-run plots and measurements.json to run_dir.

            # Clear the per-run measurer from shared stores to avoid stale refs.
            if neo4j_graph_store is not None:
                neo4j_graph_store.measurer = None
            if neo4j_feature_store is not None:
                neo4j_feature_store.measurer = None

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------
    print("\nGenerating comparison plots ...")
    plot_comparison(experiment_dir, sampler_names, num_runs)
    print(f"Done. Results in: {experiment_dir}")


if __name__ == "__main__":
    main()
