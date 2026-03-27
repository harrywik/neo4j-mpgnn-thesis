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
import shutil
import sys
from dataclasses import dataclass
from datetime import date
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
from neo4j_pyg.feature_stores import Neo4jNoCacheFS
from neo4j_pyg.graph_stores import Neo4SingleGS
from neo4j_pyg.samplers import (
    Neo4jEdgeModeSampler,
    Neo4jNeighborSampler,
    Neo4jSampler,
    Neo4jGraphSAINTRandomWalkSampler,
    OldNeighborSampler,
)
from Neo4jConnection import Neo4jConnection
from training.Training import Trainer, put_nodeLoader_args_map
from training.GraphSAINTTrainer import GraphSAINTTrainer
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


def _build_registry(profile: bool = False, config: dict | None = None) -> dict:
    """Build sampler registry. Optional ``config`` key for ``Neo4jEdgeModeSampler``:

    * ``neo4j_edge_mode`` — ``incoming`` | ``outgoing`` | ``undirected`` | ``induced``
    """
    cfg = config or {}
    edge_mode = cfg.get("neo4j_edge_mode", "incoming")

    return {
        "Neo4jSampler": RunSpec(
            "neo4j",
            lambda gs, nn, p=profile: Neo4jSampler(gs, num_neighbors=nn, rel_type="CITES", node_label="Paper", profile=p),
        ),
        "Neo4jEdgeModeSampler": RunSpec(
            "neo4j",
            lambda gs, nn, p=profile, em=edge_mode: Neo4jEdgeModeSampler(
                gs,
                num_neighbors=nn,
                edge_mode=em,
                rel_type="CITES",
                node_label="Paper",
                profile=p,
            ),
        ),
        "Neo4jNeighborSampler": RunSpec(
            "neo4j",
            lambda gs, nn, p=profile: Neo4jNeighborSampler(gs, num_neighbors=nn, profile=p),
        ),
        "Neo4jSamplerEquiv": RunSpec(
            "neo4j",
            lambda gs, nn, p=profile: Neo4jSamplerEquiv(
                gs, num_neighbors=nn, skip_neighbor_label=True, profile=p,
            ),
        ),
        "Neo4jSamplerFast": RunSpec(
            "neo4j",
            lambda gs, nn, p=profile: Neo4jSamplerFast(
                gs, num_neighbors=nn, skip_neighbor_label=True, profile=p,
            ),
        ),
        "OldNeighborSampler": RunSpec(
            "neo4j",
            lambda gs, nn: OldNeighborSampler(gs, num_neighbors=nn),
        ),
        "Neo4jGraphSAINTRandomWalkSampler": RunSpec("graphsaint_neo4j", None),
        "PyGGraphSAINTRandomWalkSampler": RunSpec("graphsaint_pyg", None),
        "PyGNeighborLoader": RunSpec("pyg", None),
    }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _next_comparison_dir(base: Path) -> Path:
    """Return the next auto-incremented comparison_N_YYYY-MM-DD directory."""
    existing = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and p.name.startswith("comparison_"):
                try:
                    existing.append(int(p.name.split("_")[1]))
                except (ValueError, IndexError):
                    pass
    next_id = (max(existing) + 1) if existing else 0
    date_str = date.today().isoformat()
    out = base / f"comparison_{next_id}_{date_str}"
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
    profile: bool = config.get("profile", False)

    SAMPLER_REGISTRY = _build_registry(profile=profile, config=config)

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
        SAMPLER_REGISTRY[n].data_key in ("neo4j", "graphsaint_neo4j") for n in sampler_names
        if n in SAMPLER_REGISTRY
    )
    needs_pyg = any(
        SAMPLER_REGISTRY[n].data_key in ("pyg", "graphsaint_pyg") for n in sampler_names
        if n in SAMPLER_REGISTRY
    )

    if needs_neo4j:
        uri = os.environ["URI"]
        user = os.environ["USERNAME"]
        password = os.environ["PASSWORD"]
        driver = Neo4jConnection(uri, user, password).get_driver()

        neo4j_feature_store = Neo4jNoCacheFS(
            driver,
            database_name="neo4j",
            dataset_name="cora",
            feature_property="embedding_bytes",
            nodeid_property="id",
            split_property_name="split",
            split_property_type="str",
            target_property="subject",
            feature_property_type="byte[]",
        )
        neo4j_graph_store = Neo4SingleGS(
            driver,
            database_name="neo4j",
            dataset_name="cora",
            feature_property="embedding_bytes",
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

            run_dir = experiment_dir / sampler_name / f"run_{run_idx}_{date.today().isoformat()}"
            measurer = ExperimentMeasurer(run_dir, config)
            measurer.write_to_configresult("sampler", sampler_name)
            measurer.write_to_configresult("run_idx", run_idx)
            measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
            measurer.write_to_configresult("num_neighbors", num_neighbors)
            # Attach the per-run measurer to the shared graph/feature stores so
            # that Neo4SingleGS.sample_from_nodes and Neo4jNoCacheFS._get_tensor
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
                    max_validation_size=config.get("max_validation_size"),
                    max_test_size=config.get("max_test_size"),
                )
            elif spec.data_key == "graphsaint_neo4j":
                train_loader = Neo4jGraphSAINTRandomWalkSampler(
                    graph_store=neo4j_graph_store,
                    feature_store=neo4j_feature_store,
                    batch_size=config["graphsaint_batch_size"],
                    walk_length=config["walk_length"],
                    num_steps=config["num_steps"],
                    sample_coverage=config["sample_coverage"],
                    node_label="Paper",
                    measurer=measurer,
                    profile=profile,
                    # Cache norms in the experiment dir so they are computed
                    # once and reused across all runs of this comparison.
                    save_dir=str(experiment_dir),
                )
                eval_sampler = Neo4jNeighborSampler(
                    neo4j_graph_store,
                    num_neighbors=num_neighbors,
                    node_label="Paper",
                    profile=profile,
                )
                trainer = GraphSAINTTrainer(
                    model=model,
                    data=(neo4j_feature_store, neo4j_graph_store),
                    train_loader=train_loader,
                    eval_sampler=eval_sampler,
                    patience=config.get("patience"),
                    min_delta=config.get("min_delta"),
                    batch_size=config["graphsaint_batch_size"],
                    lr=config.get("lr"),
                    measurer=measurer,
                )
            elif spec.data_key == "graphsaint_pyg":
                trainer = GraphSAINTTrainer(
                    model=model,
                    data=pyg_graph,
                    walk_length=config["walk_length"],
                    num_steps=config["num_steps"],
                    sample_coverage=config["sample_coverage"],
                    patience=config.get("patience"),
                    min_delta=config.get("min_delta"),
                    batch_size=config["graphsaint_batch_size"],
                    lr=config.get("lr"),
                    measurer=measurer,
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
                    nodeloader_args=nodeloader_args,
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

    # Remove per-run data directories — the aggregated plots are all we need.
    for sampler_name in sampler_names:
        sampler_dir = experiment_dir / sampler_name
        if sampler_dir.exists():
            shutil.rmtree(sampler_dir)
            print(f"  Removed run data: {sampler_dir.name}/")

    print(f"Done. Results in: {experiment_dir}")


if __name__ == "__main__":
    main()
