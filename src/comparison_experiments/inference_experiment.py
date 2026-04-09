"""inference_experiment.py — compare GNN inference strategies across N-node scales.

Trains (or loads) a model, then benchmarks three inference strategies for each
value of N (number of seed nodes to run inference on):

  full_graph          — load entire graph via PyG Planetoid once; NeighborLoader subgraph sampling
  neighborhood_sampling — Neo4j Cypher-fetch k-hop subgraphs per batch; Python forward pass
  in_db_java          — export spec/weights; call gnnProcedures.inference.run inside Neo4j

All three strategies use the same fanout ([num_neighbors]*num_hops from inference config).
full_graph includes the one-time dataset load time in every per-N timing.

Each (strategy, N) cell is repeated ``nbr_runs`` times; results are averaged and
reported as mean ± 95 % CI.

Usage
-----
    python -m comparison_experiments.inference_experiment \\
        --dataset src/configs/inference/datasets/cora.json \\
        --model   src/configs/inference/models/gcn.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
import tracemalloc
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader, NodeLoader

from Neo4jConnection import Neo4jConnection
from benchmarking_tools import Measurer
from comparison_experiments.inference_experiment_plots import plot_all
from neo4j_pyg.neo4j_model_interface.create_inference_spec import (
    create_inference_spec,
    validate_model_for_db_inference,
)
from neo4j_pyg.samplers.Neo4jSampler import Neo4jSampler
from training.Main import (
    _build_common_kwargs,
    _make_feature_store,
    _make_graph_store,
    _make_model,
    _make_sampler,
)
from training.Training import Trainer, put_nodeLoader_args_map

load_dotenv()

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = SRC_DIR / "configs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_all_configs(dataset_path: str, model_path: str) -> dict:
    """Load and merge inference + training configs into a single dict."""
    ds = _load_json(Path(dataset_path))
    mdl = _load_json(Path(model_path))

    train_ds = _load_json(
        CONFIGS_DIR / "training" / "datasets" / f"{ds['training_dataset_config']}.json"
    )
    train_impl = _load_json(
        CONFIGS_DIR / "training" / "implementations" / f"{ds['training_implementation_config']}.json"
    )

    _INFERENCE_ONLY_KEYS = {
        "training_dataset_config", "training_implementation_config",
        "node_counts", "nbr_runs", "inference_batch_size", "max_neighbors",
        "num_neighbors", "num_hops", "planetoid_root", "planetoid_name",
    }
    merged_train_ds = {**train_ds, **{k: v for k, v in ds.items() if k not in _INFERENCE_ONLY_KEYS}}

    if "hidden_dim1" in mdl:
        merged_train_ds["default_hidden_dim1"] = mdl["hidden_dim1"]
    if "hidden_dim2" in mdl:
        merged_train_ds["default_hidden_dim2"] = mdl["hidden_dim2"]

    return {
        "dataset": ds,
        "model": mdl,
        "train_ds": merged_train_ds,
        "train_impl": train_impl,
    }


# ---------------------------------------------------------------------------
# Component construction (delegates to Main.py factories)
# ---------------------------------------------------------------------------

def build_components(cfg: dict, driver):
    uri      = os.environ.get("URI", "")
    user     = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    common_kwargs = _build_common_kwargs(cfg["train_ds"], uri, user, password)
    graph_store = _make_graph_store(cfg["train_impl"], common_kwargs, driver, None, None)
    sampler = _make_sampler(cfg["train_impl"], graph_store, cfg["train_ds"], None) \
        if cfg["train_impl"].get("sampler") else None
    feature_store = _make_feature_store(
        cfg["train_impl"], common_kwargs, driver, None, None, sampler
    ) if cfg["train_impl"].get("feature_store") else None
    model = _make_model(cfg["train_impl"], cfg["train_ds"])
    return model, graph_store, feature_store, sampler


# ---------------------------------------------------------------------------
# Training / checkpoint loading
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "inference"


def _checkpoint_path(cfg: dict) -> Path:
    """Deterministic checkpoint path based on dataset + model config."""
    ds_name = cfg["dataset"].get("dataset_name", "dataset")
    model_class = cfg["model"].get("model_class", "model")
    return _CHECKPOINT_DIR / f"{ds_name}_{model_class}.pt"


def train_or_load(model, cfg: dict, graph_store, feature_store, sampler) -> None:
    # 1. Explicit checkpoint in config takes priority
    explicit = cfg["model"].get("checkpoint_path")
    if explicit:
        state = torch.load(explicit, map_location="cpu")
        model.load_state_dict(state["MODEL_STATE"])
        print(f"Loaded checkpoint from {explicit}")
        return

    # 2. Auto-saved checkpoint from a previous inference experiment run
    ckpt = _checkpoint_path(cfg)
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["MODEL_STATE"])
        print(f"Loaded cached checkpoint from {ckpt}")
        return

    # 3. No checkpoint found — train and save
    train_ds = cfg["train_ds"]
    train_impl = cfg["train_impl"]

    nla = train_impl.get("nodeloader_args", {})
    nodeloader_args = put_nodeLoader_args_map(**nla) if nla else put_nodeLoader_args_map()

    measurer = Measurer(train_ds)
    trainer = Trainer(
        model=model,
        data=(feature_store, graph_store),
        sampler=sampler,
        measurer=measurer,
        lr=train_ds["lr"],
        patience=train_ds["patience"],
        min_delta=train_ds["min_delta"],
        batch_size=train_ds["batch_size"],
        nodeloader_args=nodeloader_args,
        max_training_size=train_ds.get("max_training_size"),
        max_validation_size=train_ds.get("max_validation_size"),
        max_test_size=train_ds.get("max_test_size"),
    )
    trainer.train(max_epochs=train_ds["max_epochs"])

    # Save for next run
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"MODEL_STATE": model.state_dict()}, ckpt)
    print(f"Saved checkpoint to {ckpt}")


# ---------------------------------------------------------------------------
# Ground-truth label fetching (Neo4j)
# ---------------------------------------------------------------------------

def fetch_labels(node_ids: list[int], cfg: dict, driver) -> dict[int, int]:
    """Return {node_id: label} for the given IDs."""
    ds = cfg["train_ds"]
    query = (
        f"UNWIND $ids AS nid "
        f"MATCH (n:{ds['node_label']} {{{ds['nodeid_property']}: nid}}) "
        f"RETURN n.{ds['nodeid_property']} AS id, n.{ds['target_property']} AS label"
    )
    with driver.session(database=ds["database_name"], fetch_size=-1) as session:
        result = session.run(query, ids=node_ids)
        return {r["id"]: int(r["label"]) for r in result}


# ---------------------------------------------------------------------------
# PyG full-graph loader (Planetoid)
# ---------------------------------------------------------------------------

class PyGGraphLoader:
    """Loads a Planetoid dataset once and exposes test-node indices."""

    def __init__(self, root: str, name: str):
        self.root = root
        self.name = name
        self._data = None
        self.load_time_s: float = 0.0

    def load(self) -> None:
        t0 = time.monotonic()
        dataset = Planetoid(root=self.root, name=self.name)
        self._data = dataset[0]
        self.load_time_s = time.monotonic() - t0
        print(
            f"  [full_graph] Loaded {self._data.num_nodes} nodes, "
            f"{self._data.num_edges} edges in {self.load_time_s:.2f}s"
        )

    @property
    def data(self):
        return self._data

    def test_node_indices(self) -> list[int]:
        mask = self._data.test_mask
        return mask.nonzero(as_tuple=False).squeeze(1).tolist()


# ---------------------------------------------------------------------------
# Strategy: full_graph  (PyG Planetoid + NeighborLoader)
# ---------------------------------------------------------------------------

def run_full_graph_pyg(
    model: torch.nn.Module,
    N: int,
    run_i: int,
    pyg_loader: PyGGraphLoader,
    num_neighbors: list[int],
    batch_size: int,
) -> tuple[dict[int, int], dict[int, int], dict[str, Any]]:
    """Run inference via PyG NeighborLoader on Planetoid data.

    Returns
    -------
    preds   : {local_node_idx → predicted_class}
    labels  : {local_node_idx → true_class}  (from Planetoid)
    metrics : timing / memory / batch-latency metrics
              (load_time_s is NOT included here; caller adds it)
    """
    data = pyg_loader.data
    test_indices = pyg_loader.test_node_indices()
    sample_size = min(N, len(test_indices))
    rng = random.Random(run_i * 10007 + N)
    seed_ids = rng.sample(test_indices, sample_size)

    input_nodes = torch.tensor(seed_ids, dtype=torch.long)

    tracemalloc.start()
    t0 = time.monotonic()

    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=False,
    )

    device = next(model.parameters()).device
    model.eval()
    preds: dict[int, int] = {}
    labels_out: dict[int, int] = {}
    batch_latencies: list[float] = []

    with torch.no_grad():
        for batch in loader:
            t_batch = time.monotonic()
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            # Seed nodes are the first batch.batch_size entries
            n = batch.batch_size
            seed_global_ids = batch.n_id[:n].cpu().tolist()
            seed_preds = out[:n].argmax(dim=1).cpu().tolist()
            seed_labels = batch.y[:n].cpu().tolist()
            for nid, pred, label in zip(seed_global_ids, seed_preds, seed_labels):
                preds[nid] = pred
                labels_out[nid] = label
            batch_latencies.append((time.monotonic() - t_batch) * 1000)

    elapsed = time.monotonic() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = np.array(batch_latencies)
    metrics = {
        "total_time_s": elapsed,
        "ms_per_node": elapsed * 1000 / max(len(preds), 1),
        "throughput_nodes_per_s": len(preds) / max(elapsed, 1e-9),
        "peak_memory_mb": peak / 1024 / 1024,
        "p50_batch_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95_batch_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "n_batches": len(batch_latencies),
    }
    return preds, labels_out, metrics


# ---------------------------------------------------------------------------
# Strategy: neighborhood_sampling  (Neo4j + PyG NodeLoader)
# ---------------------------------------------------------------------------

def run_neighborhood_sampling(
    model: torch.nn.Module,
    test_ids: list[int],
    cfg: dict,
    feature_store,
    graph_store,
    inference_sampler,
) -> tuple[dict[int, int], dict[str, Any]]:
    batch_size = cfg["dataset"].get("inference_batch_size", 256)
    tracemalloc.start()
    t_total_start = time.monotonic()

    device = next(model.parameters()).device
    model.eval()
    preds: dict[int, int] = {}
    batch_latencies: list[float] = []

    input_nodes = torch.tensor(test_ids, dtype=torch.int64)
    loader = NodeLoader(
        data=(feature_store, graph_store),
        node_sampler=inference_sampler,
        input_nodes=input_nodes,
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for batch in loader:
            t_batch = time.monotonic()
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            seed_mask = torch.isin(batch.n_id, batch.input_id)
            seed_preds = out[seed_mask].argmax(dim=1).cpu().tolist()
            seed_ids_local = batch.n_id[seed_mask].cpu().tolist()
            for nid, pred in zip(seed_ids_local, seed_preds):
                preds[nid] = pred
            batch_latencies.append((time.monotonic() - t_batch) * 1000)

    elapsed = time.monotonic() - t_total_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = np.array(batch_latencies)
    metrics = {
        "total_time_s": elapsed,
        "ms_per_node": elapsed * 1000 / max(len(test_ids), 1),
        "throughput_nodes_per_s": len(test_ids) / max(elapsed, 1e-9),
        "peak_memory_mb": peak / 1024 / 1024,
        "p50_batch_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95_batch_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "n_batches": len(batch_latencies),
    }
    return preds, metrics


# ---------------------------------------------------------------------------
# Strategy: in_db_java
# ---------------------------------------------------------------------------

def run_in_db_java(
    model: torch.nn.Module,
    test_ids: list[int],
    cfg: dict,
    driver,
    *,
    spec_exported: bool,
) -> tuple[dict[int, int], dict[str, Any], bool]:
    """Returns (preds, metrics, spec_exported_flag)."""
    ds = cfg["dataset"]
    mdl = cfg["model"]
    gnn_model_dir = os.environ.get("NEO4J_GNN_MODEL_DIR", "")
    model_name = mdl.get("model_name", "experiment_gcn")

    if not spec_exported:
        if not gnn_model_dir:
            raise RuntimeError(
                "NEO4J_GNN_MODEL_DIR must be set to use the in_db_java strategy."
            )
        from neo4j_pyg.neo4j_model_interface.create_inference_spec import create_inference_spec
        # Use num_neighbors from inference config for Java procedure max_neighbors
        num_neighbors = ds.get("num_neighbors", ds.get("max_neighbors", 10))
        create_inference_spec(
            model,
            model_name,
            base_dir=gnn_model_dir,
            max_neighbors=num_neighbors,
        )
        spec_exported = True

    batch_size = ds.get("inference_batch_size", 256)
    node_label = ds["node_label"]
    edge_type = ds.get("edge_type", "")
    feature_prop = ds["feature_property"]
    feature_type = ds["feature_property_type"]
    nodeid_prop = ds["nodeid_property"]
    db = cfg["train_ds"]["database_name"]
    max_neighbors = ds.get("num_neighbors", ds.get("max_neighbors", 10))

    tracemalloc.start()
    t_total_start = time.monotonic()

    preds: dict[int, int] = {}
    batch_latencies: list[float] = []

    for i in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[i : i + batch_size]
        t_batch = time.monotonic()
        query = (
            "CALL gnnProcedures.inference.run("
            "$seedIds, $nodeIdKey, $featureKey, $featureType, "
            "$nodeLabel, $modelName, $edgeType, $maxNeighbors"
            ") YIELD nodeId, predictedClass"
        )
        with driver.session(database=db, fetch_size=-1) as session:
            result = session.run(
                query,
                seedIds=batch_ids,
                nodeIdKey=nodeid_prop,
                featureKey=feature_prop,
                featureType=feature_type,
                nodeLabel=node_label,
                modelName=model_name,
                edgeType=edge_type,
                maxNeighbors=max_neighbors,
            )
            for r in result:
                preds[r["nodeId"]] = int(r["predictedClass"])
        batch_latencies.append((time.monotonic() - t_batch) * 1000)

    elapsed = time.monotonic() - t_total_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = np.array(batch_latencies)
    metrics = {
        "total_time_s": elapsed,
        "ms_per_node": elapsed * 1000 / max(len(test_ids), 1),
        "throughput_nodes_per_s": len(test_ids) / max(elapsed, 1e-9),
        "peak_memory_mb": peak / 1024 / 1024,
        "p50_batch_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95_batch_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "n_batches": len(batch_latencies),
    }
    return preds, metrics, spec_exported


# ---------------------------------------------------------------------------
# Subgraph verification
# ---------------------------------------------------------------------------

def verify_subgraph_match(
    seed_ids: list[int],
    cfg: dict,
    driver,
) -> dict:
    """Assert that ``neighbor.sample`` and ``inference.run`` sample the same subgraph.

    Calls both Java procedures with the same seeds and parameters, then
    compares the returned ``ordered_nodes`` and ``edge_pairs``.  Raises
    ``AssertionError`` with a diff summary if they disagree.

    Returns a dict with verification details for inclusion in results JSON.
    """
    ds = cfg["dataset"]
    mdl = cfg["model"]
    node_label = ds["node_label"]
    edge_type = ds.get("edge_type", "")
    nodeid_prop = ds["nodeid_property"]
    feature_prop = ds["feature_property"]
    feature_type = ds["feature_property_type"]
    max_neighbors = ds.get("num_neighbors", ds.get("max_neighbors", 10))
    num_hops = ds.get("num_hops", 2)
    model_name = mdl.get("model_name", "experiment_gcn")
    db = cfg["train_ds"]["database_name"]

    # -- 1. Call neighbor.sample ------------------------------------------------
    sample_query = (
        "CALL gnnProcedures.sampling.neighbor.sample("
        "$seedIds, $nodeIdKey, $nodeLabel, $numNeighbors, $edgeType, $randomSeed"
        ") YIELD ordered_nodes, edge_pairs"
    )
    with driver.session(database=db) as session:
        rec = session.run(
            sample_query,
            seedIds=seed_ids,
            nodeIdKey=nodeid_prop,
            nodeLabel=node_label,
            numNeighbors=[max_neighbors] * num_hops,
            edgeType=edge_type,
            randomSeed=42,
        ).single()
        sample_nodes = list(rec["ordered_nodes"])
        sample_edges = [list(e) for e in rec["edge_pairs"]]

    # -- 2. Call inference.run and extract subgraph from first row ---------------
    infer_query = (
        "CALL gnnProcedures.inference.run("
        "$seedIds, $nodeIdKey, $featureKey, $featureType, "
        "$nodeLabel, $modelName, $edgeType, $maxNeighbors"
        ") YIELD ordered_nodes, edge_pairs"
    )
    with driver.session(database=db) as session:
        rec = session.run(
            infer_query,
            seedIds=seed_ids,
            nodeIdKey=nodeid_prop,
            featureKey=feature_prop,
            featureType=feature_type,
            nodeLabel=node_label,
            modelName=model_name,
            edgeType=edge_type,
            maxNeighbors=max_neighbors,
        ).single()
        infer_nodes = list(rec["ordered_nodes"])
        infer_edges = [list(e) for e in rec["edge_pairs"]]

    # -- 3. Compare (sort edges for order-insensitive check) --------------------
    node_match = sample_nodes == infer_nodes
    edge_match = sorted(sample_edges) == sorted(infer_edges)

    verification = {
        "seed_ids": seed_ids,
        "nodes_match": node_match,
        "edges_match": edge_match,
        "num_nodes": len(sample_nodes),
        "num_edges": len(sample_edges),
    }

    if node_match and edge_match:
        print(
            f"  ✓ Subgraph verification passed — "
            f"{len(sample_nodes)} nodes, {len(sample_edges)} edges identical"
        )
        return verification

    lines = ["Subgraph mismatch between neighbor.sample and inference.run!"]
    if not node_match:
        s_set, i_set = set(sample_nodes), set(infer_nodes)
        lines.append(f"  Nodes: sample={len(sample_nodes)}, infer={len(infer_nodes)}")
        only_sample = s_set - i_set
        only_infer = i_set - s_set
        if only_sample:
            lines.append(f"  Only in sample: {sorted(only_sample)[:20]}")
        if only_infer:
            lines.append(f"  Only in infer:  {sorted(only_infer)[:20]}")
    if not edge_match:
        s_set = set(map(tuple, sample_edges))
        i_set = set(map(tuple, infer_edges))
        lines.append(f"  Edges: sample={len(sample_edges)}, infer={len(infer_edges)}")
        only_sample_e = s_set - i_set
        only_infer_e = i_set - s_set
        if only_sample_e:
            lines.append(f"  Only in sample: {sorted(only_sample_e)[:20]}")
        if only_infer_e:
            lines.append(f"  Only in infer:  {sorted(only_infer_e)[:20]}")
    raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(preds: dict[int, int], labels: dict[int, int]) -> float:
    correct = sum(1 for nid, pred in preds.items() if labels.get(nid) == pred)
    total = len(preds)
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _ci95(values: list[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / (n ** 0.5)


def _agg(run_metrics: list[dict]) -> dict:
    """Aggregate a list of per-run metric dicts into mean ± ci95."""
    if not run_metrics:
        return {}
    keys = run_metrics[0].keys()
    result = {}
    for k in keys:
        vals = [m[k] for m in run_metrics if m[k] is not None]
        if not vals:
            result[k] = {"mean": None, "ci95": None}
        else:
            result[k] = {"mean": float(np.mean(vals)), "ci95": _ci95(vals)}
    return result


# ---------------------------------------------------------------------------
# Pretty-print table
# ---------------------------------------------------------------------------

def _fmt(mean, ci95, *, pct=False, decimals=2) -> str:
    if mean is None:
        return "—"
    suffix = "%" if pct else ""
    v = mean * 100 if pct else mean
    c = ci95 * 100 if pct else ci95
    if c == 0:
        return f"{v:.{decimals}f}{suffix}"
    return f"{v:.{decimals}f} ±{c:.{decimals}f}{suffix}"


def print_table(N: int, nbr_runs: int, results: dict[str, dict]) -> None:
    header = f"\nN = {N} nodes  ({nbr_runs} runs)"
    print(header)
    col_w = [22, 14, 12, 12, 10, 12, 12]
    cols = ["Strategy", "Accuracy", "ms/node", "Nodes/s", "Mem(MB)", "P50(ms)", "P95(ms)"]
    fmt_row = "".join(f"{{:<{w}}}" for w in col_w)
    sep = "─" * sum(col_w)
    print(fmt_row.format(*cols))
    print(sep)
    for strategy, agg in results.items():
        acc  = agg.get("accuracy", {})
        ms   = agg.get("ms_per_node", {})
        thr  = agg.get("throughput_nodes_per_s", {})
        mem  = agg.get("peak_memory_mb", {})
        p50  = agg.get("p50_batch_ms", {})
        p95  = agg.get("p95_batch_ms", {})
        row = [
            strategy,
            _fmt(acc.get("mean"), acc.get("ci95"), pct=True),
            _fmt(ms.get("mean"), ms.get("ci95"), decimals=2),
            _fmt(thr.get("mean"), thr.get("ci95"), decimals=1),
            _fmt(mem.get("mean"), mem.get("ci95"), decimals=1),
            _fmt(p50.get("mean"), p50.get("ci95"), decimals=1),
            _fmt(p95.get("mean"), p95.get("ci95"), decimals=1),
        ]
        print(fmt_row.format(*row))


def print_scaling_summary(all_results: dict[int, dict[str, dict]]) -> None:
    """Print throughput-vs-N table for each strategy."""
    strategies = list(next(iter(all_results.values())).keys())
    node_counts = sorted(all_results.keys())

    print("\n\n=== Throughput scaling (nodes/s, mean) ===")
    col_w = [12] + [20] * len(strategies)
    header = ["N"] + strategies
    fmt_row = "".join(f"{{:<{w}}}" for w in col_w)
    sep = "─" * sum(col_w)
    print(fmt_row.format(*header))
    print(sep)
    for N in node_counts:
        row = [str(N)]
        for s in strategies:
            thr = all_results[N].get(s, {}).get("throughput_nodes_per_s", {})
            row.append(_fmt(thr.get("mean"), thr.get("ci95"), decimals=1))
        print(fmt_row.format(*row))


# ---------------------------------------------------------------------------
# Build inference-specific Neo4j sampler
# ---------------------------------------------------------------------------

def _build_inference_sampler(cfg: dict, graph_store):
    """Create a Neo4jSampler with the inference fanout from config."""
    ds = cfg["dataset"]
    num_neighbors_val = ds.get("num_neighbors", ds.get("max_neighbors", 10))
    num_hops = ds.get("num_hops", 2)
    num_neighbors = [num_neighbors_val] * num_hops
    node_label = ds.get("node_label", cfg["train_ds"].get("node_label", ""))
    edge_type = ds.get("edge_type", cfg["train_ds"].get("edge_type", ""))
    return Neo4jSampler(
        graph_store=graph_store,
        num_neighbors=num_neighbors,
        node_label=node_label,
        rel_type=edge_type if edge_type else None,
    )


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict, model, graph_store, feature_store, driver) -> dict:
    ds = cfg["dataset"]
    mdl = cfg["model"]
    node_counts: list[int] = ds["node_counts"]
    _nbr_runs_cfg = ds.get("nbr_runs", 5)
    if isinstance(_nbr_runs_cfg, list):
        nbr_runs_list: list[int] = _nbr_runs_cfg
    else:
        nbr_runs_list = [int(_nbr_runs_cfg)] * len(node_counts)
    strategies: list[str] = mdl.get("strategies", ["full_graph", "neighborhood_sampling", "in_db_java"])
    batch_size: int = ds.get("inference_batch_size", 256)
    num_neighbors_val: int = ds.get("num_neighbors", ds.get("max_neighbors", 10))
    num_hops: int = ds.get("num_hops", 2)
    num_neighbors: list[int] = [num_neighbors_val] * num_hops

    # --- Build inference sampler for neighborhood_sampling ---
    inference_sampler = None
    if "neighborhood_sampling" in strategies:
        inference_sampler = _build_inference_sampler(cfg, graph_store)
        print(f"Inference sampler fanout: {num_neighbors}")

    # --- Load PyG Planetoid dataset once if needed ---
    pyg_loader: PyGGraphLoader | None = None
    if "full_graph" in strategies:
        planetoid_root = ds.get("planetoid_root", "data/Planetoid")
        planetoid_name = ds.get("planetoid_name", "Cora")
        print(f"\nLoading {planetoid_name} via PyG Planetoid (one-time cost)...")
        pyg_loader = PyGGraphLoader(root=planetoid_root, name=planetoid_name)
        pyg_loader.load()
        print(f"  Load time ({pyg_loader.load_time_s:.2f}s) will be added to all full_graph timings")

    spec_exported = False
    all_results: dict[int, dict] = {}

    # Fetch Neo4j test node IDs (used by neighborhood_sampling and in_db_java)
    neo4j_test_pool: list[int] = []
    if "neighborhood_sampling" in strategies or "in_db_java" in strategies:
        neo4j_test_pool = graph_store.get_split(split="test").tolist()
        print(f"\nNeo4j test pool: {len(neo4j_test_pool)} nodes")

    pyg_test_pool_size = len(pyg_loader.test_node_indices()) if pyg_loader else 0
    if pyg_loader:
        print(f"PyG test pool: {pyg_test_pool_size} nodes")

    print(f"Node counts to test: {node_counts}")
    print(f"Strategies: {strategies}")
    print(f"Fanout per hop: {num_neighbors}  ({num_hops} hops × {num_neighbors_val} neighbors)\n")

    for N, nbr_runs in zip(node_counts, nbr_runs_list):
        # Check if strategies have enough test nodes
        neo4j_skip = (
            any(s in strategies for s in ("neighborhood_sampling", "in_db_java"))
            and N > len(neo4j_test_pool)
        )
        pyg_skip = "full_graph" in strategies and pyg_loader and N > pyg_test_pool_size
        if neo4j_skip or pyg_skip:
            print(f"Skipping N={N}: not enough test nodes")
            continue

        run_data: dict[str, list] = {s: [] for s in strategies}
        agreement_data: list[float] = []

        for run_i in range(int(nbr_runs)):
            # Sample N Neo4j test nodes (same for neighborhood_sampling and in_db_java)
            neo4j_test_ids: list[int] = []
            neo4j_labels: dict[int, int] = {}
            if "neighborhood_sampling" in strategies or "in_db_java" in strategies:
                rng = random.Random(run_i * 10007 + N)
                neo4j_test_ids = rng.sample(neo4j_test_pool, N)
                neo4j_labels = fetch_labels(neo4j_test_ids, cfg, driver)

            run_preds: dict[str, dict[int, int]] = {}

            for strategy in strategies:
                try:
                    if strategy == "full_graph":
                        preds, pyg_labels, metrics = run_full_graph_pyg(
                            model, N, run_i, pyg_loader, num_neighbors, batch_size
                        )
                        # Add one-time load cost to inference time for fair comparison
                        metrics["total_time_s"] += pyg_loader.load_time_s
                        metrics["ms_per_node"] = metrics["total_time_s"] * 1000 / max(len(preds), 1)
                        metrics["throughput_nodes_per_s"] = len(preds) / max(metrics["total_time_s"], 1e-9)
                        acc = compute_accuracy(preds, pyg_labels)

                    elif strategy == "neighborhood_sampling":
                        preds, metrics = run_neighborhood_sampling(
                            model, neo4j_test_ids, cfg, feature_store, graph_store,
                            inference_sampler,
                        )
                        acc = compute_accuracy(preds, neo4j_labels)

                    elif strategy == "in_db_java":
                        preds, metrics, spec_exported = run_in_db_java(
                            model, neo4j_test_ids, cfg, driver, spec_exported=spec_exported
                        )
                        acc = compute_accuracy(preds, neo4j_labels)

                    else:
                        print(f"  Unknown strategy '{strategy}', skipping.")
                        continue

                    run_preds[strategy] = preds
                    entry = {"accuracy": acc, **metrics}
                    run_data[strategy].append(entry)

                except Exception as exc:
                    print(f"  [{strategy}] N={N} run {run_i+1} FAILED: {exc}")
                    traceback.print_exc()

            # Compare neighborhood_sampling vs in_db_java predictions
            ns_preds = run_preds.get("neighborhood_sampling")
            java_preds = run_preds.get("in_db_java")
            if ns_preds is not None and java_preds is not None:
                common = set(ns_preds) & set(java_preds)
                if common:
                    agree = sum(1 for nid in common if ns_preds[nid] == java_preds[nid])
                    agreement_data.append(agree / len(common))

        agg_by_strategy = {s: _agg(run_data[s]) for s in strategies if run_data[s]}
        if agreement_data:
            mean_agr = float(np.mean(agreement_data)) * 100
            ci_agr = _ci95(agreement_data) * 100
            agg_by_strategy["_ns_vs_java_agreement"] = {
                "mean": mean_agr, "ci95": ci_agr,
            }
        all_results[N] = agg_by_strategy
        print_table(N, nbr_runs, agg_by_strategy)

        if agreement_data:
            print(f"  neighborhood_sampling vs in_db_java agreement: "
                  f"{mean_agr:.1f}% ±{ci_agr:.1f}%")

    print_scaling_summary(all_results)

    if pyg_loader is not None:
        print(f"\n[full_graph] One-time graph load: {pyg_loader.load_time_s:.2f}s "
              f"(included in all per-N timings above)")

    return all_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _next_run_dir(output_dir: Path) -> Path:
    """Return a fresh run_N_YYYY-MM-DD subdirectory inside *output_dir*."""
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in output_dir.iterdir():
        if p.is_dir() and p.name.startswith("run_"):
            try:
                existing.append(int(p.name.split("_")[1]))
            except (ValueError, IndexError):
                pass
    next_id = (max(existing) + 1) if existing else 0
    date_str = date.today().isoformat()
    run_dir = output_dir / f"run_{next_id}_{date_str}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_results_and_plot(all_results: dict, cfg: dict, output_dir: str, *, subgraph_verification: dict | None = None) -> None:
    ds_name = cfg["dataset"].get("dataset_name", "dataset")
    model_class = cfg["model"].get("model_class", "model")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = _next_run_dir(Path(output_dir))

    json_name = f"{ds_name}_{model_class}_{ts}.json"
    out_path = run_dir / json_name
    serialisable = {
        str(N): {s: agg for s, agg in by_strat.items()}
        for N, by_strat in all_results.items()
    }
    payload = {}
    if subgraph_verification is not None:
        payload["subgraph_verification"] = subgraph_verification
    payload["config"] = cfg["dataset"]
    payload["model"] = cfg["model"]
    payload["results"] = serialisable
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")

    plot_all(out_path, output_dir=run_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference strategy comparison experiment.")
    parser.add_argument("--dataset", required=True,
                        help="Path to inference dataset config (src/configs/inference/datasets/*.json)")
    parser.add_argument("--model", required=True,
                        help="Path to inference model config (src/configs/inference/models/*.json)")
    parser.add_argument("--output_dir", default="experiment_results/inference_comparison",
                        help="Directory to write run_N_DATE folders into (default: experiment_results/inference_comparison)")
    args = parser.parse_args()

    cfg = load_all_configs(args.dataset, args.model)

    uri      = os.environ.get("URI", "")
    user     = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    driver = Neo4jConnection(uri, user, password).get_driver()

    model, graph_store, feature_store, training_sampler = build_components(cfg, driver)
    train_or_load(model, cfg, graph_store, feature_store, training_sampler)
    model.eval()

    # Quick sanity check: sampling procedure and inference procedure agree
    subgraph_verification = None
    if "in_db_java" in cfg["model"].get("strategies", ["full_graph", "neighborhood_sampling", "in_db_java"]):
        test_pool = graph_store.get_split(split="test").tolist()
        verify_seeds = random.sample(test_pool, min(4, len(test_pool)))
        print(f"\nVerifying subgraph match on {len(verify_seeds)} seed nodes...")
        try:
            subgraph_verification = verify_subgraph_match(verify_seeds, cfg, driver)
        except Exception as e:
            msg = str(e)
            if "ProcedureNotFound" in msg or "ordered_nodes" in msg or "edge_pairs" in msg:
                print(
                    f"  ⚠ Subgraph verification skipped — deployed plugin jar is outdated.\n"
                    f"    Run: make build-plugin NEO4J_PLUGINS_DIR=<path> to redeploy.\n"
                    f"    ({msg[:120]})"
                )
            else:
                raise

    all_results = run_experiment(cfg, model, graph_store, feature_store, driver)
    save_results_and_plot(all_results, cfg, args.output_dir, subgraph_verification=subgraph_verification)

    try:
        driver.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
