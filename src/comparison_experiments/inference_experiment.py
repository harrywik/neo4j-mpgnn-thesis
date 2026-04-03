"""inference_experiment.py — compare GNN inference strategies across N-node scales.

Trains (or loads) a model, then benchmarks three inference strategies for each
value of N (number of seed nodes to run inference on):

  full_graph          — load entire graph into RAM; single PyTorch forward pass
  neighborhood_sampling — Cypher-fetch k-hop subgraphs per batch; Python forward pass
  in_db_java          — export spec/weights; call gnnProcedures.inference.run inside Neo4j

Each (strategy, N) cell is repeated ``nbr_runs`` times with different random
seed-node samples; results are averaged and reported as mean ± 95 % CI.

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
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from torch_geometric.loader import NodeLoader

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

    # Allow inference dataset config to override training dataset fields
    merged_train_ds = {**train_ds, **{
        k: v for k, v in ds.items()
        if k not in ("training_dataset_config", "training_implementation_config",
                     "node_counts", "nbr_runs", "inference_batch_size", "max_neighbors")
    }}

    # Apply model config overrides for training dims
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
    from training.Main import (
        _build_common_kwargs,
        _make_feature_store,
        _make_graph_store,
        _make_model,
        _make_sampler,
    )

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

def train_or_load(model, cfg: dict, graph_store, feature_store, sampler) -> None:
    checkpoint = cfg["model"].get("checkpoint_path")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["MODEL_STATE"])
        print(f"Loaded checkpoint from {checkpoint}")
        return

    from training.Training import Trainer, put_nodeLoader_args_map
    from benchmarking_tools import Measurer
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


# ---------------------------------------------------------------------------
# Ground-truth label fetching
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
# Full-graph cache
# ---------------------------------------------------------------------------

class FullGraphCache:
    """Loads the entire graph into memory once and reuses it for all N values."""

    def __init__(self, cfg: dict, driver):
        self.cfg = cfg
        self.driver = driver
        self._x: torch.Tensor | None = None
        self._edge_index: torch.Tensor | None = None
        self._labels: dict[int, int] = {}
        self._id_to_local: dict[int, int] = {}
        self._local_to_id: list[int] = []
        self.load_time_s: float = 0.0

    def load(self) -> None:
        ds = self.cfg["train_ds"]
        fp = ds["feature_property"]
        fp_type = ds["feature_property_type"]
        label_prop = ds["target_property"]
        nid_prop = ds["nodeid_property"]
        node_label = ds["node_label"]
        edge_type = ds.get("edge_type", "")
        db = ds["database_name"]

        t0 = time.monotonic()

        # --- nodes ---
        node_query = (
            f"MATCH (n:{node_label}) "
            f"RETURN n.{nid_prop} AS id, n.{fp} AS feat, n.{label_prop} AS label "
            f"ORDER BY n.{nid_prop}"
        )
        rows = []
        with self.driver.session(database=db, fetch_size=-1) as session:
            for r in session.run(node_query):
                rows.append((r["id"], r["feat"], r["label"]))

        self._local_to_id = [r[0] for r in rows]
        self._id_to_local = {nid: i for i, nid in enumerate(self._local_to_id)}

        feat_list = []
        for _, feat_raw, label in rows:
            feat_list.append(_decode_feature(feat_raw, fp_type))
            nid = rows[len(feat_list) - 1][0]
            self._labels[nid] = int(label)

        self._x = torch.tensor(np.stack(feat_list), dtype=torch.float32)

        # --- edges ---
        rel = f"[:{edge_type}]" if edge_type else "[]"
        edge_query = (
            f"MATCH (a:{node_label})-{rel}->(b:{node_label}) "
            f"RETURN a.{nid_prop} AS src, b.{nid_prop} AS dst"
        )
        srcs, dsts = [], []
        with self.driver.session(database=db, fetch_size=-1) as session:
            for r in session.run(edge_query):
                ls, ld = self._id_to_local.get(r["src"]), self._id_to_local.get(r["dst"])
                if ls is not None and ld is not None:
                    srcs.append(ls)
                    dsts.append(ld)

        self._edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        self.load_time_s = time.monotonic() - t0
        print(f"  [full_graph] Loaded {len(self._local_to_id)} nodes, "
              f"{len(srcs)} edges in {self.load_time_s:.2f}s")

    @property
    def x(self) -> torch.Tensor:
        return self._x

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index

    def local_idx(self, node_id: int) -> int | None:
        return self._id_to_local.get(node_id)

    def label(self, node_id: int) -> int | None:
        return self._labels.get(node_id)


def _decode_feature(feat_raw, fp_type: str) -> np.ndarray:
    if fp_type == "byte[]":
        return np.frombuffer(bytes(feat_raw), dtype=np.float32)
    else:
        return np.asarray(feat_raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Strategy: full_graph
# ---------------------------------------------------------------------------

def run_full_graph(
    model: torch.nn.Module,
    test_ids: list[int],
    cache: FullGraphCache,
) -> tuple[dict[int, int], dict[str, Any]]:
    tracemalloc.start()
    t0 = time.monotonic()

    device = next(model.parameters()).device
    x = cache.x.to(device)
    edge_index = cache.edge_index.to(device)

    with torch.no_grad():
        logits = model(x, edge_index)

    preds: dict[int, int] = {}
    for nid in test_ids:
        local = cache.local_idx(nid)
        if local is not None:
            preds[nid] = int(logits[local].argmax().item())

    elapsed = time.monotonic() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    metrics = {
        "total_time_s": elapsed,
        "ms_per_node": elapsed * 1000 / max(len(test_ids), 1),
        "throughput_nodes_per_s": len(test_ids) / max(elapsed, 1e-9),
        "peak_memory_mb": peak / 1024 / 1024,
        "p50_batch_ms": None,
        "p95_batch_ms": None,
        "n_batches": 1,
    }
    return preds, metrics


# ---------------------------------------------------------------------------
# Strategy: neighborhood_sampling
# ---------------------------------------------------------------------------

def run_neighborhood_sampling(
    model: torch.nn.Module,
    test_ids: list[int],
    cfg: dict,
    feature_store,
    graph_store,
    sampler,
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
        node_sampler=sampler,
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
        from create_inference_spec import create_inference_spec
        create_inference_spec(
            model,
            model_name,
            base_dir=gnn_model_dir,
            max_neighbors=ds.get("max_neighbors", 10),
        )
        spec_exported = True

    batch_size = ds.get("inference_batch_size", 256)
    node_label = ds["node_label"]
    edge_type = ds.get("edge_type", "")
    feature_prop = ds["feature_property"]
    feature_type = ds["feature_property_type"]
    nodeid_prop = ds["nodeid_property"]
    db = cfg["train_ds"]["database_name"]
    max_neighbors = ds.get("max_neighbors", 10)

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
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict, model, graph_store, feature_store, sampler, driver) -> dict:
    ds = cfg["dataset"]
    mdl = cfg["model"]
    node_counts: list[int] = ds["node_counts"]
    nbr_runs: int = ds.get("nbr_runs", 5)
    strategies: list[str] = mdl.get("strategies", ["full_graph", "neighborhood_sampling", "in_db_java"])

    # Pre-load full graph once if needed
    full_graph_cache: FullGraphCache | None = None
    if "full_graph" in strategies:
        print("\nLoading full graph into memory (one-time cost)...")
        full_graph_cache = FullGraphCache(cfg, driver)
        full_graph_cache.load()

    spec_exported = False
    all_results: dict[int, dict] = {}

    # Fetch all test node IDs upfront (we'll sample from this pool)
    test_pool = graph_store.get_split(split="test").tolist()
    print(f"\nTest pool: {len(test_pool)} nodes")
    print(f"Node counts to test: {node_counts}")
    print(f"Strategies: {strategies}\n")

    for N in node_counts:
        if N > len(test_pool):
            print(f"Skipping N={N}: not enough test nodes (pool={len(test_pool)})")
            continue

        run_data: dict[str, list] = {s: [] for s in strategies}

        for run_i in range(nbr_runs):
            rng = random.Random(run_i * 10007 + N)
            test_ids = rng.sample(test_pool, N)
            labels = fetch_labels(test_ids, cfg, driver)

            for strategy in strategies:
                try:
                    if strategy == "full_graph":
                        preds, metrics = run_full_graph(model, test_ids, full_graph_cache)
                    elif strategy == "neighborhood_sampling":
                        preds, metrics = run_neighborhood_sampling(
                            model, test_ids, cfg, feature_store, graph_store, sampler
                        )
                    elif strategy == "in_db_java":
                        preds, metrics, spec_exported = run_in_db_java(
                            model, test_ids, cfg, driver, spec_exported=spec_exported
                        )
                    else:
                        print(f"  Unknown strategy '{strategy}', skipping.")
                        continue

                    acc = compute_accuracy(preds, labels)
                    entry = {"accuracy": acc, **metrics}
                    run_data[strategy].append(entry)

                except Exception as exc:
                    print(f"  [{strategy}] N={N} run {run_i+1} FAILED: {exc}")

        agg_by_strategy = {s: _agg(run_data[s]) for s in strategies if run_data[s]}
        all_results[N] = agg_by_strategy
        print_table(N, nbr_runs, agg_by_strategy)

    print_scaling_summary(all_results)

    # Report full-graph load overhead
    if full_graph_cache is not None:
        print(f"\n[full_graph] One-time graph load: {full_graph_cache.load_time_s:.2f}s "
              f"(not included in per-N timings above)")

    return all_results


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_results: dict, cfg: dict, output_dir: str) -> None:
    ds_name = cfg["dataset"].get("dataset_name", "dataset")
    mdl_name = cfg["model"].get("model_class", "model")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir) / f"{ds_name}_{mdl_name}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serialisable = {
        str(N): {s: agg for s, agg in by_strat.items()}
        for N, by_strat in all_results.items()
    }
    with open(out_path, "w") as f:
        json.dump({"config": cfg["dataset"], "model": cfg["model"], "results": serialisable}, f, indent=2)
    print(f"\nResults saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inference strategy comparison experiment.")
    parser.add_argument("--dataset", required=True,
                        help="Path to inference dataset config (src/configs/inference/datasets/*.json)")
    parser.add_argument("--model", required=True,
                        help="Path to inference model config (src/configs/inference/models/*.json)")
    parser.add_argument("--output_dir", default="results/inference_comparison",
                        help="Directory to write result JSON (default: results/inference_comparison)")
    args = parser.parse_args()

    cfg = load_all_configs(args.dataset, args.model)

    uri      = os.environ.get("URI", "")
    user     = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    from Neo4jConnection import Neo4jConnection
    driver = Neo4jConnection(uri, user, password).get_driver()

    model, graph_store, feature_store, sampler = build_components(cfg, driver)
    train_or_load(model, cfg, graph_store, feature_store, sampler)
    model.eval()

    all_results = run_experiment(cfg, model, graph_store, feature_store, sampler, driver)
    save_results(all_results, cfg, args.output_dir)

    try:
        driver.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
