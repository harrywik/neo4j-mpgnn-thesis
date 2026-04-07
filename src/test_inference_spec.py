"""
Numerical correctness test for the GNN inference spec pipeline.

What this tests
───────────────
1. create_inference_spec writes spec.json + weights.bin correctly.
2. The binary weight format round-trips (written by Python, readable by Python
   — the same format the Java procedure reads).
3. The inference algorithm (gcn_norm aggregation → linear → relu) produces
   exactly the same predictions as PyTorch on the same graph.

No Neo4j instance required.  Run with:

    python -m src.test_inference_spec
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.create_inference_spec import create_inference_spec


# ---------------------------------------------------------------------------
# Tiny GCN model (matches the real GCN.py structure exactly)
# ---------------------------------------------------------------------------

class TinyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.GCN1      = GCNConv(4, 6)
        self.GCN2      = GCNConv(6, 6)
        self.classifier = nn.Linear(6, 3)

    def forward(self, x, edge_index):
        x = self.GCN1(x, edge_index).relu_()
        x = self.GCN2(x, edge_index).relu_()
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Python reimplementation of the Java inference engine
# (must stay in sync with runEngine() in GNNProcedures.java)
# ---------------------------------------------------------------------------

def _load_weights(weights_path: str) -> dict[str, np.ndarray]:
    """Read back weights.bin — same binary format as the Java parser."""
    weights = {}
    with open(weights_path, "rb") as f:
        (num,) = struct.unpack("<i", f.read(4))
        for _ in range(num):
            (key_len,) = struct.unpack("<i", f.read(4))
            key = f.read(key_len).decode("utf-8")
            (rank,)  = struct.unpack("<i", f.read(4))
            dims     = struct.unpack(f"<{rank}i", f.read(rank * 4))
            total    = int(np.prod(dims)) if dims else 1
            data     = np.frombuffer(f.read(total * 4), dtype="<f4").astype(np.float64)
            weights[key] = data.reshape(dims) if dims else data
    return weights


def _gcn_norm_agg(
    node_id: int,
    neighbor_ids: list[int],
    features: dict[int, np.ndarray],
    in_deg: dict[int, int],
) -> np.ndarray | None:
    """GCN-normalised 1-hop aggregation with self-loop — mirrors Java gcn_norm."""
    self_feat = features.get(node_id)
    self_deg_hat = in_deg.get(node_id, 0) + 1.0

    agg = None
    if self_feat is not None:
        agg = self_feat * (1.0 / self_deg_hat)

    for nbr in neighbor_ids:
        nbr_feat = features.get(nbr)
        if nbr_feat is None:
            continue
        if agg is None:
            agg = np.zeros_like(nbr_feat)
        nbr_deg_hat = in_deg.get(nbr, 0) + 1.0
        agg = agg + nbr_feat / np.sqrt(self_deg_hat * nbr_deg_hat)

    return agg


def _linear(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return W @ x + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def run_java_engine_python(
    spec: dict,
    weights: dict[str, np.ndarray],
    seed_ids: list[int],
    features: dict[int, np.ndarray],   # node_id → raw feature vector
    adjacency: dict[int, list[int]],   # node_id → list of incoming neighbour ids
    in_deg: dict[int, int],            # node_id → stored in-degree (excl. self-loop)
) -> dict[int, int]:
    """
    Pure-Python replica of GNNProcedures.runEngine().
    Returns {node_id: predicted_class} for each seed.
    """
    num_hops = spec["num_hops"]

    # --- Step 1: collect hop rings (simplified: use full adjacency) ----------
    # For the test we provide all features upfront, so we just replicate the
    # hop-ring tracking to know which nodes get linear/relu applied per layer.
    all_nodes: set[int] = set(seed_ids)
    hop_nodes: list[list[int]] = [list(seed_ids)]

    frontier = list(seed_ids)
    for _ in range(num_hops):
        new_layer = []
        next_frontier = []
        for nid in frontier:
            for nbr in adjacency.get(nid, []):
                if nbr not in all_nodes:
                    all_nodes.add(nbr)
                    new_layer.append(nbr)
                    next_frontier.append(nbr)
        hop_nodes.append(new_layer)
        frontier = next_frontier

    # --- Step 2: current node representations --------------------------------
    current_h: dict[int, np.ndarray] = {n: features[n].copy() for n in all_nodes if n in features}

    # --- Step 3: execute layer plan ------------------------------------------
    active_level   = num_hops - 1
    last_agg_level = max(0, num_hops - 1)

    for layer in spec["layers"]:
        op = layer["op"]

        if op == "aggregate":
            new_h = {}
            for d in range(active_level + 1):
                for nid in hop_nodes[d]:
                    nbrs = adjacency.get(nid, [])
                    agg  = _gcn_norm_agg(nid, nbrs, current_h, in_deg)
                    if agg is not None:
                        new_h[nid] = agg
            current_h.update(new_h)
            last_agg_level = active_level
            active_level  -= 1

        elif op == "linear":
            W = weights[layer["weight"]]
            b = weights[layer["bias"]]
            for d in range(last_agg_level + 1):
                for nid in hop_nodes[d]:
                    if nid in current_h:
                        current_h[nid] = _linear(current_h[nid], W, b)

        elif op == "relu":
            for d in range(last_agg_level + 1):
                for nid in hop_nodes[d]:
                    if nid in current_h:
                        current_h[nid] = _relu(current_h[nid])

    # --- Step 4: predictions -------------------------------------------------
    return {sid: int(np.argmax(current_h[sid])) for sid in seed_ids if sid in current_h}


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_numerical_correctness():
    torch.manual_seed(42)
    model = TinyGCN()
    model.eval()

    # Small graph: 8 nodes, edges as (src → dst) in INCOMING direction
    # i.e. dst aggregates features from src
    edges = [(1,0),(2,0),(3,1),(4,1),(4,2),(5,2),(5,3),(6,4),(7,5),(7,6)]
    src_t = torch.tensor([e[0] for e in edges])
    dst_t = torch.tensor([e[1] for e in edges])
    edge_index = torch.stack([src_t, dst_t])   # PyG: [2, E]

    N = 8
    torch.manual_seed(7)
    X = torch.randn(N, 4)

    # PyTorch reference
    with torch.no_grad():
        pyg_logits = model(X, edge_index)           # [N, 3]
    pyg_preds = pyg_logits.argmax(dim=1).tolist()   # list of ints

    # Export spec
    with tempfile.TemporaryDirectory() as base_dir:
        os.environ["NEO4J_GNN_MODEL_DIR"] = base_dir
        create_inference_spec(model, "test")

        with open(os.path.join(base_dir, "test", "spec.json")) as f:
            spec = json.load(f)
        weights = _load_weights(os.path.join(base_dir, "test", "weights.bin"))

    # Build adjacency + in-degrees for the Python engine
    adjacency: dict[int, list[int]] = {i: [] for i in range(N)}
    in_deg: dict[int, int]          = {i: 0  for i in range(N)}
    for s, d in edges:
        adjacency[d].append(s)
        in_deg[d] += 1

    features = {i: X[i].numpy().astype(np.float64) for i in range(N)}

    # Seed = all nodes
    seed_ids = list(range(N))

    java_preds = run_java_engine_python(spec, weights, seed_ids, features, adjacency, in_deg)

    # Compare
    print(f"{'node':>5}  {'PyTorch':>8}  {'Java-eq':>8}  {'match':>6}")
    print("-" * 34)
    all_match = True
    for nid in seed_ids:
        pt  = pyg_preds[nid]
        jv  = java_preds.get(nid, -1)
        ok  = pt == jv
        all_match = all_match and ok
        print(f"{nid:>5}  {pt:>8}  {jv:>8}  {'✓' if ok else '✗':>6}")

    print()
    if all_match:
        print("All predictions match. ✓")
    else:
        print("MISMATCH — check gcn_norm_agg or weight serialisation.")
        sys.exit(1)


if __name__ == "__main__":
    test_numerical_correctness()
