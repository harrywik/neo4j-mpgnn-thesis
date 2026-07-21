"""pyg_inference_bench.py — benchmark PyG-only inference on ogbn-papers100M.

Loads the full graph into memory via OGB, then runs NeighborLoader-based
inference on N seed nodes.  Measures wall-clock time for the forward pass.

Outputs JSON to --output_json with keys:
  total_time_s, ms_per_node, throughput_nodes_per_s, n_nodes

Usage
-----
    python -m benchmarking_tools.pyg_inference_bench \\
        --n_nodes 2048 --output_json results/pyg_inference.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch_geometric.loader import NeighborLoader

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from neo4j_pyg.models.GCN import GCN


def load_ogbn_papers100M(root: str = "data/ogbn-papers100M"):
    """Load ogbn-papers100M as a PyG Data object."""
    from ogb.nodeproppred import NodePropPredDataset

    _orig_load = torch.load
    torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
    try:
        dataset = NodePropPredDataset(name="ogbn-papers100M", root=root)
        graph, labels = dataset[0]
        split_idx = dataset.get_idx_split()
    finally:
        torch.load = _orig_load

    from torch_geometric.data import Data
    x = torch.tensor(graph["node_feat"], dtype=torch.float)
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long).squeeze()

    num_nodes = x.shape[0]
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[split_idx["test"]] = True

    return Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask), split_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=2048)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/ogbn-papers100M")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"[pyg_inference] Loading ogbn-papers100M...")
    t_load = time.monotonic()
    data, split_idx = load_ogbn_papers100M(args.data_root)
    load_time = time.monotonic() - t_load
    print(f"[pyg_inference] Loaded {data.num_nodes} nodes, {data.num_edges} edges in {load_time:.1f}s")

    test_indices = split_idx["test"].tolist()
    n = min(args.n_nodes, len(test_indices))
    rng = random.Random(args.seed)
    seed_ids = rng.sample(test_indices, n)
    input_nodes = torch.tensor(seed_ids, dtype=torch.long)

    model = GCN(in_dim=128, hidden_dim1=192, hidden_dim2=192, nbr_classes=172)
    model.eval()

    print(f"[pyg_inference] Running inference on {n} nodes...")
    loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        input_nodes=input_nodes,
        batch_size=args.batch_size,
        shuffle=False,
    )

    t0 = time.monotonic()
    with torch.no_grad():
        for batch in loader:
            model(batch.x, batch.edge_index)
    elapsed = time.monotonic() - t0

    ms_per_node = elapsed * 1000 / n
    throughput = n / elapsed

    result = {
        "n_nodes": n,
        "total_time_s": round(elapsed, 4),
        "ms_per_node": round(ms_per_node, 4),
        "throughput_nodes_per_s": round(throughput, 2),
        "load_time_s": round(load_time, 2),
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[pyg_inference] Done: {elapsed:.3f}s total, {ms_per_node:.3f} ms/node, {throughput:.1f} nodes/s")
    print(f"[pyg_inference] Results → {out_path}")


if __name__ == "__main__":
    main()
