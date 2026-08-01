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


def load_ogbn_papers100M(root: str = "data/ogbn-papers100M", loading_mode: str = "full_ram"):
    """Load ogbn-papers100M as a PyG Data object.
    
    Args:
        root: Data directory
        loading_mode: "full_ram" (load everything into RAM) or "mmap" (memory-mapped)
    
    Uses OGB's native processed cache. If not found, processes with temporary swap.
    """
    import subprocess
    import numpy as np
    from pathlib import Path
    
    processed_file = Path(root) / "ogbn_papers100M" / "processed" / "data_processed"
    
    # Check if OGB's processed data exists
    if not processed_file.exists():
        print(f"[pyg_inference] OGB processed data not found — processing with temporary swap...")

        # Create temporary swap (clean up any leftover from previous OOM)
        swap_file = Path("/mnt/ssd/ogb_processing_swap")
        subprocess.run(["sudo", "swapoff", str(swap_file)], check=False)
        swap_file.unlink(missing_ok=True)
        subprocess.run(["sudo", "fallocate", "-l", "100G", str(swap_file)], check=True)
        subprocess.run(["sudo", "chmod", "600", str(swap_file)], check=True)
        subprocess.run(["sudo", "mkswap", str(swap_file)], check=True)
        subprocess.run(["sudo", "swapon", str(swap_file)], check=True)
        print("  Temporary swap enabled")
        
        try:
            from ogb.nodeproppred import NodePropPredDataset

            # Ensure processed/ exists — OGB won't create it itself
            Path(root, "ogbn_papers100M", "processed").mkdir(parents=True, exist_ok=True)

            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
            try:
                dataset = NodePropPredDataset(name="ogbn-papers100M", root=root)
                graph, labels = dataset[0]
                split_idx = dataset.get_idx_split()
            finally:
                torch.load = _orig_load
            
            print(f"[pyg_inference] OGB processing complete")
            
        finally:
            # Always remove swap
            subprocess.run(["sudo", "swapoff", str(swap_file)], check=False)
            swap_file.unlink(missing_ok=True)
            print("  Temporary swap removed")
    else:
        print(f"[pyg_inference] Loading from OGB processed cache: {processed_file}")
        from ogb.nodeproppred import NodePropPredDataset
        
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        try:
            dataset = NodePropPredDataset(name="ogbn-papers100M", root=root)
            graph, labels = dataset[0]
            split_idx = dataset.get_idx_split()
        finally:
            torch.load = _orig_load

    # Check dtypes to avoid unnecessary copies
    print(f"  node_feat dtype: {graph['node_feat'].dtype}, shape: {graph['node_feat'].shape}")
    print(f"  edge_index dtype: {graph['edge_index'].dtype}, shape: {graph['edge_index'].shape}")
    
    # Convert to torch tensors
    if loading_mode == "mmap":
        # For mmap mode, we need to save to disk and load with mmap
        # But OGB already loaded into memory, so we can't avoid the copy here
        # This mode is mainly useful if we had pre-saved numpy files
        print(f"  Warning: mmap mode not fully supported with OGB native loading, using full_ram")
    
    # Zero-copy conversion if dtypes match
    x_np = graph["node_feat"]
    if x_np.dtype == np.float32:
        x = torch.from_numpy(x_np)  # Zero-copy
    else:
        x = torch.from_numpy(x_np).float()  # Creates copy
    
    edge_np = graph["edge_index"]
    if edge_np.dtype == np.int64:
        edge_index = torch.from_numpy(edge_np)  # Zero-copy
    else:
        edge_index = torch.from_numpy(edge_np).long()  # Creates copy
    
    y_np = labels
    if y_np.dtype == np.int64:
        y = torch.from_numpy(y_np).squeeze()  # Zero-copy
    else:
        y = torch.from_numpy(y_np).long().squeeze()  # Creates copy

    num_nodes = x.shape[0]
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[split_idx["test"]] = True

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask)
    
    return data, split_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_nodes", type=int, default=2048)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="data/ogbn-papers100M")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loading_mode", type=str, default="full_ram",
                        choices=["full_ram", "mmap"],
                        help="full_ram: load all data into RAM (default), mmap: memory-mapped from disk")
    args = parser.parse_args()

    print(f"[pyg_inference] Loading ogbn-papers100M (mode={args.loading_mode})...")
    t_load = time.monotonic()
    data, split_idx = load_ogbn_papers100M(args.data_root, loading_mode=args.loading_mode)
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
