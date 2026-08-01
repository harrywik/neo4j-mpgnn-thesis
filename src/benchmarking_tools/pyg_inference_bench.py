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
    """Load ogbn-papers100M as a PyG Data object with pre-caching to avoid repeated OOM.
    
    Args:
        root: Data directory
        loading_mode: "full_ram" (load everything into RAM) or "mmap" (memory-mapped)
    
    First checks for pre-processed cache. If not found, runs OGB with temporary swap.
    Saves components separately to avoid massive single-file serialization.
    """
    import subprocess
    import numpy as np
    from pathlib import Path
    
    cache_dir = Path(root) / "preprocessed_cache"
    x_file = cache_dir / "x.npy"
    edge_index_file = cache_dir / "edge_index.npy"
    y_file = cache_dir / "y.npy"
    test_mask_file = cache_dir / "test_mask.npy"
    
    # Check for pre-processed cache
    if all(f.exists() for f in [x_file, edge_index_file, y_file, test_mask_file]):
        print(f"[pyg_inference] Loading from pre-processed cache: {cache_dir} (mode={loading_mode})")
        
        if loading_mode == "mmap":
            # Memory-mapped: data stays on disk, loaded on-demand
            x = torch.from_numpy(np.load(x_file, mmap_mode='r'))
            edge_index = torch.from_numpy(np.load(edge_index_file, mmap_mode='r'))
            y = torch.from_numpy(np.load(y_file, mmap_mode='r'))
            test_mask = torch.from_numpy(np.load(test_mask_file, mmap_mode='r'))
        else:
            # Full RAM: load everything into memory
            print(f"  Loading features into RAM...")
            x = torch.from_numpy(np.load(x_file))
            print(f"  Loading edges into RAM...")
            edge_index = torch.from_numpy(np.load(edge_index_file))
            print(f"  Loading labels into RAM...")
            y = torch.from_numpy(np.load(y_file))
            test_mask = torch.from_numpy(np.load(test_mask_file))
        
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask)
        split_idx = {"test": test_mask.nonzero(as_tuple=False).squeeze(-1)}
        return data, split_idx
    
    # No cache — need to process with swap
    print(f"[pyg_inference] No pre-processed cache found — processing with temporary swap...")
    
    # Create temporary swap
    swap_file = Path("/mnt/ssd/ogb_processing_swap")
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

        # Check dtypes to avoid unnecessary copies
        print(f"  node_feat dtype: {graph['node_feat'].dtype}, shape: {graph['node_feat'].shape}")
        print(f"  edge_index dtype: {graph['edge_index'].dtype}, shape: {graph['edge_index'].shape}")
        
        # Save components separately to avoid massive serialization
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[pyg_inference] Saving components to {cache_dir}...")
        
        # Save features
        print(f"  Saving features ({graph['node_feat'].nbytes / 1e9:.1f} GB)...")
        np.save(x_file, graph["node_feat"])
        
        # Save edges
        print(f"  Saving edges ({graph['edge_index'].nbytes / 1e9:.1f} GB)...")
        np.save(edge_index_file, graph["edge_index"])
        
        # Save labels
        print(f"  Saving labels...")
        np.save(y_file, labels.squeeze())
        
        # Save test mask
        num_nodes = graph["node_feat"].shape[0]
        test_mask = np.zeros(num_nodes, dtype=bool)
        test_mask[split_idx["test"]] = True
        print(f"  Saving test mask...")
        np.save(test_mask_file, test_mask)
        
        print(f"[pyg_inference] Cache saved")
        
        # Now load from cache
        if loading_mode == "mmap":
            x = torch.from_numpy(np.load(x_file, mmap_mode='r'))
            edge_index = torch.from_numpy(np.load(edge_index_file, mmap_mode='r'))
            y = torch.from_numpy(np.load(y_file, mmap_mode='r'))
            test_mask = torch.from_numpy(np.load(test_mask_file, mmap_mode='r'))
        else:
            x = torch.from_numpy(np.load(x_file))
            edge_index = torch.from_numpy(np.load(edge_index_file))
            y = torch.from_numpy(np.load(y_file))
            test_mask = torch.from_numpy(np.load(test_mask_file))
        
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, y=y, test_mask=test_mask)
        
        return data, split_idx
        
    finally:
        # Always remove swap
        subprocess.run(["sudo", "swapoff", str(swap_file)], check=False)
        swap_file.unlink(missing_ok=True)
        print("  Temporary swap removed")


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
