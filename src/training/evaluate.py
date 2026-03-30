from typing import List, Tuple, Union
import numpy as np
from torch_geometric.loader import NodeLoader, NeighborLoader
import torch
from torch import nn
from torch_geometric.data import GraphStore, FeatureStore, HeteroData, Data
from torch_geometric.sampler import BaseSampler


def _get_model_kwargs(model: nn.Module, batch, graph_store) -> dict:
    if not getattr(model, "_uses_hybrid_last_hop_preaggregation", False):
        return {}
    if graph_store is None or not hasattr(batch, "n_id"):
        return {}

    hop_depths_map = getattr(graph_store, "last_sampled_hop_depths", {})
    preagg_map = getattr(graph_store, "last_hybrid_preagg", {})
    node_ids = batch.n_id.detach().cpu().tolist()

    hop_depths = torch.tensor(
        [int(hop_depths_map.get(int(nid), 0)) for nid in node_ids],
        dtype=torch.long,
        device=batch.x.device,
    )
    last_hop_preagg = torch.zeros_like(batch.x)
    for i, nid in enumerate(node_ids):
        preagg = preagg_map.get(int(nid))
        if preagg is not None:
            last_hop_preagg[i] = torch.as_tensor(preagg, device=batch.x.device, dtype=batch.x.dtype)

    return {
        "hop_depths": hop_depths,
        "last_hop_preagg": last_hop_preagg,
    }

def evaluate(model: nn.Module, data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]], sampler: BaseSampler = None, split: str = "val", num_neighbors: List[int] = None, iteration: int = None, limit: int | None = None) -> float:
    """Evaluate a GNN model on a dataset split using neighbor sampling.

    The function iterates over the requested split in fixed-size chunks of seed
    nodes, constructs a one-batch `NodeLoader` per chunk, and computes accuracy
    on the seed nodes only. It returns a weighted mean of partial accuracies
    where each chunk is weighted by its number of seed nodes.

    Args:
        model: A PyTorch model that accepts `(x, edge_index)` and returns logits.
        graph_store: GraphStore providing `get_split(...)` and graph topology.
        feature_store: FeatureStore providing node features and labels.
        sampler: A PyG neighbor sampler compatible with `NodeLoader`.
        split: Dataset split name, e.g., "train", "val", or "test".

    Returns:
        The weighted accuracy for the specified split.
    """
    val_loader = None
    N: int = 256

    if isinstance(data, tuple):
        feature_store, graph_store = data
        node_ids = graph_store.get_split(limit=limit, split=split)
        val_loader = NodeLoader(
                data=(feature_store, graph_store),
                node_sampler=sampler,
                input_nodes=node_ids,
                batch_size=N,
                shuffle=False,
            )
    else:
        val_loader = NeighborLoader(
                        data,
                        # Sample 30 neighbors for each node for 2 iterations
                        num_neighbors=num_neighbors,
                        # Use a batch size of 128 for sampling training nodes
                        batch_size=N,
                        input_nodes=data.test_mask if split == "test" else data.val_mask,
                    )
    model.eval()
    device = next(model.parameters()).device
    graph_store = data[1] if isinstance(data, tuple) else None
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        counts = []
        partial_accuracies = []
        losses = []
        for data in val_loader:
            data = data.to(device)
            out: torch.Tensor = model(data.x, data.edge_index, **_get_model_kwargs(model, data, graph_store))
            seed_mask = torch.isin(data.n_id, data.input_id)
            targets = data.y[seed_mask]
            preds = out[seed_mask].argmax(dim=1)

            partial_accuracies.append((targets == preds).sum().item() / targets.numel())
            loss = criterion(out[seed_mask], targets)
            losses.append(loss)
            counts.append(seed_mask.sum().item())
            
        cnts = np.array(counts, dtype=np.float32)
        cnts /= cnts.sum()
        acc = float(cnts @ np.array(partial_accuracies))
        if iteration:
            print(f"Epoch {iteration} | {split.capitalize()} accuracy: {acc:.2f}")
        else:
            print(f"{split.capitalize()} accuracy: {acc:.2f}")
        
        average_loss = float(cnts @ np.array(losses))
        
        return acc, average_loss

