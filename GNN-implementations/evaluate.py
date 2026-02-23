from typing import Tuple
import numpy as np
from torch_geometric.loader import NodeLoader
import torch
from torch import nn
from torch_geometric.data import GraphStore, FeatureStore
from torch_geometric.sampler import BaseSampler

def evaluate(model: nn.Module, graph_store: GraphStore, feature_store: FeatureStore, sampler: BaseSampler, split: str = "val") -> float:
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
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        N: int = 256
        criterion = nn.CrossEntropyLoss()
        counts = []
        partial_accuracies = []
        node_ids = graph_store.get_split(split=split)
        val_loader = NodeLoader(
                data=(feature_store, graph_store),
                node_sampler=sampler,
                input_nodes=node_ids,
                batch_size=N,
                shuffle=False,
            )
        
        losses = []
        for data in val_loader:
            data = data.to(device)
            out: torch.Tensor = model(data.x, data.edge_index)
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
        print(f"{split.capitalize()} accuracy: {acc:.2f}")
        
        average_loss = float(cnts @ np.array(losses))
        
        return acc, average_loss

