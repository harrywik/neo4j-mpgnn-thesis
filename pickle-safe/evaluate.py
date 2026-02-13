import numpy as np
from torch_geometric.loader import NodeLoader
import torch

def evaluate(model, graph_store, feature_store, sampler, split: str = "val") -> float:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        N: int = 2**8
        i: int = 0

        counts = []
        partial_accuracies = []

        while True:
            node_ids = graph_store.get_split(N, offset=i, split=split, shuffle=False)

            if node_ids.numel() == 0:
                break

            i += node_ids.numel()

            val_loader = NodeLoader(
                data=(feature_store, graph_store),
                node_sampler=sampler,
                input_nodes=node_ids,
                batch_size=N,
                shuffle=False,
            )
            for data in val_loader:
                break

            data = data.to(device)
            out: torch.Tensor = model(data.x, data.edge_index)
            seed_mask = torch.isin(data.n_id, data.input_id)
            targets = data.y[seed_mask]
            preds = out[seed_mask].argmax(dim=1)

            counts.append(i)
            partial_accuracies.append((targets == preds).sum().item() / targets.numel())

        cnts = np.array(counts, dtype=np.float32)
        cnts /= cnts.sum()
        acc = float(cnts @ np.array(partial_accuracies))
        print(split.capitalize(), "accuracy:", acc)
        return acc
