
# Training and evaluation guidline for reproducability
This is a script to make sure that the different GNN implemenations are trained the same amount of iterations, and evaluated on the same metrics to ensure fair comparison.

### Training
`train/val/test-split:` 
Let: 
- training ids be in the range [0, 139] (torch.id)
- val ids [140, 639] (torch.id)
- test ids [1708, 2707] (torch.id)

`Epoch:` run `10` epochs where you iterate all nodes in the training data

### Evaluation
test the models using the evaluate model function below

```py
def evaluate(model, graph_store, feature_store, sampler, split: str = "val") -> None:
    
    model.eval()
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
                shuffle=False
            )
            for data in val_loader:
                break

            out: torch.Tensor = model(data.x, data.edge_index)
            seed_mask = torch.isin(data.n_id, data.input_id)
            targets = data.y[seed_mask]
            preds = out[seed_mask].argmax(dim=1)

            counts.append(i)
            partial_accuracies.append((targets == preds).sum().item() / targets.numel())

        cnts = np.array(counts, dtype=np.float32)
        cnts /= cnts.sum()
        print(split.capitalize(), "accuracy:", cnts  @ np.array(partial_accuracies))
```
