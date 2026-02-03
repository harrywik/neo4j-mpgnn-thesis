from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
from torch import optim
from InMemoryFeatureStore import InMemoryFeatureStore
from InMemoryGraph import InMemoryGraphStore
from Model import GCN
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.loader import NodeLoader
from CustomSampler import InMemorySampler
from torch_geometric.utils import mask_to_index
from torch_geometric.sampler import NodeSamplerInput
from torch_geometric.utils import subgraph

def main():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())      
    graph = dataset[0]
    
    
    fstore = InMemoryFeatureStore() # no edge features to store here
    fstore["node", "x", None] = graph.x
    fstore["node", "y", None] = graph.y  # IMPORTANT: NodeLoader will fetch y via FeatureStore
    N = graph.x.shape[0]
    edge_index = graph.edge_index
    row, col = edge_index[0].contiguous(), edge_index[1].contiguous()

    NODE_TYPE = "node"
    EDGE_TYPE = (NODE_TYPE, "to", NODE_TYPE)  # keep consistent with FeatureStore keys
    LAYOUT = EdgeLayout.COO

    gstore = InMemoryGraphStore()
    gstore.put_edge_index((row, col), edge_type=EDGE_TYPE, layout=LAYOUT)

    # Optional sanity check:
    rc = gstore.get_edge_index(edge_type=EDGE_TYPE, layout=LAYOUT)
    assert rc is not None and torch.equal(rc[0], row) and torch.equal(rc[1], col)
    
    r, c = gstore.get_edge_index(edge_type=EDGE_TYPE, layout=LAYOUT)
    sampler = InMemorySampler(gstore, EDGE_TYPE, layout=LAYOUT, undirected=True)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    model = GCN(in_dim=1433, hidden_dim=16, out_dim=16, nbr_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask
    targets = graph.y

    model.train()

    for epoch in range(300):
        seed_nodes = torch.nonzero(train_mask, as_tuple=False).view(-1)

        batches = NodeLoader(
            data=(fstore, gstore),
            node_sampler=sampler,
            input_nodes=seed_nodes,
            batch_size=10,
            shuffle=True,
        )

        for step, batch in enumerate(batches):
            optimizer.zero_grad()

            edge_index_sub = batch.edge_index              # [2, E_sub]
            x_sub = getattr(batch, "x", None)
            y_sub = getattr(batch, "y", None)
            if x_sub is None:
                x_sub = fstore["node", "x", batch.n_id]    # use n_id (global IDs)
            if y_sub is None:
                y_sub = fstore["node", "y", batch.n_id]

            logits = model(x_sub, edge_index_sub)

            # Seeds present in this batch: compare global IDs
            seed_mask = torch.isin(batch.n_id, batch.input_id)
            loss = criterion(logits[seed_mask], y_sub[seed_mask])

            loss.backward()
            optimizer.step()


    # evaluate
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        pred = logits.argmax(dim=1)
        test_acc = (pred[test_mask] == graph.y[test_mask]).float().mean().item()
    print(test_acc)

if __name__ == "__main__":
    main()