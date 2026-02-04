import sys
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
import time
import cProfile
import pstats
from pathlib import Path

def main():
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())      
    graph = dataset[0]
    train_idx = torch.where(graph.test_mask)[0]
    print(train_idx)
    
    
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
    number_workers = 1

    model.train()
    training_time_start = time.time()
    for epoch in range(300):
        seed_nodes = torch.nonzero(train_mask, as_tuple=False).view(-1) 
        
        batches = NodeLoader(   # At construction time, NodeLoader does not fetch features/edges
            data=(fstore, gstore),
            node_sampler=sampler,
            input_nodes=seed_nodes,
            batch_size=500,
            shuffle=True,
            # num_workers=number_workers,            # <-- parallel workers
            # persistent_workers=True,  # <-- keep workers alive across epochs
            # prefetch_factor=2,        # <-- batches prefetched per worker (PyTorch)
        )
        
        for step, batch in enumerate(batches): # PyTorch’s DataLoader picks 10 integers from range(len(seed_nodes)) (shuffled)
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
    training_duration = time.time() - training_time_start


    # evaluate
    model.eval()
    with torch.no_grad():
        logits = model(graph.x, graph.edge_index)
        pred = logits.argmax(dim=1)
        test_acc = (pred[test_mask] == graph.y[test_mask]).float().mean().item()
    
    print("nbr workers      :")
    print(f"accuracy         : {test_acc:.2f}")
    print(f"training duration: {training_duration:.2f}")
if __name__ == "__main__":
    
    write = True
    if write:

        BASE_DIR = Path(__file__).resolve().parent                 # folder containing Main.py
        profiles_dir = BASE_DIR.parent / "profiles"                 # sibling folder named "profile"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        folder_name = BASE_DIR.name                                # e.g. "InMemoryGNNExample"
        file_name = Path(__file__).stem                            # e.g. "Main"
        base = f"{folder_name}_{file_name}"

        prof_path = profiles_dir / f"{base}.prof"
        txt_path  = profiles_dir / f"{base}.txt"

        pr = cProfile.Profile()
        pr.enable()
        main()
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))                           # overwrites
        with txt_path.open("w") as f:                              # overwrites
            stats.stream = f
            stats.print_stats(50)

        print(f"wrote {prof_path} and {txt_path}")
    else:
        main()
