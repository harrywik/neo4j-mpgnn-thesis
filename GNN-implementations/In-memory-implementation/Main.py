import sys
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
from torch import optim
from InMemoryFeatureStore import InMemoryFeatureStore
from InMemoryGraph import InMemoryGraphStore
from Model import GCN
from torch_geometric.data.graph_store import EdgeLayout
from CustomSampler import InMemorySampler
import time
import cProfile
import pstats
from pathlib import Path

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from evaluate import evaluate
from Training import Trainer, put_nodeLoader_args_map


def main():
    # Get dataset
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())      
    graph = dataset[0]
    train_idx = torch.where(graph.test_mask)[0]    
    
    # create stores
    fstore = InMemoryFeatureStore() 
    fstore["node", "x", None] = graph.x
    fstore["node", "y", None] = graph.y 
    N = graph.x.shape[0]
    edge_index = graph.edge_index
    row, col = edge_index[0].contiguous(), edge_index[1].contiguous()
    
    NODE_TYPE = "node"
    EDGE_TYPE = (NODE_TYPE, "to", NODE_TYPE)  # keep consistent with FeatureStore keys
    LAYOUT = EdgeLayout.COO
    gstore = InMemoryGraphStore()
    gstore.put_edge_index((row, col), edge_type=EDGE_TYPE, layout=LAYOUT)

    # create sampler
    r, c = gstore.get_edge_index(edge_type=EDGE_TYPE, layout=LAYOUT)
    sampler = InMemorySampler(gstore, EDGE_TYPE, layout=LAYOUT, undirected=True)
    
    # set seed for reprodicability and create model
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = GCN(in_dim=1433, hidden_dim=16, out_dim=16, nbr_classes=7)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask

    gstore.set_split_masks(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # train model
    snapshot_path = Path(__file__).resolve().parent.parent / "profiles" / "InMemoryMain2_snapshot.pt"
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )

    trainer = Trainer(
        model=model,
        feature_store=fstore,
        graph_store=gstore,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=32,
        nodes_per_epoch=256,
        eval_every_epochs=None,
        log_train_time=True,
        nodeloader_args=nodeloader_args,
    )
    trainer.train(max_epochs=100)


    # evaluate
    evaluate(model, gstore, fstore, sampler, 'test')

    
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
