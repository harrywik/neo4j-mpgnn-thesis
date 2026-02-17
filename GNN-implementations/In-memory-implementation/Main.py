import sys
import json
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
from torch import optim
from torch_geometric.data.graph_store import EdgeLayout
import time
import cProfile
import pstats
from pathlib import Path

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from evaluate import evaluate
from models.GCN import GCN
from feature_stores.InMemoryFeatureStore import InMemoryFeatureStore
from graph_stores.InMemoryGraphStore import InMemoryGraphStore
from samplers.InMemorySampler import InMemorySampler
from Training import Trainer, put_nodeLoader_args_map


def main(config: dict):
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
    model = GCN(in_dim=1433, hidden_dim1=16, hidden_dim2=16, nbr_classes=7)
    criterion = nn.CrossEntropyLoss()
    lr = config.get("lr", 0.01)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
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
        batch_size=config.get("batch_size"),
        nodes_per_epoch=config.get("nodes_per_epoch"),
        eval_every_epochs=config.get("eval_every_epochs"),
        eval_every_batches=config.get("eval_every_batches"),
        log_train_time=config.get("log_train_time"),
        nodeloader_args=nodeloader_args,
    )
    trainer.train(max_epochs=config.get("max_epochs"))


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
        config_path = BASE_DIR.parent / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r") as f:
            config = json.load(f)

        main(config)
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))                           # overwrites
        with txt_path.open("w") as f:                              # overwrites
            stats.stream = f
            stats.print_stats(50)

        print(f"wrote {prof_path} and {txt_path}")
    else:
        config_path = Path(__file__).resolve().parent.parent / "train_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open("r") as f:
            config = json.load(f)

        main(config)
