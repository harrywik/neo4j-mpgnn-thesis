import os
import sys
import json
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
from torch import optim

# Allow running this file directly by adding GNN-implementations to sys.path
# GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
# if str(GNN_IMPL_DIR) not in sys.path:
#     sys.path.insert(0, str(GNN_IMPL_DIR))

from neo4j_pyg.models import GCN
from training.Training import Trainer, put_nodeLoader_args_map
from benchmarking_tools import Measurer


def main(config: dict):
    # Get dataset
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())      
    graph = dataset[0]
    dataset_name = 'cora'
    train_indices = torch.where(graph.train_mask.reshape(-1))[0].tolist()


    # set seed for reprodicability and create model
    model = GCN(in_dim=1433, hidden_dim1=32, hidden_dim2=32, nbr_classes=7)
    criterion = nn.CrossEntropyLoss()
    lr = config.get("lr")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    # train model
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )
    
    num_neighbors = [10, 5]
    
    measurer = Measurer(config)
    

    trainer = Trainer(
        model=model,
        data=graph,
        measurer=measurer,
        optimizer=optimizer,
        criterion=criterion,
        train_indices=train_indices,
        patience=config.get("patience"),
        min_delta=0.001,#config.get("min_delta"),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        num_neighbors=num_neighbors,
    )
    trainer.train(max_epochs=config.get("max_epochs"))
    
if __name__ == "__main__":
    config = "src/train_config.json"
    if not Path(config).exists():
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)
