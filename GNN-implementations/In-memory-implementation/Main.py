import sys
import json
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn as nn
import torch
from torch import optim
from torch_geometric.data.graph_store import EdgeLayout

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from evaluate import evaluate
from models import GCN
from feature_stores import InMemoryFeatureStore
from graph_stores import InMemoryGS
from samplers import InMemorySampler
from Training import Trainer, put_nodeLoader_args_map
from benchmarking_tools import Measurer

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
    gstore = InMemoryGS()
    gstore.put_edge_index((row, col), edge_type=EDGE_TYPE, layout=LAYOUT)

    # create sampler
    r, c = gstore.get_edge_index(edge_type=EDGE_TYPE, layout=LAYOUT)
    sampler = InMemorySampler(gstore, EDGE_TYPE, num_neighbors=[10, 5], layout=LAYOUT, undirected=True)
    
    # set seed for reprodicability and create model
    model = GCN(in_dim=1433, hidden_dim1=32, hidden_dim2=16, nbr_classes=7)
    criterion = nn.CrossEntropyLoss()
    lr = config.get("lr")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_mask = graph.train_mask
    test_mask = graph.test_mask
    val_mask = graph.val_mask

    gstore.set_split_masks(train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # train model
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )
    
    measurer = Measurer(config)
    
    patience = 20

    trainer = Trainer(
        model=model,
        feature_store=fstore,
        graph_store=gstore,
        measurer=measurer,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        patience=patience, #config.get("patience"),
        min_delta=0.001,#config.get("min_delta"),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
    )
    trainer.train(max_epochs=config.get("max_epochs"))
    
if __name__ == "__main__":
    config = "GNN-implementations/train_config.json"
    if not Path(config).exists():
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)
