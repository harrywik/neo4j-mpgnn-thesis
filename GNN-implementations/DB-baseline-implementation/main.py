import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from models import GCN
from Training import Trainer, put_nodeLoader_args_map
from feature_stores import NoCacheFeatureStore
from graph_stores import BaseLineGS
from samplers import UniformSampler
from Neo4jConnection import Neo4jConnection
from Measurer import Measurer

def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    
    measurer = Measurer(config)
    
    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_store = NoCacheFeatureStore(driver, measurer=measurer)
    graph_store = BaseLineGS(driver) 
    sampler = UniformSampler(graph_store, num_neighbors=[10, 5])
    
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 32, 7)
    lr = config.get("lr", 1e-2)

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer,
        lr=lr
    )

    trainer.train(max_epochs=config.get("max_epochs"))
    

if __name__ == "__main__":
    config = "GNN-implementations/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)

