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
from benchmarking_tools import Measurer

def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    
    measurer = Measurer(config)
    
    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_store = NoCacheFeatureStore(driver, measurer=measurer)
    graph_store = BaseLineGS(driver) 
    num_neighbors = [10, 5]
    sampler = UniformSampler(graph_store, num_neighbors=num_neighbors)
    
    split_ratios = [0.6, 0.2, 0.2]
    graph_store.train_val_test_split_db(split_ratios)
    model_args = {"in_dim": 1433, "hidden_dim1": 32, "hidden_dim2": 32, "nbr_classes": 7}
    model = GCN(**model_args)
    lr = config.get("lr", 1e-2)

    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "UniformSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "NoCacheFeatureStore")
    measurer.write_to_configresult("graph_store", "BaseLineGS")
    measurer.write_to_configresult("train_val_test_split", split_ratios)
    measurer.write_to_configresult("lr", lr)

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )
    measurer.write_to_configresult("nodeloader_args", nodeloader_args)
    

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        min_delta=config.get('min_delta'),
        patience=config.get('patience'),
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

