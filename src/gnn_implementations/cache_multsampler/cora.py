import os
import sys
import json
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from training.evaluate import evaluate
from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.models.GCN import GCN
from neo4j_pyg.feature_stores import Neo4jNoCacheFS, Neo4jCachedFS
from neo4j_pyg.graph_stores import Neo4jMultiGS
from neo4j_pyg.samplers import Neo4jNeighborSampler
from benchmarking_tools import Measurer

def main(config: dict):
    # Demo local user with unsecure passwd
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    feature_store = Neo4jCachedFS(
        uri=uri,
        user=user,
        pwd=password,
        dataset_name="neo4j",
        feature_property="embedding",
        target_property="subject",
        nodeid_property="id",
        # split_property_name="split",
        # split_property_type="str",
        feature_property_type="byte[]",
        node_label="Paper",
    )
        
    graph_store = Neo4jMultiGS(
        uri,
        user,
        password,
        dataset_name="neo4j",
        split_property_name="split",
        split_property_type="str",
        nodeid_property="id",
    )
    num_neighbors = [10, 5]
    sampler = Neo4jNeighborSampler(graph_store, num_neighbors=num_neighbors, node_label="Paper")
    model_args = {"in_dim": 1433, "hidden_dim1": 12, "hidden_dim2": 12, "nbr_classes": 7, "init_weights": config.get("init_weights")}
    model = GCN(**model_args)
    measurer = Measurer(config)
    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "NeighborSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "Neo4jCachedFS")
    measurer.write_to_configresult("graph_store", "Neo4jMultiGS")

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=True,
        shuffle=True,
        num_workers=5,        
        prefetch_factor=3,
        filter_per_worker=False, # True -> 
        persistent_workers=True,
        pin_memory=False,
    )
    measurer.write_to_configresult("nodeloader_args", nodeloader_args)

    trainer = Trainer(
        model=model,
        data=(feature_store, graph_store),
        sampler=sampler,
        lr=config.get("lr"),
        patience=config.get('patience'),
        min_delta=config.get('min_delta'),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer
    )

    trainer.train(max_epochs=config.get("max_epochs"))


if __name__ == "__main__":
    config = "src/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)
