import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from torch_geometric.datasets import Planetoid
import torch

load_dotenv()
# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from neo4j_pyg.models import GCN
from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.feature_stores import Neo4jCachedFS, Neo4jNoCacheFS
from neo4j_pyg.graph_stores import Neo4jMultiGS, Neo4SingleGS
from neo4j_pyg.samplers import Neo4jNeighborSampler
from Neo4jConnection import Neo4jConnection
from benchmarking_tools import Measurer, QueryProfileAccumulator

def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]

    profile = config.get("profile", False)
    profile_accumulator = QueryProfileAccumulator() if profile else None

    measurer = Measurer(config, profile_accumulator=profile_accumulator)
    
    dataset_name = "cora"
    database_name = "neo4j"

    driver = Neo4jConnection(uri, user, password).get_driver()

    feature_property = "embedding_bytes"
    feature_property_type = "byte[]"

    feature_store = Neo4jNoCacheFS(driver, measurer=measurer, database_name="neo4j", dataset_name=dataset_name, feature_property=feature_property, nodeid_property="id", split_property_name="split", split_property_type="str", target_property="subject", feature_property_type=feature_property_type, profile=profile, profile_accumulator=profile_accumulator, node_label="Paper")
    graph_store = Neo4SingleGS(driver=driver, measurer=measurer, database_name="neo4j", dataset_name=dataset_name, feature_property=feature_property, nodeid_property="id", split_property_name="split", split_property_type="str", target_property="subject", profile_accumulator=profile_accumulator)
    num_neighbors = [10, 5]
    sampler = Neo4jNeighborSampler(graph_store, num_neighbors=num_neighbors, profile=profile, node_label="Paper")

    model_args = {"in_dim": 1433, "hidden_dim1": 12, "hidden_dim2": 12, "nbr_classes": 7, "init_weights": config.get("init_weights")}
    model = GCN(**model_args)
    # model = BigGCN(in_dim=1433, nbr_classes=7, hidden_dim=2048)


    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "NeighborSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "NoCacheFeatureStore")
    measurer.write_to_configresult("graph_store", "Neo4SingleGS")
    measurer.write_to_configresult("dataset", dataset_name)

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=config.get("shuffle"),
    )
    measurer.write_to_configresult("nodeloader_args", nodeloader_args)
    

    trainer = Trainer(
        model=model,
        data=(feature_store, graph_store),
        sampler=sampler,
        min_delta=config.get('min_delta'),
        lr=config.get('lr'),
        patience=config.get('patience'),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer,
        cpu_monitor_interval=1 if config.get("logg_cpu_utilization", True) else None,
        max_training_size=config.get("max_training_size"),
        max_validation_size=config.get("max_validation_size"),
        max_test_size=config.get("max_test_size"),
    )

    trainer.train(max_epochs=config.get("max_epochs"))
    

if __name__ == "__main__":
    config = "src/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)

