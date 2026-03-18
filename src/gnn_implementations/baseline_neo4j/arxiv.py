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

from neo4j_pyg.models import GCN
from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.feature_stores import Neo4jNoCacheFS
from neo4j_pyg.graph_stores import Neo4SingleGS
from neo4j_pyg.samplers import Neo4jSampler
from Neo4jConnection import Neo4jConnection
from benchmarking_tools import Measurer, QueryProfileAccumulator

def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]

    profile = config.get("profile", False)
    profile_accumulator = QueryProfileAccumulator() if profile else None

    measurer = Measurer(config, profile_accumulator=profile_accumulator)

    dataset_name = "arxiv"
    database_name = "arxiv"
    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_property = "features"
    target_property = "category"
    nodeid_property = "nodeId"
    split_property_name = "split"
    split_property_type = "int"
    feature_property_type = "f64[]"
    feature_store = Neo4jNoCacheFS(driver, measurer=measurer, database_name=database_name, dataset_name=dataset_name, feature_property=feature_property, nodeid_property=nodeid_property, split_property_name=split_property_name, split_property_type=split_property_type, target_property=target_property, feature_property_type=feature_property_type, profile=profile, profile_accumulator=profile_accumulator)
    graph_store = Neo4SingleGS(driver=driver, measurer=measurer, database_name=database_name, dataset_name=dataset_name, feature_property=feature_property, nodeid_property=nodeid_property, split_property_name=split_property_name, split_property_type=split_property_type, target_property=target_property, profile_accumulator=profile_accumulator)
    num_neighbors = [10, 5]
    sampler = Neo4jSampler(graph_store, num_neighbors=num_neighbors, profile=profile)

    model_args = {"in_dim": 128, "hidden_dim1": 32, "hidden_dim2": 32, "nbr_classes": 40, "init_weights": config.get("init_weights")}
    model = GCN(**model_args)

    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "Neo4jSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "NoCacheFeatureStore")
    measurer.write_to_configresult("graph_store", "Neo4SingleGS")
    measurer.write_to_configresult("lr", config.get("lr"))
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
        patience=config.get('patience'),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer,
        lr=config.get('lr'),
    )

    trainer.train(max_epochs=config.get("max_epochs"))
    

if __name__ == "__main__":
    config = "src/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)

