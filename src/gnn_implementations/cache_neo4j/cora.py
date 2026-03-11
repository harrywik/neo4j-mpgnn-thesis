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

from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.models.GCN import GCN
from neo4j_pyg.feature_stores.PageRankCacheFeatureStore import PageRankCacheFeatureStore
from neo4j_pyg.graph_stores import BaseLineGS
from neo4j_pyg.samplers.UniformSampler import UniformSampler
from benchmarking_tools import Measurer
from Neo4jConnection import Neo4jConnection

def main(config: dict):
    # Demo local user with unsecure passwd
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    measurer = Measurer(config)

    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_store = PageRankCacheFeatureStore(
        driver,
        dataset_name="neo4j",
        feature_property="embedding",
        target_property="subject",
        nodeid_property="id",
        feature_property_type="byte[]",
    )
    
    graph_store = BaseLineGS(
        driver,
        dataset_name="neo4j",
        split_property_name="split",
        split_property_type="str",
        nodeid_property="id",
    )
    num_neighbors = [10, 5]
    sampler = UniformSampler(graph_store, num_neighbors=num_neighbors)
    split_ratios = [0.6, 0.2, 0.2]
    model_args = {"in_dim": 1433, "hidden_dim1": 32, "hidden_dim2": 16, "nbr_classes": 7}
    model = GCN(**model_args)
    lr = config.get("lr", 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    feature_store_version = config.get("feature_store_version", "002")
    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "UniformSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "PageRankCacheFeatureStore")
    measurer.write_to_configresult("feature_store_version", feature_store_version)
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
        data=(feature_store, graph_store),
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        patience=config.get("patience"),
        min_delta=config.get("min_delta"),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer,
    )

    trainer.train(max_epochs=config.get("max_epochs"))


if __name__ == "__main__":
    config = "src/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)

