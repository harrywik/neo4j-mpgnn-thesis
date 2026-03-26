import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from neo4j_pyg.models import GCN
from training.GraphSAINTTrainer import GraphSAINTTrainer
from neo4j_pyg.feature_stores import Neo4jNoCacheFS
from neo4j_pyg.graph_stores import Neo4SingleGS
from neo4j_pyg.samplers import Neo4jNeighborSampler, Neo4jGraphSAINTRandomWalkSampler
from Neo4jConnection import Neo4jConnection
from benchmarking_tools import Measurer, QueryProfileAccumulator


# All GraphSAINT implementations share a single config so that hyperparameters
# are always kept in sync.
SHARED_CONFIG_PATH = GNN_IMPL_DIR.parent / "saint_config.json"


def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]

    profile = config["profile"]
    profile_accumulator = QueryProfileAccumulator() if profile else None

    measurer = Measurer(config, profile_accumulator=profile_accumulator)

    dataset_name = "cora"

    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_store = Neo4jNoCacheFS(
        driver,
        measurer=measurer,
        database_name="neo4j",
        dataset_name=dataset_name,
        feature_property="embedding",
        nodeid_property="id",
        split_property_name="split",
        split_property_type="str",
        target_property="subject",
        feature_property_type="byte[]",
        node_label="Paper",
    )
    graph_store = Neo4SingleGS(
        driver=driver,
        measurer=measurer,
        database_name="neo4j",
        dataset_name=dataset_name,
        feature_property="embedding",
        nodeid_property="id",
        split_property_name="split",
        split_property_type="str",
        target_property="subject",
        profile_accumulator=profile_accumulator,
    )

    train_loader = Neo4jGraphSAINTRandomWalkSampler(
        graph_store=graph_store,
        feature_store=feature_store,
        batch_size=config["graphsaint_batch_size"],
        walk_length=config["walk_length"],
        num_steps=config["num_steps"],
        sample_coverage=config["sample_coverage"],
        node_label="Paper",
        measurer=measurer,
        profile=profile,
    )

    eval_sampler = Neo4jNeighborSampler(
        graph_store,
        num_neighbors=[10, 5],
        node_label="Paper",
    )

    model_args = {
        "in_dim": 1433,
        "hidden_dim1": 12,
        "hidden_dim2": 12,
        "nbr_classes": 7,
        "init_weights": config["init_weights"],
    }
    model = GCN(**model_args)

    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {
        "name": "Neo4jGraphSAINTRandomWalkSampler",
        "batch_size": config["graphsaint_batch_size"],
        "walk_length": config["walk_length"],
        "num_steps": config["num_steps"],
        "sample_coverage": config["sample_coverage"],
    })
    measurer.write_to_configresult("feature_store", "Neo4jNoCacheFS")
    measurer.write_to_configresult("graph_store", "Neo4SingleGS")
    measurer.write_to_configresult("dataset", dataset_name)

    trainer = GraphSAINTTrainer(
        model=model,
        data=(feature_store, graph_store),
        train_loader=train_loader,
        eval_sampler=eval_sampler,
        patience=config["patience"],
        min_delta=config["min_delta"],
        lr=config["lr"],
        batch_size=config["graphsaint_batch_size"],
        measurer=measurer,
        cpu_monitor_interval=1 if config["logg_cpu_utilization"] else None,
        max_validation_size=config["max_validation_size"],
        max_test_size=config["max_test_size"],
    )

    trainer.train(max_epochs=config["max_epochs"])


if __name__ == "__main__":
    if not SHARED_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Shared GraphSAINT config not found: {SHARED_CONFIG_PATH}")
    with open(SHARED_CONFIG_PATH, "r") as f:
        config = json.load(f)

    main(config)
