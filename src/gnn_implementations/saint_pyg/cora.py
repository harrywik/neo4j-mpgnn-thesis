import os
import sys
import json
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch

GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from neo4j_pyg.models import GCN
from training.GraphSAINTTrainer import GraphSAINTTrainer
from benchmarking_tools import Measurer


def main(config: dict):
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    graph = dataset[0]
    dataset_name = 'cora'

    model_args = {
        "in_dim": dataset.num_features,
        "hidden_dim1": 12,
        "hidden_dim2": 12,
        "nbr_classes": dataset.num_classes,
        "init_weights": config["init_weights"],
    }
    model = GCN(**model_args)

    measurer = Measurer(config)
    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {
        "name": "GraphSAINTRandomWalkSampler",
        "batch_size": config["graphsaint_batch_size"],
        "walk_length": config["walk_length"],
        "num_steps": config["num_steps"],
        "sample_coverage": config["sample_coverage"],
    })
    measurer.write_to_configresult("dataset", dataset_name)

    trainer = GraphSAINTTrainer(
        model=model,
        data=graph,
        patience=config["patience"],
        min_delta=config["min_delta"],
        lr=config["lr"],
        batch_size=config["graphsaint_batch_size"],
        walk_length=config["walk_length"],
        num_steps=config["num_steps"],
        sample_coverage=config["sample_coverage"],
        measurer=measurer,
        cpu_monitor_interval=1 if config["logg_cpu_utilization"] else None,
        max_validation_size=config["max_validation_size"],
        max_test_size=config["max_test_size"],
    )

    trainer.train(max_epochs=config["max_epochs"])


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent.parent / "saint_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    main(config)
