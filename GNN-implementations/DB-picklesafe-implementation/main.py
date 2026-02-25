import os
import sys
import json
from typing import Dict
import torch
import cProfile
import pstats
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from evaluate import evaluate
from Training import Trainer, put_nodeLoader_args_map
from models.GCN import GCN
from feature_stores import PickleSafeFeatureStore
from graph_stores import PickleSafeGS
from samplers import UniformSampler
from benchmark_tools import Measurer

def main(config: dict):
    # Demo local user with unsecure passwd
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    feature_store = PickleSafeFeatureStore(uri, user, password)
        
    graph_store = PickleSafeGS(uri, user, password) 
    num_neighbors = [10, 5]
    sampler = UniformSampler(graph_store, num_neighbors=num_neighbors)
    split_ratios = [0.6, 0.2, 0.2]
    graph_store.train_val_test_split_db(split_ratios)
    model_args = {"in_dim": 1433, "hidden_dim1": 32, "hidden_dim2": 16, "nbr_classes": 7}
    model = GCN(**model_args)
    lr = config.get("lr", 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    measurer = Measurer(config)
    measurer.write_to_configresult("model", {"name": "GCN", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "UniformSampler", "num_neighbors": num_neighbors})
    measurer.write_to_configresult("feature_store", "PickleSafeFeatureStore")
    measurer.write_to_configresult("graph_store", "PickleSafeGS")
    measurer.write_to_configresult("train_val_test_split", split_ratios)
    measurer.write_to_configresult("lr", lr)

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=True,
        shuffle=True,
        num_workers=2,          # must be 0 for pickle_safe=True
        prefetch_factor=2,
        filter_per_worker=True,
        persistent_workers=True,
        pin_memory=False,
    )
    measurer.write_to_configresult("nodeloader_args", nodeloader_args)

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=config.get("batch_size", 100),
        nodeloader_args=nodeloader_args,
        measurer=measurer
    )

    trainer.train(max_epochs=config.get("max_epochs", 20))


if __name__ == "__main__":
    config = "GNN-implementations/train_config.json"
    if not os.path.exists(config):
        raise FileNotFoundError(f"Config not found: {config}")
    with open(config, "r") as f:
        config = json.load(f)

    main(config)
