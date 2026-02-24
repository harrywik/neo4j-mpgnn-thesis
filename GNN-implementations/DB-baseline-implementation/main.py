import csv
import os
import sys
import json
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

from models import GCN
from evaluate import evaluate
from Training import Trainer, put_nodeLoader_args_map
from feature_stores import NoCacheFeatureStore
from graph_stores import BaseLineGS
from samplers import UniformSampler
from Neo4jConnection import Neo4jConnection
from Measurer import Measurer
import time

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

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
    )

    trainer.train(max_epochs=config.get("max_epochs"))
    measurer.log_event("program_end", 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Provide profiling versions for this experiment.")

    parser.add_argument(
            "--profile",
            action="store_true",
            default=True,
            help="Whether or not to run cProfile",
        )    
    parser.add_argument(
            "--config",
            type=str,
            default="GNN-implementations/train_config.json",
            help="Path to JSON training config",
        )

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    main(config)

