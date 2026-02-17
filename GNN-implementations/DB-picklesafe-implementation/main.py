import os
import sys
import json
from typing import Dict
import torch
import numpy as np
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
from feature_stores.PickleSafeFeatureStore import PickleSafeFeatureStore
from graph_stores.PickleSafeGraphStore import PickleSafeGraphStore
from samplers.UniformSampler import UniformSampler
from Neo4jConnection import Neo4jConnection


def main(version_dict: Dict[str, str], config: dict):
    # Demo local user with unsecure passwd
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    feature_store = PickleSafeFeatureStore(uri, user, password)
        
    graph_store = PickleSafeGraphStore(uri, user, password) 
    sampler = UniformSampler(graph_store, num_neighbors=[10, 5])
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 16, 7)
    lr = config.get("lr", 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=True,
        shuffle=True,
        num_workers=2,          # must be 0 for pickle_safe=True
        prefetch_factor=2,
        filter_per_worker=False,
        persistent_workers=False,
        pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=config.get("batch_size", 500),
        nodes_per_epoch=config.get("nodes_per_epoch"),
        eval_every_epochs=config.get("eval_every_epochs"),
        eval_every_batches=config.get("eval_every_batches"),
        log_train_time=config.get("log_train_time", True),
        nodeloader_args=nodeloader_args,
    )

    trainer.train(max_epochs=config.get("max_epochs", 20))

    evaluate(model, graph_store, feature_store, sampler, "test")


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
    parser.add_argument("--feature-store", 
                        type=str, 
                        default="002",
                        choices=["000", "001", "002"],
                        help="Feature store version")
    
    args = parser.parse_args()
    main_args = {
        "feature_store": args.feature_store
    }

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    if args.profile:
        BASE_DIR = Path(__file__).resolve().parent                  # folder containing Main.py
        profiles_dir = BASE_DIR.parent / "profiles"                 # sibling folder named "profile"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        folder_name = BASE_DIR.name                                # e.g. "InMemoryGNNExample"
        ver = f"feat_store_v{main_args['feature_store']}"

        target_dir = profiles_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        prof_path = target_dir / f"{ver}.prof"
        txt_path  = target_dir / f"{ver}.txt"

        pr = cProfile.Profile()
        pr.enable()
        main(main_args, config)
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))                           # overwrites
        with txt_path.open("w") as f:                              # overwrites
            stats.stream = f
            stats.print_stats(50)

        print(f"wrote {prof_path} and {txt_path}")
    else:
        main(main_args, config)

