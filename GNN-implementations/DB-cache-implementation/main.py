import os
import sys
import json
from typing import Dict
from Neo4jConnection import Neo4jConnection
from feature_stores.v002 import Neo4jFeatureStore as Neo4jFeatureStore002
from feature_stores.v001 import Neo4jFeatureStore as Neo4jFeatureStore001
from feature_stores.v000 import Neo4jFeatureStore as Neo4jFeatureStore000
from Neo4jGraphStore import Neo4jGraphStore
from Neo4jSampler import Neo4jSampler
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
from Model import GCN


def main(version_dict: Dict[str, str], config: dict):
    # Demo local user with unsecure passwd
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    driver = Neo4jConnection(uri, user, password).get_driver()
    match version_dict.get("feature_store", "001"):
        case "000":
            feature_store = Neo4jFeatureStore000(driver)
        case "001":
            feature_store = Neo4jFeatureStore001(
                driver,
                cache_size=config.get("cache_size", 3000),
                label_cache_size=config.get("label_cache_size"),
                batch_cache_size=config.get("batch_cache_size", 64),
                db_batch_size=config.get("db_batch_size", 1000),
            )
        case "002":
            feature_store = Neo4jFeatureStore002(driver)
        case _:
            raise Exception("Must know which impl of `FeatureStore` to use.")
        
    graph_store = Neo4jGraphStore(driver) 
    sampler = Neo4jSampler(graph_store, num_neighbors=[10, 5])
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 16, 7)
    lr = config.get("lr", 1e-2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    if config.get("prewarm_cache", True) and hasattr(feature_store, "prewarm"):
        split = config.get("prewarm_split", "train")
        prewarm_n = config.get("prewarm_max_nodes", 5000)
        warm_ids = graph_store.get_split(n=prewarm_n, split=split, shuffle=True)
        feature_store.prewarm(
            warm_ids.tolist(),
            include_embeddings=config.get("prewarm_embeddings", True),
            include_labels=config.get("prewarm_labels", True),
        )

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=True,
    )

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=config.get("batch_size"),
        nodes_per_epoch=config.get("nodes_per_epoch"),
        eval_every_epochs=config.get("eval_every_epochs"),
        eval_every_batches=config.get("eval_every_batches"),
        log_train_time=config.get("log_train_time", True),
        nodeloader_args=nodeloader_args,
    )

    trainer.train(max_epochs=config.get("max_epochs"))

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

