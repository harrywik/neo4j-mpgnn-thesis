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

from models.GCN import GCN
from evaluate import evaluate
from Training import Trainer, put_nodeLoader_args_map
from feature_stores.NoCacheFeatureStore import NoCacheFeatureStore
from graph_stores.BaseLineGS import BaseLineGS
from samplers.UniformSampler import UniformSampler
from Neo4jConnection import Neo4jConnection

def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]
    
    driver = Neo4jConnection(uri, user, password).get_driver()
    feature_store = NoCacheFeatureStore(driver)
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

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")
    with open(args.config, "r") as f:
        config = json.load(f)

    if args.profile:
        BASE_DIR = Path(__file__).resolve().parent                  # folder containing Main.py
        profiles_dir = BASE_DIR.parent / "profiles"                 # sibling folder named "profile"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        folder_name = BASE_DIR.name                                # e.g. "InMemoryGNNExample"

        target_dir = profiles_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        prof_path = target_dir / f"{folder_name}.prof"
        txt_path  = target_dir / f"{folder_name}.txt"

        pr = cProfile.Profile()
        pr.enable()
        main(config)
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))                           # overwrites
        with txt_path.open("w") as f:                              # overwrites
            stats.stream = f
            stats.print_stats(50)

        print(f"wrote {prof_path} and {txt_path}")
    else:
        main(config)

