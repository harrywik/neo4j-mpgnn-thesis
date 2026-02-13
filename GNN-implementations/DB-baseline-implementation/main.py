import sys
from typing import Dict
from Neo4jConnection import Neo4jConnection
from feature_stores.v002 import Neo4jFeatureStore as Neo4jFeatureStore002
from feature_stores.v001 import Neo4jFeatureStore as Neo4jFeatureStore001
from feature_stores.v000 import Neo4jFeatureStore as Neo4jFeatureStore000
from Neo4jGraphStore import Neo4jGraphStore
from Neo4jSampler import Neo4jSampler
from torch_geometric.loader import NodeLoader
from Model import GCN
import torch
import numpy as np
import cProfile
import pstats
import argparse
from pathlib import Path

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from evaluate import evaluate
from Trainer import Trainer


def main(version_dict: Dict[str, str]):
    # Demo local user with unsecure passwd
    driver = Neo4jConnection("bolt://localhost:7687", "neo4j", "thesis-db-0-pw").get_driver()
    match version_dict.get("feature_store", "001"):
        case "000":
            feature_store = Neo4jFeatureStore000(driver)
        case "001":
            feature_store = Neo4jFeatureStore001(driver)
        case "002":
            feature_store = Neo4jFeatureStore002(driver)
        case _:
            raise Exception("Must know which impl of `FeatureStore` to use.")
        
    graph_store = Neo4jGraphStore(driver) # Sampler handles all topology
    sampler = Neo4jSampler(graph_store, num_neighbors=[10, 5])
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 16, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(10):
        train_indices = graph_store.get_split(256, split="train", shuffle=True)

        train_loader = NodeLoader(
            data=(feature_store, graph_store), 
            node_sampler=sampler,
            input_nodes=train_indices,
            batch_size=32,
            shuffle=False
        )

        for bi, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed_mask = torch.isin(batch.n_id, batch.input_id)
            loss = criterion(out[seed_mask], batch.y[seed_mask])

            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} batch: {bi} | Loss: {loss:5f}")


    evaluate(model, graph_store, feature_store, sampler, "train")
    evaluate(model, graph_store, feature_store, sampler, "val")
    evaluate(model, graph_store, feature_store, sampler, "test")


if __name__ == "__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Provide profiling versions for this experiment.")

    parser.add_argument("--profile", action="store_true", help="Wheather or not to run cProfile")
    parser.add_argument("--feature-store", 
                        type=str, 
                        default="002",
                        choices=["000", "001", "002"],
                        help="Feature store version")
    
    args = parser.parse_args()
    main_args = {
        "feature_store": args.feature_store
    }

    if args.profile:
        BASE_DIR = Path(__file__).resolve().parent                  # folder containing Main.py
        profiles_dir = BASE_DIR.parent / "profiles"                 # sibling folder named "profile"
        profiles_dir.mkdir(parents=True, exist_ok=True)

        folder_name = BASE_DIR.name                                # e.g. "InMemoryGNNExample"
        ver = f"feat_store_v{main_args['feature_store']}"

        prof_path = profiles_dir / folder_name /f"{ver}.prof"
        txt_path  = profiles_dir / folder_name / f"{ver}.txt"

        pr = cProfile.Profile()
        pr.enable()
        main(main_args)
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))                           # overwrites
        with txt_path.open("w") as f:                              # overwrites
            stats.stream = f
            stats.print_stats(50)

        print(f"wrote {prof_path} and {txt_path}")
    else:
        main(main_args)

