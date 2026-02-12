import time
import os
from typing import Dict
from feature_stores.v002 import Neo4jFeatureStore as Neo4jFeatureStore002
from feature_stores.v001 import Neo4jFeatureStore as Neo4jFeatureStore001
from feature_stores.v000 import Neo4jFeatureStore as Neo4jFeatureStore000
from Neo4jGraphStore import Neo4jGraphStore
from Neo4jSampler import Neo4jSampler
from torch_geometric.loader import NodeLoader
from Model import GCN
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import cProfile
import pstats
import argparse
from pathlib import Path

from remote_backend import Trainer, evaluate

def ddp_setup() -> None:
    # Check if we are actually in a distributed environment
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("Not in distributed mode; skipping hardware setup.")
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    
    dist.init_process_group(
        backend="nccl", 
        init_method="env://", 
        rank=rank, 
        world_size=world_size
    )
    
    # Optional, ensure all processes are synced before starting
    dist.barrier()
    
    
def build_label_map(uri, user, pwd) -> dict[str, int]:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    q = "MATCH (n) RETURN DISTINCT n.subject AS s ORDER BY s ASC"
    with driver.session() as session:
        labels = [r["s"] for r in session.run(q)]
    driver.close()
    return {s: i for i, s in enumerate(labels)}

def main(
    version_dict: Dict[str, str],
    save_every: int,
    total_epochs: int,
    batch_size: int,
    snapshot_path: str = "snapshot.pt",
    max_train_seconds: int = 3600,
):
    ddp_setup()
    
    # Initialize your stores...
    uri = "bolt://localhost:7687"
    user = "neo4j"
    pwd = "thesis-db-0-pw"
    label_map = build_label_map(uri, user, pwd)

    # Store logic...
    feature_store = Neo4jFeatureStore002(uri, user, pwd, label_map=label_map)
    graph_store   = Neo4jGraphStore(uri, user, pwd)
    sampler       = Neo4jSampler(graph_store, [10, 5])
    
    # Ensure only one process does the DB split to avoid race conditions
    if dist.get_rank() == 0:
        graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    dist.barrier() # Wait for Rank 0 to finish the split

    model = GCN(1433, 32, 16, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    trainer = Trainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        optimizer=optimizer,
        save_every=save_every,
        snapshot_path=snapshot_path,
        batch_size=batch_size,
        max_train_seconds=max_train_seconds,
    )
    
    trainer.train(total_epochs)

    dist.barrier()
    if dist.get_rank() == 0:
        evaluate(trainer.model.module, graph_store, feature_store, sampler, "test")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Provide profiling versions for this experiment.",
        epilog=(
            "Examples:\n"
            "  1. Standard Run (2 GPUs):\n"
            "     torchrun --nproc_per_node=2 your_filename.py --total_epochs 50 --batch_size 64\n\n"
            "  2. Profile Run (2 GPUs):\n"
            "     torchrun --nproc_per_node=2 your_filename.py --profile --total_epochs 5"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--profile", action="store_true", help="Whether or not to run cProfile")
    parser.add_argument(
        "--total_epochs",
        default=10,
        type=int,
        help="Total epochs to train the model (default: 10)",
    )
    parser.add_argument(
        "--save_every",
        default=1,
        type=int,
        help="How often to save a snapshot (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Input batch size on each device (default: 32)",
    )
    parser.add_argument(
        "--snapshot_path",
        default="snapshot.pt",
        type=str,
        help="Path to save training snapshots",
    )
    parser.add_argument(
        "--max_train_seconds",
        default=3600,
        type=int,
        help="Max training time in seconds (default: 3600)",
    )

    # local rank is passed by torch run
    parser.add_argument("--local-rank", "--local_rank", type=int, help="Local rank for distributed training. (Added for torchrun compatibility)")
    parser.add_argument("--feature-store", 
                        type=str, 
                        default="002",
                        choices=["000", "001", "002"],
                        help="Feature store version")
    
    args = parser.parse_args()
    
    # Pack version into dict as expected by main()
    main_args = {
        "feature_store": args.feature_store
    }

    if args.profile:
        # Get rank to avoid file collisions during profiling
        # We use environment variable because dist.init_process_group hasn't run yet
        rank = os.environ.get("RANK", "0") 
        
        BASE_DIR = Path(__file__).resolve().parent
        profiles_dir = BASE_DIR.parent / "profiles"
        folder_name = BASE_DIR.name
        ver = f"feat_store_v{args.feature_store}_rank{rank}"

        target_dir = profiles_dir / folder_name
        target_dir.mkdir(parents=True, exist_ok=True)

        prof_path = target_dir / f"{ver}.prof"
        txt_path  = target_dir / f"{ver}.txt"

        pr = cProfile.Profile()
        pr.enable()
        
        main(
            main_args,
            save_every=args.save_every,
            total_epochs=args.total_epochs,
            batch_size=args.batch_size,
            snapshot_path=args.snapshot_path,
            max_train_seconds=args.max_train_seconds,
            val_patience=args.val_patience,
            val_min_improve=args.val_min_improve,
        )
        
        pr.disable()

        stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(str(prof_path))
        with txt_path.open("w") as f:
            stats.stream = f
            stats.print_stats(50)

        # Only print completion message from one rank to keep terminal clean
        if rank == "0":
            print(f"Profiling complete. Results saved to {target_dir}")
    else:
        main(
            main_args,
            save_every=args.save_every,
            total_epochs=args.total_epochs,
            batch_size=args.batch_size,
            snapshot_path=args.snapshot_path,
            max_train_seconds=args.max_train_seconds,
            val_patience=args.val_patience,
            val_min_improve=args.val_min_improve,
        )