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
    
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_store,
        graph_store,
        sampler,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        batch_size: int,
        nodes_per_epoch: int = 256,
        max_train_seconds: int = 3600,
    ) -> None:
        # Use dist functions to get rank/world_size rather than manual args
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.gpu_id = torch.cuda.current_device()

        self.feature_store = feature_store
        self.graph_store = graph_store
        self.sampler = sampler
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.nodes_per_epoch = nodes_per_epoch
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0

        self.model = model.to(self.gpu_id)
        
        # Load snapshot before wrapping in DDP
        if os.path.exists(snapshot_path):
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.criterion = torch.nn.CrossEntropyLoss()

    def _load_snapshot(self, snapshot_path: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.rank == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int) -> None:
        # Save from the .module to avoid 'module.' prefix in state_dict
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _get_train_indices(self) -> torch.Tensor:
        # Rank 0 (Global) fetches the split
        if self.rank == 0:
            indices = self.graph_store.get_split(
                self.nodes_per_epoch, # this shuold be the number of ndoe s in the training data
                split="train",
                shuffle=True,
            ).to(torch.long)
        else:
            indices = torch.empty(self.nodes_per_epoch, dtype=torch.long)
        
        # Broadcast the work from Rank 0 to everyone else
        dist.broadcast(indices, src=0)
        
        # Each rank takes its slice (Data Parallelism)
        return indices[self.rank :: self.world_size]

    def _run_batch(self, batch) -> None:
        batch = batch.to(self.gpu_id)
        self.optimizer.zero_grad()
        out = self.model(batch.x, batch.edge_index)
        
        # Filter for seed nodes (Graph GCN specific)
        seed_mask = torch.isin(batch.n_id, batch.input_id)
        loss = self.criterion(out[seed_mask], batch.y[seed_mask])
        
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int) -> None:
        train_indices = self._get_train_indices()
        
        # NodeLoader handles the graph sampling logic
        train_loader = NodeLoader(
            data=(self.feature_store, self.graph_store),
            node_sampler=self.sampler,
            input_nodes=train_indices,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True, # Better performance for multiple epochs
        )
        
        if self.rank == 0:
            print(f"Epoch {epoch} | Steps: {len(train_loader)}")
            
        for batch in train_loader:
            self._run_batch(batch)

    def train(self, max_epochs: int) -> None:
        start_time = time.monotonic()
        for epoch in range(self.epochs_run, max_epochs):
            if time.monotonic() - start_time >= self.max_train_seconds:
                if self.rank == 0:
                    print("Stopping: max training time reached.")
                break
            
            self._run_epoch(epoch)
            
            # Only save and log from Rank 0
            if self.rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

def evaluate(model, graph_store, feature_store, sampler, split: str = "val") -> None:
    model.eval()
    device = next(model.parameters()).device

    


def build_label_map(uri, user, pwd) -> dict[str, int]:
    # ... (Rest of label map code remains same) ...
    pass

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
        )