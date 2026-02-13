import os
import time
import torch

from evaluate import evaluate
from torch_geometric.loader import NodeLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


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
        torch.dist.broadcast(indices, src=0)
        
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
