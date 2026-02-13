import os
import time
import math
import torch

from evaluate import evaluate
from torch_geometric.loader import NodeLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from Training import put_nodeLoader_args_map


class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_store,
        graph_store,
        sampler,
        save_every: int = 50,
        snapshot_path: str | None = None,
        batch_size: int = 500,
        criterion=None,
        lr: float = 1e-2,
        optimizer: torch.optim.Optimizer = None,
        nodes_per_epoch: int | None = None,
        max_train_seconds: int = 3600,
        eval_every_epochs: int | None = None,
        eval_every_batches: int | None = None,
        eval_split: str = "val",
        evaluate_fn=None,
        log_train_time: bool = False,
        nodeloader_args: dict | None = None,
    ) -> None:
        # Use dist functions to get rank/world_size rather than manual args
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.gpu_id = torch.cuda.current_device()

        self.feature_store = feature_store
        self.graph_store = graph_store
        self.sampler = sampler
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.nodes_per_epoch = nodes_per_epoch
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0
        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
        self.eval_every_epochs = eval_every_epochs
        self.eval_every_batches = eval_every_batches
        self.eval_split = eval_split
        self.evaluate_fn = evaluate if evaluate_fn is None else evaluate_fn
        self.log_train_time = log_train_time
        self.nodeloader_args = nodeloader_args or put_nodeLoader_args_map(
            pickle_safe=False,
            num_workers=0,
            prefetch_factor=2,
            filter_per_worker=False,
            persistent_workers=False,
            pin_memory=False,
            shuffle=True,
        )

        self.model = model.to(self.gpu_id)
        
        # Load snapshot before wrapping in DDP
        if self.snapshot_path and os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path: str) -> None:
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.rank == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int) -> None:
        if not self.snapshot_path:
            return
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
                self.nodes_per_epoch,
                split="train",
                shuffle=True,
            ).to(torch.long)
        else:
            n = self.nodes_per_epoch or 0
            indices = torch.empty(n, dtype=torch.long)
        
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
        if self.nodeloader_args["pickle_safe"]:
            train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args["shuffle"],
                filter_per_worker=self.nodeloader_args["filter_per_worker"],
                num_workers=self.nodeloader_args["num_workers"],
                persistent_workers=self.nodeloader_args["persistent_workers"],
                prefetch_factor=self.nodeloader_args["prefetch_factor"],
                pin_memory=self.nodeloader_args["pin_memory"],
            )
        else:
            train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args["shuffle"],
                pin_memory=self.nodeloader_args["pin_memory"],
            )
        
        if self.rank == 0:
            num_steps = int(math.ceil(train_indices.numel() / self.batch_size)) if self.batch_size > 0 else 0
            print(f"Epoch {epoch} | Steps: {num_steps}")
            
        for batch_idx, batch in enumerate(train_loader):
            self._run_batch(batch)
            eval_batches = self.eval_every_batches or num_steps
            if (
                self.rank == 0
                and self.eval_every_epochs is not None
                and self.eval_every_epochs > 0
                and eval_batches > 0
                and (epoch % self.eval_every_epochs == 0)
                and ((batch_idx + 1) % eval_batches == 0)
            ):
                self.evaluate_fn(
                    self.model.module,
                    self.graph_store,
                    self.feature_store,
                    self.sampler,
                    self.eval_split,
                )
                self.model.train()

    def train(self, max_epochs: int) -> None:
        start_time = time.monotonic()
        for epoch in range(self.epochs_run, max_epochs):
            if time.monotonic() - start_time >= self.max_train_seconds:
                if self.rank == 0:
                    print("Stopping: max training time reached.")
                break
            
            self._run_epoch(epoch)
            
            # Only save and log from Rank 0
            if self.rank == 0 and self.snapshot_path and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

        if self.rank == 0 and self.log_train_time:
            duration = time.monotonic() - start_time
            print(f"Training duration: {duration:.2f}s")


def put_nodeLoader_args_map(
    pickle_safe: bool | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    filter_per_worker: bool | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    shuffle: bool | None = None,
) -> dict:
    """Helper function to create a consistent dictionary of NodeLoader arguments.
    If GNN implementations are not pickle safe, pickle safe has to be set to False,
    and all argumenets setting multiple workers configuration can be ignored"""
    return {
        "pickle_safe": bool(pickle_safe) if pickle_safe is not None else False,
        "num_workers": int(num_workers) if num_workers is not None else 0,
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else 2,
        "filter_per_worker": bool(filter_per_worker) if filter_per_worker is not None else False,
        "persistent_workers": bool(persistent_workers) if persistent_workers is not None else False,
        "pin_memory": bool(pin_memory) if pin_memory is not None else False,
        "shuffle": bool(shuffle) if shuffle is not None else True,
    }