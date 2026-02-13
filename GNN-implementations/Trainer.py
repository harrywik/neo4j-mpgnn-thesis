import time
import math
import torch

from torch_geometric.loader import NodeLoader
from torch import optim
from evaluate import evaluate


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_store,
        graph_store,
        sampler,
        save_every: int = 50,
        snapshot_path: str | None = None,
        batch_size: int = 500,
        criterion = None,
        optimizer: torch.optim.Optimizer = None,
        nodes_per_epoch: int | None = None,
        max_train_seconds: int = 3600,
        device: str = "cpu",
        rank: int = 0,
        world_size: int = 1,
        eval_every_epochs: int | None = None,
        eval_every_batches: int | None = None,
        eval_split: str = "val",
        evaluate_fn=None,
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.sampler = sampler
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            model.parameters(), lr=0.01, weight_decay=5e-4
        )

        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.nodes_per_epoch = nodes_per_epoch
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0        
        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
        self.device = torch.device(device)
        self.rank = rank
        self.world_size = world_size
        self.eval_every_epochs = eval_every_epochs
        self.eval_every_batches = eval_every_batches
        self.eval_split = eval_split
        self.evaluate_fn = evaluate if evaluate_fn is None else evaluate_fn

        self.model.to(self.device)

    def _save_snapshot(self, epoch: int) -> None:
        if not self.snapshot_path:
            return
        # Save from the .module to avoid 'module.' prefix in state_dict
        model = self.model.module if hasattr(self.model, "module") else self.model
        snapshot = {
            "MODEL_STATE": model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _get_train_indices(self) -> torch.Tensor:
        indices = self.graph_store.get_split(
            self.nodes_per_epoch,
            split="train",
            shuffle=True,
        ).to(torch.long)

        # Each rank takes its slice (Data Parallelism)
        return indices[self.rank :: self.world_size]

    def _run_batch(self, batch) -> None:
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(batch.x, batch.edge_index)
        
        # Filter for seed nodes (Graph GCN specific)
        seed_mask = torch.isin(batch.n_id, batch.input_id)
        loss = self.criterion(out[seed_mask], batch.y[seed_mask])
        
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int) -> None:
        self.model.train()
        train_indices = self._get_train_indices()
        
        # NodeLoader handles the graph sampling logic
        train_loader = NodeLoader(
            data=(self.feature_store, self.graph_store),
            node_sampler=self.sampler,
            input_nodes=train_indices,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
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
                    self.model,
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
