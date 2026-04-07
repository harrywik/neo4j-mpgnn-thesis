import cProfile
import os
import pstats
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import NodeLoader

from benchmarking_tools import start_cpu_monitor, start_cpu_burst
from src.training.evaluate import evaluate
from src.training.Training import put_nodeLoader_args_map


class DistributedTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        feature_store,
        graph_store,
        sampler,
        measurer=None,
        save_every: int = 50,
        snapshot_path: str | None = None,
        batch_size: int = 500,
        criterion=None,
        lr: float = 1e-2,
        optimizer: torch.optim.Optimizer = None,
        nodes_per_epoch: int | None = None,
        max_train_seconds: int = 3600,
        eval_every_epochs: int | None = None,
        eval_split: str = "val",
        evaluate_fn=None,
        log_train_time: bool = False,
        nodeloader_args: dict | None = None,
    ) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            self.device = torch.device("cuda", local_rank)
        else:
            self.device = torch.device("cpu")

        self.feature_store = feature_store
        self.graph_store = graph_store
        self.sampler = sampler
        self.measurer = measurer  # None on non-rank-0 processes
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.nodes_per_epoch = nodes_per_epoch
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0
        self.criterion = torch.nn.CrossEntropyLoss() if criterion is None else criterion
        self.eval_every_epochs = eval_every_epochs
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

        # Fetch and shard training indices once (rank 0 broadcasts to all ranks)
        train_indices = self._get_train_indices()
        self._log("nbr_training_datapoints", len(train_indices))

        # Build the NodeLoader once, matching Trainer's structure
        if self.nodeloader_args["pickle_safe"]:
            self.train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args["shuffle"],
                filter_per_worker=self.nodeloader_args["filter_per_worker"],
                num_workers=self.nodeloader_args["num_workers"],
                persistent_workers=self.nodeloader_args["persistent_workers"],
                multiprocessing_context="spawn",
                prefetch_factor=self.nodeloader_args["prefetch_factor"],
                pin_memory=self.nodeloader_args["pin_memory"],
            )
        else:
            self.train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args["shuffle"],
                pin_memory=self.nodeloader_args["pin_memory"],
            )

        self.model = model.to(self.device)

        # Load snapshot before wrapping in DDP
        if self.snapshot_path and os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)

        device_ids = [self.device.index] if self.device.type == "cuda" else None
        self.model = DDP(self.model, device_ids=device_ids)

        # Create optimizer after DDP wrapping so it tracks the DDP parameters
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=5e-4
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, event: str, value=1) -> None:
        """Log an event — no-op when measurer is None (non-rank-0 processes)."""
        if self.measurer is not None:
            self.measurer.log_event(event, value)

    def _set_phase(self, phase: str) -> None:
        if self.measurer is not None:
            self.measurer.set_phase(phase)

    def _get_train_indices(self) -> torch.Tensor:
        # Rank 0 fetches the split and broadcasts its size first
        if self.rank == 0:
            indices = self.graph_store.get_split(
                self.nodes_per_epoch,
                split="train",
                shuffle=True,
            ).to(torch.long)
            size_tensor = torch.tensor([indices.numel()], dtype=torch.long)
        else:
            size_tensor = torch.tensor([0], dtype=torch.long)

        dist.broadcast(size_tensor, src=0)
        n = size_tensor.item()

        if self.rank != 0:
            indices = torch.empty(n, dtype=torch.long)

        dist.broadcast(indices, src=0)
        return indices[self.rank :: self.world_size]

    def _load_snapshot(self, snapshot_path: str) -> None:
        snapshot = torch.load(snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        if self.rank == 0:
            print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _save_snapshot(self, epoch: int) -> None:
        if not self.snapshot_path:
            return
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def _run_batch(self, batch) -> None:
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(batch.x, batch.edge_index)
        seed_mask = torch.isin(batch.n_id, batch.input_id)
        loss = self.criterion(out[seed_mask], batch.y[seed_mask])
        loss.backward()
        self.optimizer.step()
        self._log("batch_train_loss", loss.item())

    def _run_epoch(self, epoch: int) -> None:
        it = iter(self.train_loader)
        nbr_batches = len(self.train_loader)

        _BURST_BATCHES = self.measurer.cpu_burst_batches if self.measurer is not None else 0
        burst_handle = None
        if (epoch == 0 or epoch == 1) and _BURST_BATCHES > 0:
            burst_handle = start_cpu_burst(self.measurer)

        for batch_idx in range(nbr_batches):
            self._set_phase("sampling")
            self._log("start_batch_fetch", 1)
            batch = next(it)
            self._log("end_batch_fetch", 1)

            if self.measurer is not None:
                if hasattr(batch, "n_id"):
                    self.measurer.log_node_visits(batch.n_id.tolist())

                if hasattr(batch, "edge_index") and hasattr(batch, "n_id"):
                    ei = batch.edge_index.detach().cpu()
                    nid = batch.n_id.detach().cpu()
                    row = nid[ei[0]].tolist()
                    col = nid[ei[1]].tolist()
                    edge_keys = [
                        f"{a}_{b}" if a <= b else f"{b}_{a}"
                        for a, b in zip(row, col)
                    ]
                    self.measurer.log_edge_visits(edge_keys)

            self._log("batch_nbr_nodes_total", int(batch.x.shape[0]))
            self._log("batch_nbr_edges_total", int(batch.edge_index.shape[1]))
            if hasattr(batch, "input_id"):
                self._log("batch_nbr_seed_nodes", int(batch.input_id.shape[0]))

            self._set_phase("training")
            self._log("start_batch_processing", 1)
            self._run_batch(batch)
            self._log("end_batch_processing", 1)

            if burst_handle is not None and batch_idx == _BURST_BATCHES - 1:
                stop_event, thread = burst_handle
                stop_event.set()
                thread.join(timeout=0.5)
                burst_handle = None

        if burst_handle is not None:
            stop_event, thread = burst_handle
            stop_event.set()
            thread.join(timeout=0.5)

        self._set_phase("idle")

    def _start_training(self, max_epochs: int, start_time: float) -> None:
        validation_loss_minimum = None
        for epoch in range(self.epochs_run, max_epochs):
            self._log("epoch_start", 1)
            self._run_epoch(epoch)
            self._log("epoch_end", 1)

            # Validation on rank 0 only; barrier keeps all ranks in lockstep
            if self.rank == 0 and self.eval_every_epochs is not None and (epoch % self.eval_every_epochs == 0):
                self._log("start_validation_accuracy", 1)
                val_acc, val_loss = self.evaluate_fn(
                    self.model.module,
                    (self.feature_store, self.graph_store),
                    sampler=self.sampler,
                    split=self.eval_split,
                    iteration=epoch,
                )
                self._log("end_validation_accuracy", 1)
                self._log("validation_accuracy", val_acc)
                self._log("validation_loss", val_loss)

                if validation_loss_minimum is None or val_loss < validation_loss_minimum:
                    validation_loss_minimum = val_loss
                    self._log("start_saving_weights")
                    self._save_snapshot(epoch)
                    self._log("end_saving_weights")

            # All ranks sync after epoch (rank 1 waits while rank 0 validates/saves)
            dist.barrier()

            if time.monotonic() - start_time >= self.max_train_seconds:
                if self.rank == 0:
                    print("Stopping: max training time reached.")
                self._log("training_time_exceeded", epoch + 1)
                break

        self._log("program_end", 1)

    def train(self, max_epochs: int) -> None:
        self.model.train()
        start_time = time.monotonic()

        # cProfile and CPU monitor only on rank 0 (measurer is None on other ranks)
        pr = None
        monitor = None
        if self.measurer is not None:
            run_dir = self.measurer.run_results_path
            pr = cProfile.Profile()
            pr.enable()
            monitor = start_cpu_monitor(self.measurer, interval=self.measurer.coarse_cpu_interval)

        try:
            self._start_training(max_epochs, start_time)
        finally:
            if monitor is not None:
                stop_event, thread = monitor
                stop_event.set()
                thread.join(timeout=5.0)

            if pr is not None:
                pr.disable()
                txt_path = run_dir / "train_profile.txt"
                stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
                with txt_path.open("w") as f:
                    stats.stream = f
                    stats.print_stats(150)
                prof_path = run_dir / "train_profile.prof"
                pr.dump_stats(str(prof_path))

            if self.rank == 0 and self.log_train_time:
                duration = time.monotonic() - start_time
                print(f"Training duration: {duration:.2f}s")

            # Test evaluation on rank 0
            if self.rank == 0:
                try:
                    test_acc, _ = self.evaluate_fn(
                        self.model.module,
                        (self.feature_store, self.graph_store),
                        sampler=self.sampler,
                        split="test",
                    )
                    self._log("test_accuracy", test_acc)
                except Exception as e:
                    print(f"Warning: Test evaluation failed: {e}")

            # Flush and summarize results
            if self.measurer is not None:
                print(f"[rank0] Writing results to: {self.measurer.run_results_path}")
                self.measurer.close()
                try:
                    self.measurer.summarize()
                    print(f"[rank0] Summary written to: {self.measurer.run_results_path / 'measurements.json'}")
                except Exception as e:
                    print(f"[rank0] Warning: Failed to summarize measurements: {e}")
