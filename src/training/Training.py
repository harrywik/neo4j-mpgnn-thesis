import time
from typing import Tuple, Union
import torch
import torch.nn.functional as F
import cProfile
import pstats
from torch_geometric.loader import NeighborLoader, NodeLoader
from torch_geometric.sampler import BaseSampler
from torch_geometric.data import Data, GraphStore, FeatureStore, HeteroData
from torch import nn
from torch import optim
from training.evaluate import evaluate
from benchmarking_tools import Measurer, start_cpu_monitor, start_cpu_burst
from training.EarlyStopping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        patience:int,
        min_delta:float,
        measurer: Measurer,
        lr:float,
        num_neighbors: list[int] | None = None,
        train_indices: torch.Tensor | None = None,
        sampler: BaseSampler = None,
        snapshot_path: str | None = None,
        batch_size: int = 100,
        drop_last: bool = True,
        optimizer: optim.Optimizer = None,
        max_train_seconds: int = 3600,
        device: str = "cpu",
        nodeloader_args: dict | None = None,
        criterion = None,
        cpu_monitor_interval: float | None = 1,
        max_training_size: int | None = None,
        max_validation_size: int | None = None,
        max_test_size: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.measurer = measurer
        # IF data is a tuple of (FeatureStore, GraphStore), then we need a sampler to create the train_loader.
        # this is the case when the graph is stored in the database
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], FeatureStore) and isinstance(data[1], GraphStore):
            if sampler is None:
                raise ValueError("Sampler cannot be None when data is a tuple of (FeatureStore, GraphStore)")
            self.feature_store, self.graph_store = data
            self.train_indices = train_indices if train_indices is not None else self._get_train_indices(max_training_size)
            self.sampler = sampler
            self.data = None
            self.nodeloader_args = nodeloader_args or put_nodeLoader_args_map(
                pickle_safe=False,
                num_workers=4,
                prefetch_factor=2,
                filter_per_worker=True,
                persistent_workers=True,
                pin_memory=False,
                shuffle=False,
                drop_last=drop_last,
            )
            if self.nodeloader_args["pickle_safe"]:
                self.train_loader = NodeLoader(
                    data=(self.feature_store, self.graph_store),
                    node_sampler=self.sampler,
                    input_nodes=self.train_indices,
                    batch_size=self.batch_size,
                    shuffle=self.nodeloader_args["shuffle"],
                    filter_per_worker=self.nodeloader_args["filter_per_worker"],
                    num_workers=self.nodeloader_args["num_workers"],
                    persistent_workers=self.nodeloader_args["persistent_workers"],
                    multiprocessing_context="spawn",
                    prefetch_factor=self.nodeloader_args["prefetch_factor"],
                    pin_memory=self.nodeloader_args["pin_memory"],
                    drop_last=drop_last
                )
            else:
                self.train_loader = NodeLoader(
                    data=(self.feature_store, self.graph_store),
                    node_sampler=self.sampler,
                    input_nodes=self.train_indices,
                    batch_size=self.batch_size,
                    shuffle=self.nodeloader_args["shuffle"],
                    pin_memory=self.nodeloader_args["pin_memory"],
                    drop_last=drop_last
                )
            self.measurer.log_event("nbr_training_datapoints", len(self.train_indices))
        # If data is not a tuple, we assume it's already a Data or HeteroData object ready for training, and we don't use a sampler.
        # This is the case when the graph is stored in RAM
        else:
            if train_indices is None or num_neighbors is None:
                raise ValueError("Train indices and num_neighbors cannot be None when data is a Data or HeteroData object")
            if nodeloader_args is None or "shuffle" not in nodeloader_args:
                raise ValueError("nodeloader_args with a 'shuffle' key is required when data is a Data or HeteroData object")
            self.train_indices = train_indices 
            self.data = data
            self.feature_store = None
            self.graph_store = None
            self.sampler = None
            self.num_neighbors = num_neighbors
            import torch_geometric.typing as T

            if T.WITH_PYG_LIB and T.WITH_TORCH_SPARSE:
                backend = "pyg-lib"
            elif T.WITH_PYG_LIB:
                backend = "pyg-lib"
            elif T.WITH_TORCH_SPARSE:
                backend = "torch-sparse"
            else:
                backend = "pure-python"
            self.measurer.log_event("sampling_backend", backend)
            _shuffle = nodeloader_args["shuffle"]
            self.train_loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                replace=False,
                disjoint=False,
                input_nodes=data.train_mask,
                subgraph_type="directional",
                shuffle=_shuffle,
                drop_last=drop_last
            )
            self.measurer.log_event("nbr_training_datapoints", int(data.train_mask.sum().item()))
        self._finish_init(
            model=model,
            optimizer=optimizer,
            lr=lr,
            snapshot_path=snapshot_path,
            max_train_seconds=max_train_seconds,
            device=device,
            patience=patience,
            min_delta=min_delta,
            criterion=criterion,
            cpu_monitor_interval=cpu_monitor_interval,
            max_validation_size=max_validation_size,
            max_test_size=max_test_size,
        )

    def _finish_init(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer | None,
        lr: float,
        snapshot_path: str | None,
        max_train_seconds: int,
        device: str,
        patience: int,
        min_delta: float,
        criterion,
        cpu_monitor_interval: float | None,
        max_validation_size: int | None,
        max_test_size: int | None,
    ) -> None:
        """Shared post-loader initialisation. Called by __init__ and subclass __init__."""
        self.model = model
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )
        self.snapshot_path = snapshot_path
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0
        self.device = torch.device(device)
        self.model.to(self.device)
        self.nbr_training_datapoints = len(self.train_indices)
        self.early_stopping = EarlyStopping(min_delta=min_delta, patience=patience)
        self.validation_loss_minimum = None
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        self.cpu_monitor_interval = cpu_monitor_interval
        self.max_validation_size = max_validation_size
        self.max_test_size = max_test_size

    def _evaluate_split(self, split: str, limit: int | None, iteration: int | None = None) -> tuple:
        """Evaluate on a dataset split. Override in subclasses for custom eval logic."""
        if self.data is not None:
            return evaluate(
                model=self.model,
                data=self.data,
                num_neighbors=self.num_neighbors,
                split=split,
                iteration=iteration,
                limit=limit,
            )
        return evaluate(
            model=self.model,
            data=(self.feature_store, self.graph_store),
            sampler=self.sampler,
            split=split,
            iteration=iteration,
            limit=limit,
        )

    def _log_seed_nodes(self, batch) -> None:
        """Log the number of seed (training) nodes in a batch. Override in subclasses."""
        if hasattr(batch, "input_id"):
            self.measurer.log_event("batch_nbr_seed_nodes", int(batch.input_id.shape[0]))

    def _save_snapshot(self, epoch: int) -> None:
        if not self.snapshot_path:
            return
        model = self.model.module if hasattr(self.model, "module") else self.model
        snapshot = {
            "MODEL_STATE": model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _get_train_indices(self, limit: int | None = None) -> torch.Tensor:
        indices = self.graph_store.get_split(
            limit=limit,
            split="train",
            # shuffle=True,
        ).to(torch.long)
        return indices

    def _run_batch(self, batch) -> None:
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(batch.x, batch.edge_index)
        
        # Filter for seed nodes (Graph GCN specific)
        seed_mask = torch.isin(batch.n_id, batch.input_id)
        loss = self.criterion(out[seed_mask], batch.y[seed_mask])
        self.measurer.log_event("batch_train_loss", loss.item())
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int) -> None:
        it = iter(self.train_loader)
        nbr_batches = len(self.train_loader)

        # One continuous burst covering the first N consecutive batches of epoch 0.
        _BURST_BATCHES = self.measurer.cpu_burst_batches
        burst_handle = None
        if epoch == 0 or epoch == 1:
            burst_handle = start_cpu_burst(self.measurer)

        for batch_idx in range(nbr_batches):
            self.measurer.set_phase("sampling")
            self.measurer.log_event("start_batch_fetch", 1)
            batch = next(it)
            self.measurer.log_event("end_batch_fetch", 1)

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

            self.measurer.log_event("batch_nbr_nodes_total", int(batch.x.shape[0]))
            self.measurer.log_event("batch_nbr_edges_total", int(batch.edge_index.shape[1]))
            self._log_seed_nodes(batch)

            self.measurer.set_phase("training")
            self.measurer.log_event("start_batch_processing", 1)
            self._run_batch(batch)
            self.measurer.log_event("end_batch_processing", 1)

            # Stop burst after the 3rd consecutive batch completes.
            if burst_handle is not None and batch_idx == _BURST_BATCHES - 1:
                stop_event, thread = burst_handle
                stop_event.set()
                thread.join(timeout=0.5)
                burst_handle = None

        # Ensure burst is stopped if the epoch had fewer than _BURST_BATCHES batches.
        if burst_handle is not None:
            stop_event, thread = burst_handle
            stop_event.set()
            thread.join(timeout=0.5)

        self.measurer.set_phase("idle")                    
                

    def train(self, max_epochs: int) -> None:
        start_time = time.monotonic()
        self.model.train()
        run_dir = self.measurer.run_results_path
        txt_path = run_dir / "train_profile.txt"
        pr = cProfile.Profile()
        pr.enable()
        monitor = None
        try:
            monitor = start_cpu_monitor(self.measurer, interval=self.measurer.coarse_cpu_interval)
            self._start_training(max_epochs, start_time)
        finally:
            if monitor is not None:
                stop_event, thread = monitor
                stop_event.set()
                thread.join(timeout=max(1.0, float(self.cpu_monitor_interval or 1)))
            pr.disable()
            stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
            with txt_path.open("w") as f:
                stats.stream = f
                stats.print_stats(150)
            prof_path = run_dir / "train_profile.prof"
            pr.dump_stats(str(prof_path))
        duration = time.monotonic() - start_time
        print(f"Training duration: {duration:.2f}s")
        test_accuracy, _ = self._evaluate_split("test", self.max_test_size)
        self.measurer.log_event("test_accuracy", test_accuracy)
        self.measurer.close()
        try:
            self.measurer.summarize()
        except Exception as e:
            print(f"Warning: Failed to summarize measurements: {e}")

    def _start_training(self, max_epochs: int, start_time: float) -> None:
        validation_loss_minimum = None
        for epoch in range(self.epochs_run, max_epochs):
            self.measurer.log_event("epoch_start", 1)
            self._run_epoch(epoch)
            self.measurer.log_event("epoch_end", 1)
            self.measurer.log_event("start_validation_accuracy", 1)
            validation_acc, validation_loss = self._evaluate_split("val", self.max_validation_size, iteration=epoch)
            self.measurer.log_event("validation_accuracy", validation_acc)
            self.measurer.log_event("validation_loss", validation_loss)
            self.measurer.log_event("end_validation_accuracy", 1)
            if validation_loss_minimum is None or validation_loss < validation_loss_minimum:
                validation_loss_minimum = validation_loss
                self.measurer.log_event("start_saving_weights")
                self._save_snapshot(epoch)
                self.measurer.log_event("end_saving_weights")
            if self.early_stopping(validation_loss):
                self.measurer.log_event("training_converged", (epoch + 1))
                break
            if time.monotonic() - start_time >= self.max_train_seconds:
                print("Stopping: max training time reached.")
                self.measurer.log_event("training_time_exceeded", (epoch + 1))
                break
        self.measurer.log_event("program_end", 1)



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
    and all arguments setting multiple workers configuration can be ignored."""
    return {
        "pickle_safe": bool(pickle_safe) if pickle_safe is not None else False,
        "num_workers": int(num_workers) if num_workers is not None else 0,
        "prefetch_factor": int(prefetch_factor) if prefetch_factor is not None else 2,
        "filter_per_worker": bool(filter_per_worker) if filter_per_worker is not None else False,
        "persistent_workers": bool(persistent_workers) if persistent_workers is not None else False,
        "pin_memory": bool(pin_memory) if pin_memory is not None else False,
        "shuffle": bool(shuffle) if shuffle is not None else True,
    }
