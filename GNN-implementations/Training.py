import time
from typing import Tuple, Union
import torch
import cProfile
import pstats
from torch_geometric.loader import NeighborLoader, NodeLoader
from torch_geometric.sampler import BaseSampler
from torch_geometric.data import Data, GraphStore, FeatureStore, HeteroData
from torch import nn
from torch import optim
from evaluate import evaluate
from benchmarking_tools import Measurer, start_cpu_monitor
from EarlyStopping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        data: Union[Data, HeteroData, Tuple[FeatureStore, GraphStore]],
        patience:int,
        min_delta:float,
        measurer: Measurer,
        num_neighbors: list[int] | None = None,
        train_indices: torch.Tensor | None = None,
        sampler: BaseSampler = None,
        snapshot_path: str | None = None,
        batch_size: int = 100,
        lr:float = 1e-2,
        optimizer: optim.Optimizer = None,
        max_train_seconds: int = 3600,
        device: str = "cpu",
        nodeloader_args: dict | None = None,
        criterion = None,
        cpu_monitor_interval: float | None = 1,
    ) -> None:
        self.batch_size = batch_size
        self.measurer = measurer
        # IF data is a tuple of (FeatureStore, GraphStore), then we need a sampler to create the train_loader.
        # this is the case when the graph is stored in the database
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], FeatureStore) and isinstance(data[1], GraphStore):
            if sampler is None:
                raise ValueError("Sampler cannot be None when data is a tuple of (FeatureStore, GraphStore)")
            self.feature_store, self.graph_store = data
            self.train_indices = train_indices if train_indices is not None else self._get_train_indices()
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
                )
            else:
                self.train_loader = NodeLoader(
                    data=(self.feature_store, self.graph_store),
                    node_sampler=self.sampler,
                    input_nodes=self.train_indices,
                    batch_size=self.batch_size,
                    shuffle=self.nodeloader_args["shuffle"],
                    pin_memory=self.nodeloader_args["pin_memory"],
                )
            self.measurer.log_event("nbr_training_datapoints", len(self.train_indices))
        # If data is not a tuple, we assume it's already a Data or HeteroData object ready for training, and we don't use a sampler.
        # This is the case when the graph is stored in RAM
        else:
            if train_indices is None or num_neighbors is None:
                raise ValueError("Train indices and num_neighbors cannot be None when data is a Data or HeteroData object")
            self.train_indices = train_indices 
            self.data = data
            self.feature_store = None
            self.graph_store = None
            self.sampler = None
            self.num_neighbors = num_neighbors
            self.train_loader = NeighborLoader(
                data,
                # Sample 30 neighbors for each node for 2 iterations
                num_neighbors=num_neighbors,
                # Use a batch size of 128 for sampling training nodes
                batch_size=batch_size,
                input_nodes=data.train_mask,
            )
            self.measurer.log_event("nbr_training_datapoints", int(data.train_mask.sum().item()))
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

    def _get_train_indices(self) -> torch.Tensor:
        indices = self.graph_store.get_split(
            split="train",
            shuffle=True,
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

        for _ in range(nbr_batches):
            self.measurer.log_event("start_batch_fetch", 1)
            batch = next(it) 
            self.measurer.log_event("end_batch_fetch", 1)

            self.measurer.log_event("start_batch_processing", 1)
            print(batch)
            self._run_batch(batch)
            self.measurer.log_event("end_batch_processing", 1)                    
                

    def train(self, max_epochs: int) -> None:
        start_time = time.monotonic()
        self.model.train()
        run_dir = self.measurer.run_results_path
        txt_path = run_dir / "train_profile.txt"
        pr = cProfile.Profile()
        pr.enable()
        try:
            start_cpu_monitor(self.measurer, interval=self.cpu_monitor_interval)
            self._start_training(max_epochs, start_time)
        finally:
            pr.disable()
            stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
            with txt_path.open("w") as f:
                stats.stream = f
                stats.print_stats(150)
        duration = time.monotonic() - start_time
        print(f"Training duration: {duration:.2f}s")
        test_accuracy = None
        if self.data:
            test_accuracy, _ = evaluate(
                model=self.model,
                data=self.data,
                num_neighbors=self.num_neighbors,
                split="test",
            )
        else:
            test_accuracy, _ = evaluate(
                model=self.model,
                data=(self.feature_store, self.graph_store),
                sampler=self.sampler,
                split="test",
            )
        self.measurer.log_event("test_accuracy", test_accuracy)
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
            # if we have a data object and not FS and GS
            if self.data:
                validation_acc, validation_loss = evaluate(
                    model=self.model,
                    data=self.data,
                    num_neighbors=self.num_neighbors,
                    split="val",
                )
            else:
                validation_acc, validation_loss = evaluate(
                    model=self.model,
                    data=(self.feature_store, self.graph_store),
                    sampler=self.sampler,
                    split="val",
                )
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