from pathlib import Path
import time
import psutil
import torch
import cProfile
import pstats
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import BaseSampler
from torch_geometric.data import GraphStore, FeatureStore
from torch import nn
from torch import optim
from evaluate import evaluate
from benchmarking_tools import Measurer, start_cpu_monitor
from EarlyStopping import EarlyStopping


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        feature_store: FeatureStore,
        graph_store: GraphStore,
        sampler: BaseSampler,
        snapshot_path: str | None = None,
        batch_size: int = 500,
        lr:float = 1e-2,
        optimizer: optim.Optimizer = None,
        max_train_seconds: int = 3600,
        device: str = "cpu",
        nodeloader_args: dict | None = None,
        measurer: Measurer | None = None,
        criterion = None
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.sampler = sampler
        self.optimizer = optimizer if optimizer is not None else optim.Adam(
            model.parameters(), lr=lr, weight_decay=5e-4
        )
        self.measurer = measurer
        self.snapshot_path = snapshot_path
        self.batch_size = batch_size
        self.max_train_seconds = max_train_seconds
        self.epochs_run = 0        
        self.device = torch.device(device)
        self.nodeloader_args = nodeloader_args or put_nodeLoader_args_map(
            pickle_safe=False,
            num_workers=0,
            prefetch_factor=2,
            filter_per_worker=False,
            persistent_workers=False,
            pin_memory=False,
            shuffle=True,
        )
        self.model.to(self.device)
        self.train_indices = self._get_train_indices()
        self.nbr_training_datapoints = len(self.train_indices)
        self.measurer.log_event("nbr_training_datapoints", len(self.train_indices))
        self.early_stopping = EarlyStopping(min_delta=1e-3, patience=2)
        self.validation_loss_minimum = None
        self.criterion = nn.CrossEntropyLoss() if criterion is None else criterion
        

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
        # NodeLoader handles the graph sampling logic\
        if self.nodeloader_args['pickle_safe']:
            train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=self.train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args['shuffle'],
                filter_per_worker=self.nodeloader_args['filter_per_worker'],
                num_workers=self.nodeloader_args['num_workers'],
                persistent_workers=self.nodeloader_args['persistent_workers'],
                prefetch_factor=self.nodeloader_args['prefetch_factor'],
                pin_memory=self.nodeloader_args['pin_memory'],
            )
        else:
            # there are some arguments we cant use if the implementation is not pickle safe, so we force them to be single-process
            train_loader = NodeLoader(
                data=(self.feature_store, self.graph_store),
                node_sampler=self.sampler,
                input_nodes=self.train_indices,
                batch_size=self.batch_size,
                shuffle=self.nodeloader_args['shuffle'],
                pin_memory=self.nodeloader_args['pin_memory']

                #ARGUMENTS BELOW IGNORED IF NOT PICKLE SAFE
                # filter_per_worker=self.nodeloader_args['filter_per_worker'],
                # num_workers=0,  # Force single-process loading
                # persistent_workers=False,
                # prefetch_factor=0,
            )
        it = iter(train_loader)
        nbr_batches = len(train_loader)

        for _ in range(nbr_batches):
            self.measurer.log_event("start_batch_fetch", 1)
            batch = next(it)  # sampling + filter_fn + feature_store happens here
            self.measurer.log_event("end_batch_fetch", 1)

            self.measurer.log_event("start_batch_processing", 1)
            self._run_batch(batch)  # forward/backward/optimizer
            self.measurer.log_event("end_batch_processing", 1)                    
                

    def train(self, max_epochs: int) -> None:
        start_time = time.monotonic()
        self.model.train()
        validation_loss_minimum = None
        run_dir = self.measurer.run_results_path
        txt_path = run_dir / "train_profile.txt"
        pr = cProfile.Profile()
        pr.enable()
        try:
            start_cpu_monitor(self.measurer)
            self._start_training(max_epochs, start_time)
        finally:
            pr.disable()
            stats = pstats.Stats(pr).strip_dirs().sort_stats("cumtime")
            with txt_path.open("w") as f:
                stats.stream = f
                stats.print_stats(150)
        duration = time.monotonic() - start_time
        print(f"Training duration: {duration:.2f}s")
        test_accuracy, _ = evaluate(self.model, self.graph_store, self.feature_store, self.sampler, split="test")
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
            validation_acc, validation_loss = evaluate(
                self.model,
                self.graph_store,
                self.feature_store,
                self.sampler,
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