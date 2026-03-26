from typing import Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.data import Data, GraphStore, FeatureStore
from torch_geometric.loader import GraphSAINTRandomWalkSampler
from torch_geometric.sampler import BaseSampler

from training.Training import Trainer
from training.evaluate import evaluate
from benchmarking_tools import Measurer


class GraphSAINTTrainer(Trainer):
    """Trainer variant for GraphSAINT Random Walk sampling.

    Each training batch is an induced subgraph sampled via random walks starting
    from random seed nodes. The loss is weighted by per-node normalization
    coefficients (node_norm) to produce an unbiased gradient estimator.

    Supports two backends determined by the type of ``data``:

    * **In-memory** (``data: Data``) — builds a ``GraphSAINTRandomWalkSampler``
      internally; evaluation uses full-graph inference.
    * **Neo4j** (``data: (FeatureStore, GraphStore)``) — accepts an externally
      constructed ``train_loader`` (``Neo4jGraphSAINTRandomWalkSampler``) and
      ``eval_sampler`` (``Neo4jNeighborSampler``); evaluation uses the existing
      ``evaluate()`` helper with the Neo4j stores.

    Args:
        model: GNN model accepting (x, edge_index).
        data: In-memory PyG ``Data`` object **or** a ``(FeatureStore, GraphStore)``
            tuple for Neo4j-backed training.
        patience: Early-stopping patience in epochs.
        min_delta: Minimum validation-loss improvement to reset patience.
        measurer: Benchmarking measurer instance.
        lr: Adam learning rate.
        batch_size: Number of nodes in each sampled subgraph.
        walk_length: Random-walk length per root node (in-memory path only).
        num_steps: Number of subgraphs sampled per epoch (in-memory path only).
        sample_coverage: Pre-sampling rounds for node_norm (in-memory path only).
        train_loader: Pre-built ``Neo4jGraphSAINTRandomWalkSampler``
            (Neo4j path only).
        eval_sampler: Pre-built ``Neo4jNeighborSampler`` for val/test evaluation
            (Neo4j path only).
        snapshot_path: Optional path to save model checkpoints.
        optimizer: Override the default Adam optimizer.
        max_train_seconds: Wall-clock training budget in seconds.
        device: Torch device string (e.g. "cpu", "cuda").
        criterion: Override the default CrossEntropyLoss.
        cpu_monitor_interval: Seconds between CPU utilisation samples (None to disable).
        max_validation_size: Cap on validation nodes evaluated per epoch.
        max_test_size: Cap on test nodes evaluated at end of training.
    """

    def __init__(
        self,
        model: nn.Module,
        data: Union[Data, Tuple[FeatureStore, GraphStore]],
        patience: int,
        min_delta: float,
        measurer: Measurer,
        lr: float,
        batch_size: int,
        walk_length: int | None = None,
        num_steps: int | None = None,
        sample_coverage: int | None = None,
        train_loader=None,
        eval_sampler: BaseSampler | None = None,
        snapshot_path: str | None = None,
        optimizer: optim.Optimizer = None,
        max_train_seconds: int = 3600,
        device: str = "cpu",
        criterion=None,
        cpu_monitor_interval: float | None = 1,
        max_validation_size: int | None = None,
        max_test_size: int | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.measurer = measurer
        self.num_neighbors = None

        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[0], FeatureStore):
            # Neo4j path: train_loader and eval_sampler are injected externally.
            if train_loader is None:
                raise ValueError("train_loader is required when data is (FeatureStore, GraphStore)")
            self.feature_store, self.graph_store = data
            self.data = None
            self.sampler = eval_sampler
            self.train_loader = train_loader
            self.train_indices = self.graph_store.get_split(split="train")
        else:
            # In-memory path: build the PyG sampler internally.
            if walk_length is None or num_steps is None or sample_coverage is None:
                raise ValueError(
                    "walk_length, num_steps, and sample_coverage are required "
                    "for the in-memory GraphSAINT path"
                )
            self.data = data
            self.feature_store = None
            self.graph_store = None
            self.sampler = None
            self.train_indices = torch.where(data.train_mask)[0]
            self.train_loader = GraphSAINTRandomWalkSampler(
                data,
                batch_size=batch_size,
                walk_length=walk_length,
                num_steps=num_steps,
                sample_coverage=sample_coverage,
            )

        self.measurer.log_event("nbr_training_datapoints", len(self.train_indices))
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

    def _run_batch(self, batch) -> None:
        batch = batch.to(self.device)
        self.optimizer.zero_grad()
        out = self.model(batch.x, batch.edge_index)
        train_mask = batch.train_mask
        loss = F.cross_entropy(out[train_mask], batch.y[train_mask], reduction='none')
        if hasattr(batch, 'node_norm'):
            weights = batch.node_norm[train_mask]
            loss = (loss * weights).sum() / weights.sum()
        else:
            loss = loss.mean()
        self.measurer.log_event("batch_train_loss", loss.item())
        loss.backward()
        self.optimizer.step()

    def _log_seed_nodes(self, batch) -> None:
        if hasattr(batch, "train_mask"):
            self.measurer.log_event("batch_nbr_seed_nodes", int(batch.train_mask.sum().item()))

    def _evaluate_split(self, split: str, limit: int | None, iteration: int | None = None) -> tuple:
        """Evaluate the model on a dataset split.

        In-memory path: full-graph inference with no re-sampling.
        Neo4j path: mini-batch inference via the injected eval_sampler.
        """
        if self.data is not None:
            graph = self.data
            device = self.device
            mask = graph.val_mask if split == "val" else graph.test_mask
            self.model.eval()
            with torch.no_grad():
                out = self.model(graph.x.to(device), graph.edge_index.to(device))
                targets = graph.y[mask].to(device)
                preds = out[mask].argmax(dim=1)
                acc = float((targets == preds).sum().item()) / int(mask.sum().item())
                loss = float(self.criterion(out[mask], targets).item())
            self.model.train()
            if iteration is not None:
                print(f"Epoch {iteration} | {split.capitalize()} accuracy: {acc:.2f}")
            else:
                print(f"{split.capitalize()} accuracy: {acc:.2f}")
            return acc, loss

        return evaluate(
            model=self.model,
            data=(self.feature_store, self.graph_store),
            sampler=self.sampler,
            split=split,
            limit=limit,
            iteration=iteration,
        )
