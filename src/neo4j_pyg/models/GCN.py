from random import seed

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim1:int, hidden_dim2:int, nbr_classes:int, init_weights:bool=True):
        super().__init__()
        self.GCN1 = GCNConv(in_dim, hidden_dim1)
        self.GCN2 = GCNConv(hidden_dim1, hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, nbr_classes)
        if init_weights:
            self.reset_parameters()
        
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN1(X, edge_index).relu_()
        X = self.GCN2(X, edge_index).relu_()
        logits = self.classifier(X)
        return logits
    
    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.GCN1.reset_parameters()
        self.GCN2.reset_parameters()
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)


class BigGCN(nn.Module):
    """A deliberately oversized GCN to maximise training-phase CPU time.

    8 GCNConv layers with 2048-wide hidden representations, followed by a
    wide MLP classifier. Purely for benchmarking — not intended to converge.
    """

    def __init__(self, in_dim: int, nbr_classes: int, hidden_dim: int = 2048, init_weights: bool = True):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, nbr_classes),
        )
        if init_weights:
            self.reset_parameters()

    def forward(self, X: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            X = conv(X, edge_index).relu_()
        return self.classifier(X)

    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        for conv in self.convs:
            conv.reset_parameters()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)


class TinyGCN(nn.Module):
    def __init__(self, in_dim:int, hidden_dim:int, nbr_classes:int, init_weights:bool=True):
        super().__init__()
        self.GCN = GCNConv(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, nbr_classes)
        if init_weights:
            self.reset_parameters()
        
    def forward(self, X:Tensor, edge_index:Tensor) -> Tensor:
        X = self.GCN(X, edge_index).relu_()
        logits = self.classifier(X)
        return logits
    
    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        self.GCN.reset_parameters()
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)


class GCNPostAggregation(nn.Module):
    """GCN model for the Option-D hybrid: neighbour aggregation was already
    performed server-side by the ``custom.gcn.aggregateNeighbors`` Java UDP.

    Input ``x`` is the *pre-aggregated* feature matrix (one row per seed node,
    already the mean of that node's 1-hop neighbours).  No ``GCNConv`` layers
    are used — only two linear transformations followed by a classifier head,
    which is sufficient to keep the full autograd graph intact for training.

    ``edge_index`` is accepted but ignored so the model can be plugged into
    the standard training loop without modification.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        nbr_classes: int,
        init_weights: bool = True,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim1)
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.classifier = nn.Linear(hidden_dim2, nbr_classes)
        if init_weights:
            self.reset_parameters()

    def forward(self, X: Tensor, edge_index: Tensor = None) -> Tensor:
        X = F.relu(self.lin1(X))
        X = F.relu(self.lin2(X))
        return self.classifier(X)

    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        init.xavier_uniform_(self.lin1.weight)
        init.zeros_(self.lin1.bias)
        init.xavier_uniform_(self.lin2.weight)
        init.zeros_(self.lin2.bias)
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)