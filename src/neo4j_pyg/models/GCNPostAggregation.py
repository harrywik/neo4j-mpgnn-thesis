import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch_geometric.nn import GCNConv

class GCNPostAggregation(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, nbr_classes, init_weights=True):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim1)      # layer 1: agg done in Java, only linear here
        self.gcn2 = GCNConv(hidden_dim1, hidden_dim2)   # layer 2: full GCNConv on seed→1-hop edges
        self.classifier = nn.Linear(hidden_dim2, nbr_classes)

    def forward(self, X: Tensor, edge_index: Tensor) -> Tensor:
        X = F.relu(self.lin1(X))          # applied to pre-aggregated 1-hop node features
        X = self.gcn2(X, edge_index).relu_()  # aggregates 1-hop → seed nodes
        return self.classifier(X)

class MLPPostAggregation(nn.Module):
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
