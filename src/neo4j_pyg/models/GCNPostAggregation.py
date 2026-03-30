import copy
from typing import Tuple

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
    performed server-side by the ``gnnProcedures.aggregation.neighbor.mean`` Java UDP.

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


# ---------------------------------------------------------------------------
# GCNConv <-> pre-aggregated nn.Linear conversion
# ---------------------------------------------------------------------------

def _get_parent_and_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def gcnconv_to_linear(conv: GCNConv) -> nn.Linear:
    """Extract trained weights from a GCNConv into a plain nn.Linear.

    The returned layer skips aggregation and is correct when input has already
    been aggregated server-side (e.g. via the Neo4j neighbor.mean UDP).
    Metadata is stored on the layer so linear_to_gcnconv can restore it.

    Note: exact weight transfer is only valid when server-side aggregation
    matches GCNConv's degree-normalised sum. Plain mean is an approximation.
    """
    has_bias = conv.bias is not None
    lin = nn.Linear(conv.in_channels, conv.out_channels, bias=has_bias)
    with torch.no_grad():
        lin.weight.copy_(conv.lin.weight)
        if has_bias:
            lin.bias.copy_(conv.bias)
    lin._from_gcnconv = True
    lin._gcnconv_kwargs = {
        "improved": conv.improved,
        "cached": conv.cached,
        "add_self_loops": conv.add_self_loops,
        "normalize": conv.normalize,
    }
    return lin


def linear_to_gcnconv(lin: nn.Linear) -> GCNConv:
    """Restore a GCNConv from a layer produced by gcnconv_to_linear.

    Raises ValueError for layers not tagged by gcnconv_to_linear, preventing
    silent incorrect reconstruction from an arbitrary nn.Linear.
    """
    if not getattr(lin, "_from_gcnconv", False):
        raise ValueError(
            "This nn.Linear was not produced by gcnconv_to_linear — "
            "cannot safely reconstruct a GCNConv."
        )
    kwargs = getattr(lin, "_gcnconv_kwargs", {})
    conv = GCNConv(lin.in_features, lin.out_features, bias=lin.bias is not None, **kwargs)
    with torch.no_grad():
        conv.lin.weight.copy_(lin.weight)
        if lin.bias is not None:
            conv.bias.copy_(lin.bias)
    return conv


def to_preaggregated(model: nn.Module) -> nn.Module:
    """Return a deep copy of *model* with every GCNConv replaced by a
    weight-equivalent nn.Linear.  The result expects pre-aggregated node
    features as input and ignores any edge_index argument.

    Only GCNConv layers are converted.  Any other MessagePassing layer is
    left unchanged — this is intentional, as e.g. GATConv cannot be separated
    from its aggregation step.
    """
    model = copy.deepcopy(model)
    for name, module in list(model.named_modules()):
        if isinstance(module, GCNConv):
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, gcnconv_to_linear(module))
    return model


def from_preaggregated(model: nn.Module) -> nn.Module:
    """Reverse of to_preaggregated: restore GCNConv layers from the tagged
    nn.Linear replacements.  Only works on a model returned by to_preaggregated.
    """
    model = copy.deepcopy(model)
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and getattr(module, "_from_gcnconv", False):
            parent, attr = _get_parent_and_attr(model, name)
            setattr(parent, attr, linear_to_gcnconv(module))
    return model
