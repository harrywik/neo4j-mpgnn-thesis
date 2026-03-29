from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor


class SIGNPostAggregation(nn.Module):
    """SIGN model for the UDP hybrid with k-hop neighbourhood aggregation.

    Input ``x`` is the concatenation of per-hop mean-aggregated feature vectors
    produced by ``gnnProcedures.aggregation.sign.multiHop``:

        x = [x_0 || x_1 || ... || x_k]   shape: (N, feature_dim * (hops + 1))

    where ``x_0`` is the seed's own features, ``x_h`` is the mean of features
    at hop-h distance.  A linear projection head maps the concatenation to class
    logits; weight matrices stay in PyTorch so autograd is fully intact.

    Equivalent to the SIGN architecture (Rossi et al., 2020) with the graph
    diffusion computed server-side in Neo4j.

    Parameters
    ----------
    feature_dim:
        Dimension of each per-hop feature vector.
    hops:
        Number of hops (k).  Input dimension = feature_dim * (hops + 1).
    hidden_dims:
        List of hidden layer widths, e.g. ``[256, 128]``.  An empty list
        produces a single linear layer (no hidden layers).
    nbr_classes:
        Number of output classes.
    init_weights:
        Whether to initialise weights with Xavier uniform.
    """

    def __init__(
        self,
        feature_dim: int,
        hops: int,
        hidden_dims: List[int],
        nbr_classes: int,
        init_weights: bool = True,
    ):
        super().__init__()
        in_dim = feature_dim * (hops + 1)
        dims = [in_dim] + hidden_dims + [nbr_classes]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        if init_weights:
            self.reset_parameters()

    def forward(self, X: Tensor, edge_index: Tensor = None) -> Tensor:
        for layer in self.layers[:-1]:
            X = F.relu(layer(X))
        return self.layers[-1](X)

    def reset_parameters(self):
        cpu_state = torch.get_rng_state()
        gpu_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        for layer in self.layers:
            init.xavier_uniform_(layer.weight)
            init.zeros_(layer.bias)

        torch.set_rng_state(cpu_state)
        if gpu_state is not None:
            torch.cuda.set_rng_state_all(gpu_state)
