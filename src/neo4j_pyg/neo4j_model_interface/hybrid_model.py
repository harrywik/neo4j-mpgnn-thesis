"""HybridAggModel: K-1 frontier pre-aggregation wrapper.

Wraps any model whose first MessagePassing layer is registered in
``_ADAPTER_REGISTRY``.  During the forward pass, frontier nodes (deepest-hop
nodes in the sampled subgraph) have their raw features zeroed and their
first-layer output replaced with Neo4j's pre-aggregated result, so all
later layers run on correct hidden representations.
"""

import copy
from typing import Optional

import torch.nn as nn
from torch import Tensor

from neo4j_pyg.neo4j_model_interface.adapters import (
    _ADAPTER_REGISTRY,
    _replace_first_mp_layer,
)


class HybridAggModel(nn.Module):
    """Wrapper that replaces only the deepest-hop raw features with
    pre-aggregated inputs for the K-1 frontier, for any model whose first
    MessagePassing layer has a registered adapter in ``_ADAPTER_REGISTRY``.

    The sampled topology remains unchanged.  The first MP layer is found
    automatically via ``_replace_first_mp_layer``; no specific model
    attribute names are assumed.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        inner = copy.deepcopy(inner)
        adapter = _replace_first_mp_layer(inner)
        if adapter is None:
            supported = ", ".join(c.__name__ for c in _ADAPTER_REGISTRY)
            raise ValueError(
                f"HybridAggModel requires a model with a supported first "
                f"MessagePassing layer. Supported: {supported}."
            )
        self.inner = inner
        self._adapter = adapter
        self._uses_hybrid_last_hop_preaggregation = True

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        frontier_mask: Optional[Tensor] = None,
        aggregated_neighbors: Optional[Tensor] = None,
        target_mask: Optional[Tensor] = None,
    ) -> Tensor:
        use_hybrid = (
            frontier_mask is not None
            and aggregated_neighbors is not None
            and frontier_mask.numel() > 0
            and frontier_mask.any()
        )
        if not use_hybrid:
            return self.inner(x, edge_index)

        if target_mask is None:
            target_mask = aggregated_neighbors.abs().sum(dim=1) > 0

        # Zero frontier features so the first layer doesn't double-count them.
        x = x.clone()
        x[frontier_mask] = 0

        # Install a one-shot post-hook on the adapter that replaces target
        # nodes' first-layer output with the pre-aggregated result.  Later
        # layers then run on the corrected hidden representations.
        _target_mask = target_mask
        _agg_neighbors = aggregated_neighbors

        def _fix_targets(module, _input, output):
            if _target_mask.any():
                output = output.clone()
                output[_target_mask] = module.apply_preagg(
                    _agg_neighbors[_target_mask]
                )
            return output

        handle = self._adapter.register_forward_hook(_fix_targets)
        try:
            out = self.inner(x, edge_index)
        finally:
            handle.remove()
        return out
