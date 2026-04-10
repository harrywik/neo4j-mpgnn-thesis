"""Pre-aggregation adapter interface and registry.

Defines :class:`PreAggAdapter` and concrete adapters for supported
MessagePassing layers.  The registry ``_ADAPTER_REGISTRY`` maps each
supported layer class to its adapter class.

To add support for a new layer type:
  1. Implement a :class:`PreAggAdapter` subclass.
  2. Add an entry to ``_ADAPTER_REGISTRY`` after the class definition.

Attention-based layers (GATConv, TransformerConv) are explicitly blocked
because their message function cannot be separated from aggregation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_geometric.nn.conv import MessagePassing

# Layers whose aggregation cannot be separated from their message function.
_UNSUPPORTED_MP_LAYERS = ()
try:
    from torch_geometric.nn import GATConv
    _UNSUPPORTED_MP_LAYERS += (GATConv,)
except ImportError:
    pass
try:
    from torch_geometric.nn import TransformerConv
    _UNSUPPORTED_MP_LAYERS += (TransformerConv,)
except ImportError:
    pass

# Registry mapping MessagePassing layer class → PreAggAdapter class.
# To add support for a new layer type, add an entry here and implement
# the corresponding PreAggAdapter subclass.
_ADAPTER_REGISTRY: Dict[type, type] = {}


# ---------------------------------------------------------------------------
# Abstract adapter
# ---------------------------------------------------------------------------

class PreAggAdapter(nn.Module, ABC):
    """Abstract base for first-layer pre-aggregation adapters.

    Subclasses wrap a specific MessagePassing layer and implement two
    execution modes depending on whether ``preagg`` is provided.
    """

    @abstractmethod
    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None,
                preagg: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Run the layer.

        Parameters
        ----------
        x:
            Node feature matrix.
        edge_index:
            Graph connectivity.  Required when ``preagg`` is ``None``.
        preagg:
            Pre-aggregated neighbourhood tensor produced by Neo4j.  When
            provided, ``edge_index`` is not used for this layer.
        """

    @abstractmethod
    def apply_preagg(self, preagg: Tensor) -> Tensor:
        """Apply the layer's learnable transform to a pre-aggregated tensor.

        Unlike ``forward``, this method does not trigger hooks and is safe to
        call from within a forward hook on the same adapter.
        """

    @abstractmethod
    def preagg_spec(self) -> Dict[str, Any]:
        """Describe what the server-side aggregation must compute."""


# ---------------------------------------------------------------------------
# GCNConv adapter
# ---------------------------------------------------------------------------

class PreAggGCNConvAdapter(PreAggAdapter):
    """Adapter for :class:`~torch_geometric.nn.GCNConv`.

    GCNConv applies a linear transform ``Θ`` followed by degree-normalised
    aggregation.  Because these two operations commute:

        Θ · (Ã X)  ==  Ã · (X Θ)

    the weight matrix is identical regardless of which is done first.  If
    Neo4j already computed the degree-normalised aggregation ``Ã X``, this
    adapter bypasses ``propagate`` and feeds the pre-aggregated tensor directly
    into the linear transform.

    Important: for exact equivalence, the server-side aggregation must match
    GCNConv semantics exactly (degree-normalised sum with self-loops).  A plain
    ``neighbor.mean`` UDP is an approximation and will cause a small but
    non-zero output mismatch.
    """

    def __init__(self, conv: GCNConv) -> None:
        super().__init__()
        self.conv = conv

    def apply_preagg(self, preagg: Tensor) -> Tensor:
        out = self.conv.lin(preagg)
        if self.conv.bias is not None:
            out = out + self.conv.bias
        return out

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None,
                preagg: Optional[Tensor] = None, **kwargs) -> Tensor:
        if preagg is not None:
            return self.apply_preagg(preagg)
        if edge_index is None:
            raise ValueError(
                "PreAggGCNConvAdapter requires either 'preagg' or 'edge_index'."
            )
        return self.conv(x, edge_index, **kwargs)

    def preagg_spec(self) -> Dict[str, Any]:
        return {
            "op": "gcn_norm",
            "include_self": True,
            "requires_degree": True,
            "supports_edge_weight": self.conv.normalize,
            "note": (
                "preagg must equal the degree-normalised aggregated input "
                "D^{-1/2} A_hat D^{-1/2} X, not a plain mean of neighbours."
            ),
        }


_ADAPTER_REGISTRY[GCNConv] = PreAggGCNConvAdapter


# ---------------------------------------------------------------------------
# Recursive first-layer replacement
# ---------------------------------------------------------------------------

def _replace_first_mp_layer(module: nn.Module) -> Optional[PreAggAdapter]:
    """Walk *module*'s children and replace the first MessagePassing layer.

    Raises ``ValueError`` for any MessagePassing layer that is not in
    ``_ADAPTER_REGISTRY`` (including attention-based layers).  Returns the
    installed adapter on success, or ``None`` if no MP layer was found.
    """
    for name, child in module.named_children():
        if isinstance(child, MessagePassing):
            if _UNSUPPORTED_MP_LAYERS and isinstance(child, _UNSUPPORTED_MP_LAYERS):
                raise ValueError(
                    f"Layer '{name}' ({type(child).__name__}) is attention-based "
                    f"and cannot separate aggregation from its message function."
                )
            adapter_cls = _ADAPTER_REGISTRY.get(type(child))
            if adapter_cls is None:
                supported = ", ".join(c.__name__ for c in _ADAPTER_REGISTRY)
                raise ValueError(
                    f"Layer '{name}' ({type(child).__name__}) is a MessagePassing "
                    f"layer with no pre-aggregation adapter. "
                    f"Supported: {supported}."
                )
            adapter = adapter_cls(child)
            setattr(module, name, adapter)
            return adapter

        adapter = _replace_first_mp_layer(child)
        if adapter is not None:
            return adapter

    return None
