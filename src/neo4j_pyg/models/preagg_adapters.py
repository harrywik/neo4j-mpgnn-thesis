"""Pre-aggregation adapter framework.

Wraps a PyG model so its *first* supported MessagePassing layer can receive
a tensor that Neo4j already aggregated (e.g. via the neighbor.mean or gcn_norm
UDP), while all later layers run unchanged.

Usage
-----
    model = GCN(...)
    result = to_preaggregated_first_layer(model)

    # result.model  — wrapped model, same weights, dual-mode forward
    # result.preagg_spec — dict describing what Neo4j must compute

    out = result.model(x, edge_index)                   # normal path
    out = result.model(x, edge_index, preagg=preagg)    # first layer pre-agg

Notes
-----
Only GCNConv is supported in the current implementation.  Other supported
candidates (SAGEConv, GINConv) raise NotImplementedError for now.  Attention-
based layers (GATConv, TransformerConv) raise ValueError because their message
function cannot be separated from aggregation.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any

import torch
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


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class PreAggResult:
    """Returned by :func:`to_preaggregated_first_layer`.

    Attributes
    ----------
    model:
        A :class:`PreAggModelWrapper` around the original model.  Supports
        both normal and pre-aggregated execution with the same weights.
    preagg_spec:
        Dict describing what the server-side (Neo4j) pre-aggregation must
        compute so that the wrapped model's first layer produces the same
        output as the original.
    """
    model: nn.Module
    preagg_spec: Dict[str, Any]


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
    def preagg_spec(self) -> Dict[str, Any]:
        """Describe what the server-side aggregation must compute.

        The returned dict is forwarded to the caller via
        :class:`PreAggResult` so the data pipeline can select the correct
        Neo4j procedure and parameters.
        """


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

    def forward(self, x: Tensor, edge_index: Optional[Tensor] = None,
                preagg: Optional[Tensor] = None, **kwargs) -> Tensor:
        if preagg is not None:
            out = self.conv.lin(preagg)
            if self.conv.bias is not None:
                out = out + self.conv.bias
            return out
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


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class PreAggModelWrapper(nn.Module):
    """Thin shell around a model whose first MP layer has been replaced by a
    :class:`PreAggAdapter`.

    The wrapper accepts an optional ``preagg`` keyword argument and threads it
    to the adapted first layer via a forward pre-hook, leaving the inner
    model's ``forward`` signature completely unchanged.
    """

    def __init__(self, inner: nn.Module, adapter: PreAggAdapter) -> None:
        super().__init__()
        self.inner = inner
        self._adapter = adapter  # kept as reference so the hook can inject

    def forward(self, x: Tensor, edge_index: Tensor,
                preagg: Optional[Tensor] = None, **kwargs) -> Tensor:
        if preagg is not None:
            # Install a one-shot pre-hook that intercepts the adapter's next
            # forward call and injects preagg into its arguments.
            def _inject(module, args, kw):
                kw["preagg"] = preagg
                return args, kw

            handle = self._adapter.register_forward_pre_hook(_inject, with_kwargs=True)
            try:
                out = self.inner(x, edge_index, **kwargs)
            finally:
                handle.remove()
        else:
            out = self.inner(x, edge_index, **kwargs)
        return out


class HybridLastHopGCNWrapper(nn.Module):
    """Wrapper for a 2-layer GCN that replaces only the deepest-hop raw
    features with pre-aggregated inputs for the K-1 frontier.

    The sampled topology remains unchanged.  The wrapper assumes a standard
    2-layer ``GCN``-style model with attributes ``GCN1``, ``GCN2``, and
    ``classifier``.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        for attr in ("GCN1", "GCN2", "classifier"):
            if not hasattr(inner, attr):
                raise ValueError(
                    "HybridLastHopGCNWrapper expects a model with attributes "
                    "GCN1, GCN2, and classifier."
                )
        self.inner = inner
        self._uses_hybrid_last_hop_preaggregation = True

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        hop_depths: Optional[Tensor] = None,
        last_hop_preagg: Optional[Tensor] = None,
    ) -> Tensor:
        if hop_depths is None or last_hop_preagg is None or hop_depths.numel() == 0:
            hidden = self.inner.GCN1(x, edge_index).relu_()
            hidden = self.inner.GCN2(hidden, edge_index).relu_()
            return self.inner.classifier(hidden)

        max_depth = int(hop_depths.max().item())
        if max_depth <= 0:
            hidden = self.inner.GCN1(x, edge_index).relu_()
            hidden = self.inner.GCN2(hidden, edge_index).relu_()
            return self.inner.classifier(hidden)

        deepest_mask = hop_depths == max_depth
        frontier_mask = hop_depths == (max_depth - 1)

        raw_x = x.clone()
        raw_x[deepest_mask] = 0

        hidden = self.inner.GCN1(raw_x, edge_index)
        if frontier_mask.any():
            replaced = self.inner.GCN1.lin(last_hop_preagg[frontier_mask])
            if self.inner.GCN1.bias is not None:
                replaced = replaced + self.inner.GCN1.bias
            hidden = hidden.clone()
            hidden[frontier_mask] = replaced

        hidden = hidden.relu_()
        hidden = self.inner.GCN2(hidden, edge_index).relu_()
        return self.inner.classifier(hidden)


# ---------------------------------------------------------------------------
# Recursive first-layer replacement
# ---------------------------------------------------------------------------

def _replace_first_mp_layer(module: nn.Module) -> Optional[PreAggAdapter]:
    """Walk *module*'s direct children and replace the first supported MP layer.

    Returns the installed adapter on success, or ``None`` if no supported
    layer was found at this level.  Raises ``ValueError`` for unsupported MP
    layers encountered before any supported one.
    """
    for name, child in module.named_children():
        if isinstance(child, GCNConv):
            adapter = PreAggGCNConvAdapter(child)
            setattr(module, name, adapter)
            return adapter

        if _UNSUPPORTED_MP_LAYERS and isinstance(child, _UNSUPPORTED_MP_LAYERS):
            raise ValueError(
                f"Layer '{name}' ({type(child).__name__}) is a MessagePassing "
                f"layer whose aggregation cannot be separated from its message "
                f"function (e.g. attention-based).  "
                f"to_preaggregated_first_layer does not support this layer type."
            )

        # Not a direct MP layer — recurse one level deeper.
        if isinstance(child, nn.Module):
            adapter = _replace_first_mp_layer(child)
            if adapter is not None:
                return adapter

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def to_preaggregated_first_layer(model: nn.Module) -> PreAggResult:
    """Wrap *model* so its first GCNConv layer can run on pre-aggregated input.

    Returns a :class:`PreAggResult` with:

    - ``result.model`` — a :class:`PreAggModelWrapper` that supports both
      ``model(x, edge_index)`` and ``model(x, edge_index, preagg=preagg)``.
    - ``result.preagg_spec`` — dict describing what Neo4j must compute so that
      the first layer produces the same output as in normal mode.

    Parameters
    ----------
    model:
        Any ``nn.Module`` that contains at least one ``GCNConv`` layer.
        The original model is not modified; a deep copy is used.

    Raises
    ------
    ValueError
        If no supported first MessagePassing layer is found, or if an
        unsupported attention-based layer is encountered first.
    """
    inner = copy.deepcopy(model)
    adapter = _replace_first_mp_layer(inner)

    if adapter is None:
        raise ValueError(
            "to_preaggregated_first_layer: no supported MessagePassing layer "
            "found in the model.  Currently supported: GCNConv."
        )

    wrapped = PreAggModelWrapper(inner, adapter)
    return PreAggResult(model=wrapped, preagg_spec=adapter.preagg_spec())


def to_hybrid_last_hop_gcn(model: nn.Module) -> nn.Module:
    """Wrap a 2-layer GCN so deepest-hop raw features can be replaced by
    pre-aggregated inputs for the K-1 frontier while keeping full topology.
    """
    return HybridLastHopGCNWrapper(copy.deepcopy(model))
