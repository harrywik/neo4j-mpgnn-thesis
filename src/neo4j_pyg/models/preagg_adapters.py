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
Currently supported layers: GCNConv.  Any other MessagePassing layer raises
``ValueError`` with a message listing what is supported.  To add a new layer,
implement a :class:`PreAggAdapter` subclass and register it in
``_ADAPTER_REGISTRY``.  Attention-based layers (GATConv, TransformerConv) are
explicitly blocked because their message function cannot be separated from
aggregation.
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

# Registry mapping MessagePassing layer class → PreAggAdapter class.
# To add support for a new layer type, add an entry here and implement
# the corresponding PreAggAdapter subclass.
_ADAPTER_REGISTRY: Dict[type, type] = {}  # populated after class definitions


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
    def apply_preagg(self, preagg: Tensor) -> Tensor:
        """Apply the layer's learnable transform to a pre-aggregated tensor.

        Unlike ``forward``, this method does not trigger hooks and is safe to
        call from within a forward hook on the same adapter.
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
        if preagg:
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


class HybridLastHopWrapper(nn.Module):
    """Wrapper that replaces only the deepest-hop raw features with
    pre-aggregated inputs for the K-1 frontier, for any model whose first
    MessagePassing layer has a registered :class:`PreAggAdapter`.

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
                f"HybridLastHopWrapper requires a model with a supported first "
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
    ) -> Tensor:
        use_hybrid = (
            frontier_mask is not None
            and aggregated_neighbors is not None
            and frontier_mask.numel() > 0
            and frontier_mask.any()
        )
        if not use_hybrid:
            return self.inner(x, edge_index)

        target_mask = aggregated_neighbors.abs().sum(dim=1) > 0

        def _zero_frontier(module, args, kw):
            x_in = args[0].clone()
            x_in[frontier_mask] = 0
            return (x_in,) + args[1:], kw

        def _patch_targets(module, args, output):
            if target_mask.any():
                patched = output.clone()
                patched[target_mask] = self._adapter.apply_preagg(
                    aggregated_neighbors[target_mask]
                )
                return patched
            return output

        h_pre = self._adapter.register_forward_pre_hook(_zero_frontier, with_kwargs=True)
        h_post = self._adapter.register_forward_hook(_patch_targets)
        try:
            out = self.inner(x, edge_index)
        finally:
            h_pre.remove()
            h_post.remove()
        return out


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

        # Not an MP layer — recurse one level deeper.
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
        Any ``nn.Module`` containing at least one MessagePassing layer
        registered in ``_ADAPTER_REGISTRY``.  The original model is not
        modified; a deep copy is used.

    Raises
    ------
    ValueError
        If the first MessagePassing layer found has no adapter (listing
        supported layer types), or if no MessagePassing layer is found at all.
    """
    inner = copy.deepcopy(model)
    adapter = _replace_first_mp_layer(inner)

    if adapter is None:
        supported = ", ".join(c.__name__ for c in _ADAPTER_REGISTRY)
        raise ValueError(
            "to_preaggregated_first_layer: no MessagePassing layer found in the "
            f"model. Supported layer types: {supported}."
        )

    wrapped = PreAggModelWrapper(inner, adapter)
    return PreAggResult(model=wrapped, preagg_spec=adapter.preagg_spec())


# Populate registry once all adapter classes are defined.
_ADAPTER_REGISTRY[GCNConv] = PreAggGCNConvAdapter


def to_hybrid_last_hop_gcn(model: nn.Module) -> nn.Module:
    """Wrap *model* so deepest-hop raw features can be replaced by
    pre-aggregated inputs for the K-1 frontier while keeping full topology.

    Works with any model whose first MessagePassing layer has a registered
    adapter in ``_ADAPTER_REGISTRY``.  The original model is not modified.
    """
    return HybridLastHopWrapper(model)
