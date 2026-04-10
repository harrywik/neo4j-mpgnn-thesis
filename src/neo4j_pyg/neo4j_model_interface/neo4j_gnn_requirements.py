"""Pydantic configuration models for GNN architectures with DB compatibility enforcement.

Three config classes encode the constraints for each supported DB operation mode:

  NeighborAggGNNConfig  — first MessagePassing layer must be DB-aggregatable
                          (used when to_preaggregated_first_layer=True in config)

  InferenceGNNConfig    — ALL MessagePassing layers must be in Java AGG_REGISTRY_F
                          (used when the model is exported via create_inference_spec
                          and run via gnnProcedures.inference.run)

  DualModeGNNConfig     — both constraints (first-layer agg + full inference)

Each config exposes build_model(in_dim) -> GenericGNN, a plain nn.Module whose
named_children() layout is compatible with create_inference_spec._build_spec.

Adding a new layer type (e.g. SAGEConv):
  1. Add an entry to _DB_INFERENCE_READY / _DB_AGGREGATABLE as appropriate.
  2. Register a matching Java AggregationFn in GNNProcedures.AGGREGATION_REGISTRY.
  3. Add to CONV_AGGREGATION_MAP in create_inference_spec.py.
  4. Implement a PreAggAdapter subclass in adapters.py if neighbor-agg
     support is also needed.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, model_validator

try:
    import torch.nn as nn
    from torch import Tensor
    from torch_geometric.nn import GCNConv
    from torch_geometric.nn.conv import MessagePassing
    _HAS_PYG = True
except ImportError:  # pragma: no cover
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConvLayerType(str, Enum):
    """PyG MessagePassing layer types usable in a GNNLayerDef."""
    GCNConv = "GCNConv"
    # SAGEConv = "SAGEConv"  # Uncomment once Java "sage" aggregation is registered


class Activation(str, Enum):
    """Activations supported by the Java inference engine (LayerSpec.op)."""
    relu = "relu"
    tanh = "tanh"


# ---------------------------------------------------------------------------
# DB-compatibility sets
# These must stay in sync with:
#   Java  → GNNProcedures.AGGREGATION_REGISTRY / AGG_REGISTRY_F
#   Python → CONV_AGGREGATION_MAP in create_inference_spec.py
#   Python → _ADAPTER_REGISTRY in adapters.py
# ---------------------------------------------------------------------------

# Layers whose first-hop aggregation can be offloaded to Neo4j and later
# bypassed by a PreAggAdapter (message function ⊥ aggregation).
_DB_AGGREGATABLE: frozenset[ConvLayerType] = frozenset({ConvLayerType.GCNConv})

# Layers for which Java can execute the full aggregate+linear+activation cycle
# inside gnnProcedures.inference.run (must be in Java AGG_REGISTRY_F).
_DB_INFERENCE_READY: frozenset[ConvLayerType] = frozenset({ConvLayerType.GCNConv})

# Attention-based layers that are structurally incompatible with DB aggregation
# (kept for documentation; used in error messages only).
_UNSUPPORTED_FOR_DB: frozenset[str] = frozenset({"GATConv", "TransformerConv"})


# ---------------------------------------------------------------------------
# Per-layer specification
# ---------------------------------------------------------------------------

class GNNLayerDef(BaseModel):
    """Describes one message-passing layer in the GNN stack.

    Parameters
    ----------
    conv_type:
        The PyG layer class to instantiate.
    out_dim:
        Output feature dimension.  The input dimension is inferred from the
        previous layer (or from ``in_dim`` passed to ``build_model``).
    activation:
        Activation applied after this layer.  ``None`` means no activation
        (e.g. last conv layer before a custom classifier head).
        Only ``relu`` / ``tanh`` are supported for in-DB inference.
    """
    conv_type: ConvLayerType
    out_dim: int
    activation: Optional[Activation] = Activation.relu


# ---------------------------------------------------------------------------
# GenericGNN — nn.Module built from a list of GNNLayerDef
# ---------------------------------------------------------------------------

class GenericGNN(nn.Module):
    """Dynamically assembled GNN from a validated list of GNNLayerDef.

    Layer naming convention mirrors the hand-written GCN class so that
    ``create_inference_spec._build_spec`` can walk named_children() correctly:

        conv_0, relu_0, conv_1, relu_1, ..., classifier

    Activation modules are registered as explicit named children (nn.ReLU /
    nn.Tanh) rather than applied via in-place ops.  This lets the spec builder
    see them as siblings of their parent conv layer and skip auto-insertion.
    """

    def __init__(
        self,
        layers: List[GNNLayerDef],
        in_dim: int,
        num_classes: int,
    ) -> None:
        if not _HAS_PYG:  # pragma: no cover
            raise ImportError("torch_geometric is required to build GenericGNN")
        super().__init__()

        current_dim = in_dim
        fwd: list[tuple[str, bool]] = []  # (attr_name, is_conv)

        for i, layer_def in enumerate(layers):
            if layer_def.conv_type is ConvLayerType.GCNConv:
                mod: nn.Module = GCNConv(current_dim, layer_def.out_dim)
            else:  # pragma: no cover
                raise ValueError(
                    f"ConvLayerType {layer_def.conv_type.value!r} is not implemented "
                    f"in GenericGNN. Add a branch in GenericGNN.__init__."
                )
            conv_name = f"conv_{i}"
            setattr(self, conv_name, mod)
            fwd.append((conv_name, True))
            current_dim = layer_def.out_dim

            if layer_def.activation is Activation.relu:
                act_name = f"relu_{i}"
                setattr(self, act_name, nn.ReLU())
                fwd.append((act_name, False))
            elif layer_def.activation is Activation.tanh:
                act_name = f"tanh_{i}"
                setattr(self, act_name, nn.Tanh())
                fwd.append((act_name, False))
            # activation=None → no activation registered

        self.classifier = nn.Linear(current_dim, num_classes)
        fwd.append(("classifier", False))

        # Plain list — not an nn.Module, so it won't appear in named_children().
        self._fwd: list[tuple[str, bool]] = fwd

    def forward(self, X: Tensor, edge_index: Tensor) -> Tensor:
        for name, is_conv in self._fwd:
            mod = getattr(self, name)
            X = mod(X, edge_index) if is_conv else mod(X)
        return X


# ---------------------------------------------------------------------------
# Config 1 — Neighbor aggregation in DB (first layer only)
# ---------------------------------------------------------------------------

class NeighborAggGNNConfig(BaseModel):
    """GNN where the *first* layer's aggregation runs server-side in Neo4j.

    The Java ``gnnProcedures.aggregation.neighbor.mean`` (or ``gcn_norm``) UDP
    computes the first-hop aggregation; remaining layers execute in PyTorch via
    a PreAggAdapter wrapper (``to_preaggregated_first_layer``).

    Constraint
    ----------
    The first layer's ``conv_type`` must be in ``_DB_AGGREGATABLE`` so that a
    ``PreAggAdapter`` exists for it.  Attention-based layers (GATConv,
    TransformerConv) are structurally blocked — their message function cannot
    be separated from aggregation.
    """

    layers: List[GNNLayerDef]
    num_classes: int
    max_neighbors: int = 10

    @model_validator(mode="after")
    def _check_first_layer(self) -> NeighborAggGNNConfig:
        if not self.layers:
            raise ValueError("At least one GNNLayerDef is required.")
        first = self.layers[0]
        if first.conv_type not in _DB_AGGREGATABLE:
            supported = [t.value for t in _DB_AGGREGATABLE]
            raise ValueError(
                f"First layer conv_type must be one of {supported} for DB-side "
                f"neighbor aggregation; got {first.conv_type.value!r}. "
                f"Attention-based layers ({', '.join(sorted(_UNSUPPORTED_FOR_DB))}) "
                f"cannot separate their message function from aggregation."
            )
        return self

    def build_model(self, in_dim: int) -> GenericGNN:
        """Construct the nn.Module.

        Wrap the result with ``to_preaggregated_first_layer()`` before passing
        it to the training loop when DB aggregation is active.
        """
        return GenericGNN(self.layers, in_dim, self.num_classes)


# ---------------------------------------------------------------------------
# Config 2 — Full inference in DB
# ---------------------------------------------------------------------------

class InferenceGNNConfig(BaseModel):
    """GNN whose *entire* forward pass runs inside Neo4j via
    ``gnnProcedures.inference.run``.

    Constraints
    -----------
    * Every ``conv_type`` must be in ``_DB_INFERENCE_READY`` — i.e. present in
      Java's ``AGG_REGISTRY_F`` and in Python's ``CONV_AGGREGATION_MAP``.
    * Activations must be ``relu`` or ``tanh`` — the only ops the Java engine
      recognises in a ``LayerSpec``.
    """

    layers: List[GNNLayerDef]
    num_classes: int
    model_name: str = "experiment_gcn"
    num_hops: Optional[int] = None
    max_neighbors: int = 10

    @model_validator(mode="after")
    def _check_all_layers(self) -> InferenceGNNConfig:
        if not self.layers:
            raise ValueError("At least one GNNLayerDef is required.")
        for i, layer in enumerate(self.layers):
            if layer.conv_type not in _DB_INFERENCE_READY:
                supported = [t.value for t in _DB_INFERENCE_READY]
                raise ValueError(
                    f"Layer {i} ({layer.conv_type.value!r}) is not supported for "
                    f"in-DB inference. All layers must be one of {supported}. "
                    f"Java AGG_REGISTRY_F currently contains: gcn_norm, mean."
                )
        return self

    def build_model(self, in_dim: int) -> GenericGNN:
        """Construct the nn.Module.

        Pass the result to ``create_inference_spec`` or ``upload_inference_spec``
        to export it for in-database execution.
        """
        return GenericGNN(self.layers, in_dim, self.num_classes)


# ---------------------------------------------------------------------------
# Config 3 — Dual mode (neighbor agg + full inference)
# ---------------------------------------------------------------------------

class DualModeGNNConfig(BaseModel):
    """GNN that supports *both* DB-side neighbor aggregation *and* full in-DB
    inference — the intersection of the two constraint sets.

    Any model built from this config can be:
    * Wrapped with ``to_preaggregated_first_layer`` for hybrid training.
    * Exported with ``create_inference_spec`` / ``upload_inference_spec`` for
      fully in-database inference at serve time.
    """

    layers: List[GNNLayerDef]
    num_classes: int
    model_name: str = "experiment_gcn"
    num_hops: Optional[int] = None
    max_neighbors: int = 10

    @model_validator(mode="after")
    def _check_dual_mode(self) -> DualModeGNNConfig:
        if not self.layers:
            raise ValueError("At least one GNNLayerDef is required.")
        dual_supported = _DB_AGGREGATABLE & _DB_INFERENCE_READY
        supported_names = [t.value for t in dual_supported]
        for i, layer in enumerate(self.layers):
            if layer.conv_type not in dual_supported:
                raise ValueError(
                    f"Layer {i} ({layer.conv_type.value!r}) does not satisfy "
                    f"dual-mode constraints (DB aggregation + DB inference). "
                    f"All layers must be one of {supported_names}."
                )
        return self

    def build_model(self, in_dim: int) -> GenericGNN:
        return GenericGNN(self.layers, in_dim, self.num_classes)
