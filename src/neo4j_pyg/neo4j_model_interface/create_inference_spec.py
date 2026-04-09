"""
create_inference_spec(model, output_dir)

Inspects a PyTorch / PyG GNN model and writes two files to output_dir/:

  spec.json    — layer-by-layer execution plan (architecture)
  weights.bin  — all referenced tensors in a keyed binary format

Java then reads both files to execute full GNN inference inside Neo4j via:

  CALL gnnProcedures.inference.run(
      $seed_ids, "id", "embedding_bytes", "byte[]", "Paper", "CITES",
      "/absolute/path/to/output_dir/"
  ) YIELD nodeId, predictedClass, logits

───────────────────────────────────────────────────────────────────────────

Adding support for a new aggregation type (e.g. SAGEConv):
  1. Add an entry to CONV_AGGREGATION_MAP below.
  2. Register a matching Java AggregationFn in GNNProcedures.AGGREGATION_REGISTRY.

spec.json layer ops
───────────────────
  {"op": "aggregate", "method": "<registry_key>"}
      Run the named Java aggregation on the current node representations.
      Each aggregate op consumes one hop ring of the pre-fetched subgraph.

  {"op": "linear", "weight": "<state_dict_key>", "bias": "<state_dict_key>"}
      Affine transform applied per-node.

  {"op": "relu"}
      Element-wise ReLU applied per-node.

weights.bin layout (all little-endian)
───────────────────────────────────────
  num_tensors : int32
  [repeated num_tensors times]
    key_length : int32
    key        : UTF-8 bytes[key_length]
    rank       : int32
    dims       : int32[rank]
    data       : float32[product(dims)]   (row-major)
"""

from __future__ import annotations

import json
import os
import struct
from typing import Callable, Optional

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GCNConv, SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


# ---------------------------------------------------------------------------
# Module-type → aggregation method mapping
# Keys must match entries in Java's GNNProcedures.AGGREGATION_REGISTRY.
# ---------------------------------------------------------------------------
CONV_AGGREGATION_MAP: dict[type, str] = {}

if _HAS_PYG:
    CONV_AGGREGATION_MAP[GCNConv] = "gcn_norm"
    # SAGEConv uses mean aggregation of neighbours then concatenates with self.
    # Uncomment once the Java "sage" aggregation is registered:
    # CONV_AGGREGATION_MAP[SAGEConv] = "sage"


# ---------------------------------------------------------------------------
# Per-module-type weight extraction
# Returns list of (weight_state_dict_key, bias_state_dict_key) tuples for the
# linear ops produced by this module type.  The state-dict keys are scoped to
# the full module path (e.g. "GCN1.lin.weight").
# ---------------------------------------------------------------------------
def _linear_keys_for(module_type: type, module_path: str) -> list[tuple[str, str]]:
    if _HAS_PYG and module_type is GCNConv:
        # GCNConv stores its transform in self.lin (no bias there) and self.bias.
        return [(f"{module_path}.lin.weight", f"{module_path}.bias")]
    if module_type is nn.Linear:
        return [(f"{module_path}.weight", f"{module_path}.bias")]
    return []


# ---------------------------------------------------------------------------
# Core spec builder
# ---------------------------------------------------------------------------
def _ops_for_module(
    name: str,
    module: nn.Module,
    is_last_child: bool,
    activation: str,
    siblings: list[tuple[str, nn.Module]],
    sibling_index: int,
) -> list[dict]:
    """Return the list of spec ops for a single named child module."""
    ops: list[dict] = []
    mt = type(module)

    if mt in CONV_AGGREGATION_MAP:
        method = CONV_AGGREGATION_MAP[mt]
        ops.append({"op": "aggregate", "method": method})
        for wk, bk in _linear_keys_for(mt, name):
            ops.append({"op": "linear", "weight": wk, "bias": bk})
        # Add activation unless the next sibling is an explicit activation module.
        next_is_activation = (
            sibling_index + 1 < len(siblings)
            and isinstance(siblings[sibling_index + 1][1], (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid))
        )
        if activation and not is_last_child and not next_is_activation:
            ops.append({"op": activation})

    elif isinstance(module, nn.Linear):
        ops.append({"op": "linear", "weight": f"{name}.weight", "bias": f"{name}.bias"})
        # No auto-activation for bare Linear — user adds nn.ReLU or it's the classifier.

    elif isinstance(module, (nn.ReLU, nn.GELU)):
        ops.append({"op": "relu"})

    elif isinstance(module, nn.Tanh):
        ops.append({"op": "tanh"})

    elif isinstance(module, nn.ModuleList):
        items = list(module.named_children())
        for i, (sub_name, sub_mod) in enumerate(items):
            full_name = f"{name}.{sub_name}"
            sub_last = i == len(items) - 1
            ops.extend(_ops_for_module(
                full_name, sub_mod, sub_last, activation, items, i
            ))
            # Add relu between linear layers (but not after the last one).
            if not sub_last and isinstance(sub_mod, nn.Linear):
                ops.append({"op": activation})

    return ops


# ---------------------------------------------------------------------------
# Internal helpers shared by both public functions
# ---------------------------------------------------------------------------

def _build_spec(
    model: nn.Module,
    *,
    activation: str,
    num_hops: Optional[int],
    max_neighbors: int,
) -> dict:
    """Return the spec dict for *model* without touching disk or Neo4j."""
    children = list(model.named_children())
    layers: list[dict] = []
    for i, (name, module) in enumerate(children):
        is_last = i == len(children) - 1
        layers.extend(_ops_for_module(name, module, is_last, activation, children, i))

    agg_count = sum(1 for op in layers if op.get("op") == "aggregate")
    return {
        "num_hops": num_hops if num_hops is not None else agg_count,
        "max_neighbors": max_neighbors,
        "layers": layers,
    }


def _referenced_tensors(model: nn.Module, spec: dict) -> dict[str, "torch.Tensor"]:
    """Extract the state-dict tensors referenced by *spec* from *model*."""
    state = model.state_dict()
    referenced: list[str] = []
    for op in spec["layers"]:
        for field in ("weight", "bias"):
            key = op.get(field)
            if key and key not in referenced:
                referenced.append(key)

    missing = [k for k in referenced if k not in state]
    if missing:
        raise ValueError(
            "The following keys appear in the spec but are missing from the "
            "model's state_dict — check CONV_AGGREGATION_MAP or _linear_keys_for:\n"
            + "\n".join(f"  {k}" for k in missing)
        )
    return {k: state[k] for k in referenced}


def _write_weights(path: str, tensors: dict[str, "torch.Tensor"]) -> None:
    """Write *tensors* to a keyed binary file at *path*."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(tensors)))
        for key, tensor in tensors.items():
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<i", len(key_bytes)))
            f.write(key_bytes)
            shape = list(tensor.shape)
            f.write(struct.pack("<i", len(shape)))
            if shape:
                f.write(struct.pack(f"<{len(shape)}i", *shape))
            f.write(tensor.detach().cpu().numpy().astype("<f4").tobytes())


def _print_summary(spec: dict, tensors: dict, label: str) -> None:
    print(f"Inference spec ready ({label})")
    print(f"  num_hops     : {spec['num_hops']}")
    print(f"  max_neighbors: {spec['max_neighbors']}")
    print(f"  ops          : {len(spec['layers'])}")
    for op in spec["layers"]:
        print(f"    {op}")
    print(f"  tensors      : {len(tensors)}")
    for k, t in tensors.items():
        print(f"    {k}: {list(t.shape)}")


# ---------------------------------------------------------------------------
# Public API — file-based
# ---------------------------------------------------------------------------

def create_inference_spec(
    model: nn.Module,
    model_name: str,
    *,
    base_dir: Optional[str] = None,
    activation: str = "relu",
    num_hops: Optional[int] = None,
    max_neighbors: int = 10,
) -> str:
    """
    Inspect *model* and write ``spec.json`` + ``weights.bin`` to a directory.

    The output directory is ``<base_dir>/<model_name>/``.  *base_dir* defaults
    to the ``NEO4J_GNN_MODEL_DIR`` environment variable, which is also where
    the Java procedure looks for models at inference time — so you only need to
    set that variable once and both sides agree on the location.

    Parameters
    ----------
    model
        Trained PyTorch model.
    model_name
        Name passed to ``gnnProcedures.inference.run`` in Cypher.
    base_dir
        Root directory that contains model sub-folders.  Defaults to the
        ``NEO4J_GNN_MODEL_DIR`` environment variable.

    Returns the absolute path to the model directory.
    """
    if base_dir is None:
        base_dir = os.environ.get("NEO4J_GNN_MODEL_DIR")
    if not base_dir:
        raise ValueError(
            "base_dir not given and NEO4J_GNN_MODEL_DIR is not set. "
            "Either pass base_dir explicitly or export NEO4J_GNN_MODEL_DIR."
        )

    output_dir = os.path.join(base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    spec    = _build_spec(model, activation=activation, num_hops=num_hops, max_neighbors=max_neighbors)
    tensors = _referenced_tensors(model, spec)

    with open(os.path.join(output_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)

    _write_weights(os.path.join(output_dir, "weights.bin"), tensors)

    _print_summary(spec, tensors, f"files → {os.path.abspath(output_dir)}/")
    return os.path.abspath(output_dir)


