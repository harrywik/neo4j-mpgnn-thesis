"""cypher_inference.py — Cypher-only GNN inference engine.

Builds a single, read-only Cypher query from a spec.json + weights.bin pair and
runs it against Neo4j via Bolt.  No plugin procedures are required.

Pipeline per batch call
-----------------------
  1. Sampler block (from SAMPLER_REGISTRY[spec["sampler"]]) — multi-hop subgraph.
  2. Init block — builds h_map, in_degrees, adj maps in Cypher.
  3. Per-layer blocks  — aggregate / linear / relu / tanh, one WITH-continuation each.
  4. Argmax block — UNWIND seeds, reduce over logits, RETURN nodeId + predictedClass.

The query string is assembled *once* at experiment init.  Only $seed_ids changes
between batches; weight tensors are bound as static params ($W_0, $b_0, $W_2, ...).
Shape constants (out_dim, in_dim, feat_dim, active_level) are inlined into the
query string at build time so that the query plan is stable across batches.

Adding a new aggregation: add one entry to AGG_REGISTRY.
Adding a new sampler:     add one entry to SAMPLER_REGISTRY.
Both registries follow the extensibility pattern of Java's AGGREGATION_REGISTRY
in GNNProcedures.java.
"""

from __future__ import annotations

import struct
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# 1. Weights binary parser  (mirrors Java parseWeightsFromChannel)
# ---------------------------------------------------------------------------

def parse_weights_bin(path: str) -> dict[str, np.ndarray]:
    """Parse weights.bin written by create_inference_spec._write_weights.

    Format (all little-endian):
      num_tensors : int32
      [repeated num_tensors times]
        key_length : int32
        key        : UTF-8 bytes[key_length]
        rank       : int32
        dims       : int32[rank]
        data       : float32[prod(dims)]  (row-major)

    Returns a dict mapping tensor key → float32 numpy array (shape preserved).
    """
    result: dict[str, np.ndarray] = {}
    with open(path, "rb") as f:
        (num_tensors,) = struct.unpack("<i", f.read(4))
        for _ in range(num_tensors):
            (key_len,) = struct.unpack("<i", f.read(4))
            key = f.read(key_len).decode("utf-8")
            (rank,) = struct.unpack("<i", f.read(4))
            dims = list(struct.unpack(f"<{rank}i", f.read(4 * rank))) if rank > 0 else []
            total = int(np.prod(dims)) if dims else 1
            data = np.frombuffer(f.read(4 * total), dtype="<f4").copy()
            result[key] = data.reshape(dims) if dims else data
    return result


# ---------------------------------------------------------------------------
# 2. Sampler registry
# ---------------------------------------------------------------------------

@dataclass
class SamplerCtx:
    """Context passed to every sampler template builder."""
    node_label: str   # "" → no label filter
    edge_type: str    # "" → match any relationship type
    nodeid_prop: str  # property key holding the application-level node ID
    feature_prop: str # f64[] property name consumed by the init block
    fanouts: list     # per-hop fan-out; -1 = take all


def _build_neighbor_uniform_sampler(ctx: SamplerCtx) -> str:
    """Multi-hop uniform neighbour sampler, ported from Neo4jSampler._build_fanout_query.

    Uses apoc.coll.randomItems for sampling (same as Neo4jSampler.py).
    Follows INCOMING edges: (neighbor)-[:EDGE]->(src).

    Block ends with:
      WITH visited AS ordered_nodes, edges, nodes_by_hop
    so the init block can continue directly.
    """
    nid       = ctx.nodeid_prop
    s_label   = f":{ctx.node_label}" if ctx.node_label else ""
    nbr_label = f":{ctx.node_label}" if ctx.node_label else ""
    rel       = f":{ctx.edge_type}"  if ctx.edge_type  else ""
    edge_pat  = f"<-[r{rel}]-"

    parts: list[str] = []

    # Seed initialisation
    parts.append(f"""\
UNWIND range(0, size($seed_ids)-1) AS i
WITH i, $seed_ids[i] AS seed_id
MATCH (s{s_label})
WHERE s.{nid} = seed_id
WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges, [collect(s)] AS nodes_by_hop""")

    for k in ctx.fanouts:
        take_all_cond = f"{k} < 0 OR {k} >= size(cand_rels)"
        parts.append(f"""\
CALL (frontier, visited, edges, nodes_by_hop) {{
    UNWIND range(0, size(frontier)-1) AS i
    WITH i, frontier[i] AS src, visited, edges
    MATCH (src){edge_pat}(neighbor{nbr_label})
    WITH i, src, visited, edges, collect(r) AS cand_rels
    WITH i, src, visited, edges,
        CASE
            WHEN {take_all_cond}
            THEN cand_rels
            ELSE apoc.coll.randomItems(cand_rels, {k}, false)
        END AS picked_rels
    WITH i, visited, edges,
        [rel IN picked_rels | startNode(rel)]                                       AS picked_nbrs,
        [rel IN picked_rels | [startNode(rel).{nid}, endNode(rel).{nid}]]           AS new_edges
    ORDER BY i
    WITH visited, edges,
        apoc.coll.flatten(collect(picked_nbrs))  AS picked_nbrs,
        apoc.coll.flatten(collect(new_edges))    AS new_edges
    WITH visited, edges, new_edges,
        apoc.coll.toSet(
            [n IN picked_nbrs WHERE NOT n IN visited]
        ) AS next_frontier
    RETURN
        next_frontier,
        visited + next_frontier             AS next_visited,
        edges + new_edges                   AS next_edges,
        nodes_by_hop + [next_frontier]      AS next_nodes_by_hop
}}
WITH next_frontier AS frontier,
     next_visited  AS visited,
     next_edges    AS edges,
     next_nodes_by_hop AS nodes_by_hop""")

    parts.append("WITH visited AS ordered_nodes, edges, nodes_by_hop")
    return "\n".join(parts)


SAMPLER_REGISTRY: dict[str, Callable[[SamplerCtx], str]] = {
    "neighbor_uniform": _build_neighbor_uniform_sampler,
}


# ---------------------------------------------------------------------------
# 3. Init block  (h_map, in_degrees, adj from sampler output)
# ---------------------------------------------------------------------------

def _build_init_block(ctx: SamplerCtx) -> str:
    """Emit the Cypher block that builds the three maps consumed by layer blocks.

    Requires ``ordered_nodes``, ``edges``, ``nodes_by_hop`` in scope (from sampler).

    Produces:
      h_map       : {toString(nodeId) → featureVector as List<Float>}
      in_degrees  : {toString(nodeId) → in_degree as Float}
      adj         : {toString(dstNodeId) → [toString(srcNodeId), ...]}
    """
    nid  = ctx.nodeid_prop
    feat = ctx.feature_prop
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop,
    apoc.map.fromPairs(
        [n IN ordered_nodes | [toString(n.{nid}), n.{feat}]]
    ) AS h_map,
    apoc.map.fromPairs(
        [n IN ordered_nodes | [toString(n.{nid}), toFloat(coalesce(n.in_degree, 0))]]
    ) AS in_degrees,
    apoc.map.fromPairs(
        [dst IN ordered_nodes |
            [toString(dst.{nid}),
             [e IN edges WHERE e[1] = dst.{nid} | toString(e[0])]]
        ]
    ) AS adj"""


# ---------------------------------------------------------------------------
# 4. Aggregation-template registry
# ---------------------------------------------------------------------------

def _build_gcn_norm_block(active_level: int, feat_dim: int, nid: str) -> str:
    """GCN-normalised aggregation with self-loop.

    self-weight  = 1 / d̂_v
    neighbour u  = 1 / sqrt(d̂_v * d̂_u)   where d̂ = in_degree + 1

    Mirrors Java AGG_REGISTRY_F["gcn_norm"] (GNNProcedures.java lines 183-200).
    """
    al = active_level
    fd = feat_dim
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {al}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 reduce(
                     _acc = [_i IN range(0, {fd} - 1) |
                                 h_map[toString(v.{nid})][_i]
                                 / (in_degrees[toString(v.{nid})] + 1.0)],
                     _u_str IN coalesce(adj[toString(v.{nid})], []) |
                         [_i IN range(0, {fd} - 1) |
                              _acc[_i] +
                              h_map[_u_str][_i]
                              / sqrt(
                                    (in_degrees[toString(v.{nid})] + 1.0)
                                    * (in_degrees[_u_str] + 1.0)
                                )
                         ]
                 )
                ]
        ])
    ) AS h_map"""


def _build_mean_block(active_level: int, feat_dim: int, nid: str) -> str:
    """Mean aggregation over neighbours (no self-loop, no normalisation).

    Falls back to the node's own features when it has no sampled neighbours.
    Mirrors Java AGG_REGISTRY_F["mean"] (GNNProcedures.java lines 202-215).

    We compute mean element-by-element to avoid nested list division.
    """
    al = active_level
    fd = feat_dim
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {al}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 CASE
                     WHEN size(coalesce(adj[toString(v.{nid})], [])) = 0
                     THEN h_map[toString(v.{nid})]
                     ELSE
                         [_i IN range(0, {fd} - 1) |
                             reduce(
                                 _s = 0.0,
                                 _u_str IN coalesce(adj[toString(v.{nid})], []) |
                                     _s + h_map[_u_str][_i]
                             )
                             / toFloat(size(coalesce(adj[toString(v.{nid})], [])))
                         ]
                 END
                ]
        ])
    ) AS h_map"""


# Registry maps method name → builder function.
# Signature: (active_level: int, feat_dim: int, nid: str) -> str
AGG_REGISTRY: dict[str, Callable[[int, int, str], str]] = {
    "gcn_norm": _build_gcn_norm_block,
    "mean":     _build_mean_block,
}


# ---------------------------------------------------------------------------
# 5. Linear / relu / tanh layer templates
# ---------------------------------------------------------------------------

def _build_linear_block(
    last_agg_level: int,
    layer_idx: int,
    out_dim: int,
    in_dim: int,
    nid: str,
) -> str:
    """Affine transform: h_v = W @ h_v + b.

    $W_{layer_idx}  — flat float list parameter, row-major [out_dim, in_dim].
    $b_{layer_idx}  — float list parameter [out_dim].
    out_dim / in_dim are inlined as integer literals.
    """
    la = last_agg_level
    l  = layer_idx
    od = out_dim
    id_ = in_dim
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {la}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 [_oi IN range(0, {od} - 1) |
                     reduce(
                         _s = $b_{l}[_oi],
                         _j IN range(0, {id_} - 1) |
                             _s + $W_{l}[_oi * {id_} + _j]
                                * h_map[toString(v.{nid})][_j]
                     )
                 ]
                ]
        ])
    ) AS h_map"""


def _build_relu_block(last_agg_level: int, nid: str) -> str:
    la = last_agg_level
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {la}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 [_x IN h_map[toString(v.{nid})] |
                     CASE WHEN _x < 0.0 THEN 0.0 ELSE _x END]
                ]
        ])
    ) AS h_map"""


def _build_tanh_block(last_agg_level: int, nid: str) -> str:
    la = last_agg_level
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {la}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 [_x IN h_map[toString(v.{nid})] |
                     (exp(_x) - exp(-_x)) / (exp(_x) + exp(-_x))]
                ]
        ])
    ) AS h_map"""


# ---------------------------------------------------------------------------
# 6. Final argmax block
# ---------------------------------------------------------------------------

def _build_argmax_block() -> str:
    return """\
WITH h_map, $seed_ids AS _seeds
UNWIND _seeds AS _sid
WITH _sid, h_map[toString(_sid)] AS _logits
WHERE _logits IS NOT NULL
RETURN _sid AS nodeId,
    reduce(_best = 0, _i IN range(1, size(_logits) - 1) |
        CASE WHEN _logits[_i] > _logits[_best] THEN _i ELSE _best END
    ) AS predictedClass"""


# ---------------------------------------------------------------------------
# 7. Prepared inference state
# ---------------------------------------------------------------------------

@dataclass
class CypherInferencePrepared:
    """Pre-built artefacts for the in_db_cypher strategy.  Built once per experiment."""
    query: str
    static_params: dict[str, Any]
    database: str


# ---------------------------------------------------------------------------
# 8. Query builder
# ---------------------------------------------------------------------------

def build_cypher_inference_query(
    spec: dict,
    weights: dict[str, np.ndarray],
    *,
    node_label: str,
    edge_type: str,
    nodeid_prop: str,
    feature_prop: str,
) -> tuple[str, dict[str, Any]]:
    """Assemble the full Cypher query and its static parameter map from a spec.

    Parameters
    ----------
    spec:
        Parsed spec.json dict.  Must contain ``num_hops``, ``max_neighbors``,
        ``layers``, and optionally ``sampler`` (default ``"neighbor_uniform"``).
    weights:
        Dict of tensor-name → numpy array from ``parse_weights_bin``.
    node_label / edge_type / nodeid_prop / feature_prop:
        Dataset-specific graph-schema identifiers.

    Returns
    -------
    (cypher_string, static_params)
        ``static_params`` holds every $W_L and $b_L binding.
        The caller only needs to add ``{"seed_ids": [...]}`` per batch.
    """
    fanouts: list = spec["max_neighbors"]
    if isinstance(fanouts, int):
        fanouts = [fanouts] * spec["num_hops"]

    sampler_key: str = spec.get("sampler", "neighbor_uniform")
    if sampler_key not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{sampler_key}'. Available: {list(SAMPLER_REGISTRY)}"
        )

    ctx = SamplerCtx(
        node_label=node_label,
        edge_type=edge_type,
        nodeid_prop=nodeid_prop,
        feature_prop=feature_prop,
        fanouts=fanouts,
    )

    blocks: list[str] = []
    static_params: dict[str, Any] = {}

    # ── 1. Sampler block ────────────────────────────────────────────────────
    blocks.append(SAMPLER_REGISTRY[sampler_key](ctx))

    # ── 2. Init block ───────────────────────────────────────────────────────
    blocks.append(_build_init_block(ctx))

    # ── 3. Layer blocks (activeLevel / lastAggLevel mirror Java runEngine) ──
    num_hops        = spec["num_hops"]
    active_level    = max(0, num_hops - 1)
    last_agg_level  = max(0, num_hops - 1)
    layer_idx       = 0  # unique integer suffix for $W_{l} / $b_{l}

    for layer in spec["layers"]:
        op = layer["op"]

        if op == "aggregate":
            method = layer["method"]
            if method not in AGG_REGISTRY:
                raise ValueError(
                    f"Unknown aggregation '{method}'. Available: {list(AGG_REGISTRY)}"
                )
            feat_dim = _next_linear_in_dim(spec["layers"], weights, layer)
            block = AGG_REGISTRY[method](active_level, feat_dim, nodeid_prop)
            last_agg_level = active_level
            active_level   = max(0, active_level - 1)

        elif op == "linear":
            W = weights[layer["weight"]]
            b = weights[layer["bias"]]
            if W.ndim != 2:
                raise ValueError(
                    f"Weight '{layer['weight']}' has unexpected shape {W.shape}; expected rank-2."
                )
            out_dim, in_dim = W.shape
            static_params[f"W_{layer_idx}"] = W.flatten().tolist()
            static_params[f"b_{layer_idx}"] = b.flatten().tolist()
            block = _build_linear_block(
                last_agg_level, layer_idx, int(out_dim), int(in_dim), nodeid_prop
            )

        elif op == "relu":
            block = _build_relu_block(last_agg_level, nodeid_prop)

        elif op == "tanh":
            block = _build_tanh_block(last_agg_level, nodeid_prop)

        else:
            raise ValueError(f"Unknown op '{op}' in spec layers.")

        blocks.append(block)
        layer_idx += 1

    # ── 4. Argmax block ─────────────────────────────────────────────────────
    blocks.append(_build_argmax_block())

    cypher = "\n".join(blocks)
    return cypher, static_params


def _next_linear_in_dim(
    layers: list[dict],
    weights: dict[str, np.ndarray],
    agg_layer: dict,
) -> int:
    """Look ahead from agg_layer to find the in_dim of the next linear op.

    This tells us the feature dimension that the aggregation output must match,
    needed to bound the ``reduce`` loop in the gcn_norm / mean templates.
    Falls back to 0 (no loop) if no linear follows.
    """
    found = False
    for layer in layers:
        if layer is agg_layer:
            found = True
            continue
        if found and layer["op"] == "linear":
            W = weights.get(layer["weight"])
            if W is not None and W.ndim == 2:
                return int(W.shape[1])
            break
    return 0


# ---------------------------------------------------------------------------
# 9. Feature property materializer  (byte[] → f64[])
# ---------------------------------------------------------------------------

def materialize_float_features(
    driver,
    *,
    database: str,
    node_label: str,
    nodeid_prop: str,
    byte_prop: str,
    float_prop: str,
    batch_size: int = 500,
) -> int:
    """Decode a packed float32 byte[] node property to a sibling f64[] property.

    Cora stores features as little-endian IEEE-754 float32 packed in a byte[]
    (``feature_property_type: "byte[]"``).  Pure Cypher cannot decode this, so
    we do it in Python and write the f64[] back in batches.

    This is a *one-time* setup step; cost is reported separately from per-batch
    inference timing.  Nodes that already have ``float_prop`` set are skipped.

    Returns the number of nodes updated.
    """
    print(
        f"  [cypher] Materialising float features "
        f"'{byte_prop}' → '{float_prop}' on :{node_label}…"
    )

    fetch_q = (
        f"MATCH (n:{node_label}) "
        f"WHERE n.{byte_prop} IS NOT NULL AND n.{float_prop} IS NULL "
        f"RETURN n.{nodeid_prop} AS nid, n.{byte_prop} AS raw"
    )
    set_q = (
        f"UNWIND $rows AS row "
        f"MATCH (n:{node_label} {{{nodeid_prop}: row.nid}}) "
        f"SET n.{float_prop} = row.floats"
    )

    updated = 0
    batch: list[dict] = []

    with driver.session(database=database, fetch_size=-1) as session:
        result = session.run(fetch_q)
        for record in result:
            raw: bytes = bytes(record["raw"])
            n_floats = len(raw) // 4
            floats = list(struct.unpack_from(f"<{n_floats}f", raw))
            batch.append({"nid": record["nid"], "floats": floats})

            if len(batch) >= batch_size:
                with driver.session(database=database) as ws:
                    ws.run(set_q, rows=batch)
                updated += len(batch)
                batch = []

    if batch:
        with driver.session(database=database) as ws:
            ws.run(set_q, rows=batch)
        updated += len(batch)

    print(f"  [cypher] Materialised {updated} nodes.")
    return updated


# ---------------------------------------------------------------------------
# 10. Inference driver
# ---------------------------------------------------------------------------

def run_cypher_inference(
    driver,
    database: str,
    query: str,
    static_params: dict[str, Any],
    seed_ids: list[int],
    batch_size: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    """Run the prebuilt combined Cypher inference query over seed_ids in batches.

    Per batch a single Bolt round-trip is made, mirroring run_in_db_java.

    Parameters
    ----------
    driver:         neo4j.Driver instance.
    database:       Target database name.
    query:          Combined Cypher string from build_cypher_inference_query.
    static_params:  $W_L / $b_L parameters, constant across batches.
    seed_ids:       Application-level node IDs to classify.
    batch_size:     Number of seed IDs per Bolt call.

    Returns
    -------
    (preds, metrics)
        preds   : {nodeId → predicted_class}
        metrics : timing / memory / batch-latency dict matching run_in_db_java output.
    """
    preds: dict[int, int] = {}
    batch_latencies: list[float] = []

    tracemalloc.start()
    t_total = time.monotonic()

    for i in range(0, len(seed_ids), batch_size):
        batch = seed_ids[i: i + batch_size]
        t_batch = time.monotonic()
        params = {**static_params, "seed_ids": batch}
        with driver.session(database=database, fetch_size=-1) as session:
            result = session.run(query, params)
            for record in result:
                preds[int(record["nodeId"])] = int(record["predictedClass"])
        batch_latencies.append((time.monotonic() - t_batch) * 1000)

    elapsed = time.monotonic() - t_total
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = np.array(batch_latencies)
    metrics = {
        "total_time_s": elapsed,
        "ms_per_node": elapsed * 1000 / max(len(seed_ids), 1),
        "throughput_nodes_per_s": len(seed_ids) / max(elapsed, 1e-9),
        "peak_memory_mb": peak / 1024 / 1024,
        "p50_batch_ms": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95_batch_ms": float(np.percentile(lat, 95)) if len(lat) else None,
        "n_batches": len(batch_latencies),
    }
    return preds, metrics


# ===========================================================================
# OPTIMIZED CYPHER INFERENCE  —  vector_distance linear blocks
#
# Identical to in_db_cypher but replaces the interpreted reduce(...) inner
# loop inside every linear block with vector_distance(vector(..., FLOAT32),
# vector(..., FLOAT32), DOT).
#
# vector_distance(a, b, DOT) = -(a · b)  (compiled JVM primitive-float code,
# available since Neo4j 2025.10).
#
# h_out[i] = W[i] · h + b[i]
#           = b[i] - vector_distance(vector(h, d, FLOAT32),
#                                    vector(W_row_i, d, FLOAT32), DOT)
#
# The outer list-comprehension [_oi IN range(0, out_dim-1) | ...] still runs
# in Cypher, but each iteration dispatches one compiled-JVM call instead of
# a ~1433-step interpreted reduce loop.
#
# Aggregation blocks (gcn_norm, mean) compute weighted sums over neighbours
# and are unchanged; after the first linear layer the feature dimension
# drops to 64 so subsequent aggregations are cheap regardless.
# ===========================================================================

# ---------------------------------------------------------------------------
# 11. vector_distance-based linear block
# ---------------------------------------------------------------------------

def _build_vd_linear_block(
    last_agg_level: int,
    layer_idx: int,
    out_dim: int,
    in_dim: int,
    nid: str,
) -> str:
    """Linear block that uses vector_distance(..., DOT) for the dot product.

    Replaces the inner ``reduce`` loop with a single compiled-JVM call:

        h_out[i] = b[i] - vector_distance(vector(h, in_dim, FLOAT32),
                                           vector(W_row_i, in_dim, FLOAT32),
                                           DOT)

    Requires Neo4j ≥ 2025.10.  $W_L and $b_L are the same Bolt parameters
    as in the baseline linear block — no model node needed.
    """
    la  = last_agg_level
    l   = layer_idx
    od  = out_dim
    id_ = in_dim
    return f"""\
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.coll.flatten([_d IN range(0, {la}) | nodes_by_hop[_d]]) AS _active,
    h_map
WITH ordered_nodes, edges, nodes_by_hop, adj, in_degrees,
    apoc.map.merge(
        h_map,
        apoc.map.fromPairs([
            v IN _active |
                [toString(v.{nid}),
                 [_oi IN range(0, {od} - 1) |
                     $b_{l}[_oi]
                     - vector_distance(
                         vector(h_map[toString(v.{nid})], {id_}, FLOAT32),
                         vector($W_{l}[_oi * {id_} .. (_oi + 1) * {id_}], {id_}, FLOAT32),
                         DOT
                       )
                 ]
                ]
        ])
    ) AS h_map"""


# ---------------------------------------------------------------------------
# 12. Prepared state and query builder for the optimised strategy
# ---------------------------------------------------------------------------

@dataclass
class OptimizedCypherInferencePrepared:
    """Pre-built artefacts for the in_db_cypher_opt strategy.

    Identical shape to CypherInferencePrepared — the only difference is that
    the query uses vector_distance linear blocks instead of reduce loops.
    run_cypher_inference is reused unchanged.
    """
    query: str
    static_params: dict[str, Any]
    database: str


def build_optimized_cypher_inference_query(
    spec: dict,
    weights: dict[str, np.ndarray],
    *,
    node_label: str,
    edge_type: str,
    nodeid_prop: str,
    feature_prop: str,
) -> tuple[str, dict[str, Any]]:
    """Assemble the optimised combined Cypher query.

    Identical walk to ``build_cypher_inference_query`` but linear blocks use
    ``_build_vd_linear_block`` (vector_distance DOT) instead of
    ``_build_linear_block`` (reduce).  Returns ``(query_str, static_params)``.
    """
    fanouts: list = spec["max_neighbors"]
    if isinstance(fanouts, int):
        fanouts = [fanouts] * spec["num_hops"]

    sampler_key: str = spec.get("sampler", "neighbor_uniform")
    if sampler_key not in SAMPLER_REGISTRY:
        raise ValueError(
            f"Unknown sampler '{sampler_key}'. Available: {list(SAMPLER_REGISTRY)}"
        )

    ctx = SamplerCtx(
        node_label=node_label,
        edge_type=edge_type,
        nodeid_prop=nodeid_prop,
        feature_prop=feature_prop,
        fanouts=fanouts,
    )

    blocks: list[str] = []
    static_params: dict[str, Any] = {}

    # ── 1. Sampler block (unchanged) ─────────────────────────────────────────
    blocks.append(SAMPLER_REGISTRY[sampler_key](ctx))

    # ── 2. Init block (unchanged) ────────────────────────────────────────────
    blocks.append(_build_init_block(ctx))

    # ── 3. Layer blocks ───────────────────────────────────────────────────────
    num_hops        = spec["num_hops"]
    active_level    = max(0, num_hops - 1)
    last_agg_level  = max(0, num_hops - 1)
    layer_idx       = 0

    for layer in spec["layers"]:
        op = layer["op"]

        if op == "aggregate":
            method = layer["method"]
            if method not in AGG_REGISTRY:
                raise ValueError(
                    f"Unknown aggregation '{method}'. Available: {list(AGG_REGISTRY)}"
                )
            feat_dim = _next_linear_in_dim(spec["layers"], weights, layer)
            blocks.append(AGG_REGISTRY[method](active_level, feat_dim, nodeid_prop))
            last_agg_level = active_level
            active_level   = max(0, active_level - 1)

        elif op == "linear":
            W = weights[layer["weight"]]
            b = weights[layer["bias"]]
            if W.ndim != 2:
                raise ValueError(
                    f"Weight '{layer['weight']}' shape {W.shape}; expected rank-2."
                )
            out_dim, in_dim = W.shape
            static_params[f"W_{layer_idx}"] = W.flatten().tolist()
            static_params[f"b_{layer_idx}"] = b.flatten().tolist()
            # Use vector_distance block instead of reduce block
            blocks.append(_build_vd_linear_block(
                last_agg_level, layer_idx, int(out_dim), int(in_dim), nodeid_prop
            ))

        elif op == "relu":
            blocks.append(_build_relu_block(last_agg_level, nodeid_prop))

        elif op == "tanh":
            blocks.append(_build_tanh_block(last_agg_level, nodeid_prop))

        else:
            raise ValueError(f"Unknown op '{op}' in spec layers.")

        layer_idx += 1

    # ── 4. Argmax block (unchanged) ──────────────────────────────────────────
    blocks.append(_build_argmax_block())

    # Prepend CYPHER 25 directive so Neo4j parses vector() / vector_distance()
    # type keywords (FLOAT32, DOT) correctly — these are Cypher 25-only tokens.
    return "CYPHER 25\n" + "\n".join(blocks), static_params
