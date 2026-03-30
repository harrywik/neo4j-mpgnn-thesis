"""Neo4jUDPFeatureStore — feature store that serves UDP-aggregated features.

Usage
-----
Use a regular topology sampler to produce the sampled node IDs and edge_index,
and let this feature store aggregate the requested node features on demand::

    sampler = Neo4jJavaNeighborSampler(graph_store, ...)
    feature_store = Neo4jUDPFeatureStore(graph_store, max_neighbors=10, ...)

For the ``x`` attribute, ``_multi_get_tensor`` uses the requested node IDs from
PyG and calls ``gnnProcedures.aggregation.neighbor.mean`` to return one-hop
aggregated features for those nodes.

For paired ``x``/``y`` requests, the same UDP call also returns labels so the
feature store can avoid a second DB round-trip.
"""

from functools import cached_property
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from torch_geometric.data.feature_store import TensorAttr

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS


class Neo4jUDPFeatureStore(Neo4jAbstractFS):
    """Feature store backed by the ``gnnProcedures.aggregation.neighbor.*`` UDPs.

    Parameters
    ----------
    max_neighbors:
        Maximum number of neighbours to aggregate per requested node.
    edge_type:
        Relationship type to aggregate across. Empty string means any type.
    aggregation_mode:
        Server-side aggregation to use. Supported values are ``"mean"`` and
        ``"gcnNorm"`` and ``"sampledGcnNorm"``.
    improved:
        When ``aggregation_mode="gcnNorm"``, use the improved GCN self-loop
        weight of ``2`` instead of ``1``.
    All remaining keyword arguments are forwarded verbatim to
    :class:`Neo4jAbstractFS`.
    """

    def __init__(
        self,
        sampler=None,
        graph_store=None,
        max_neighbors: int = 10,
        edge_type: str = "",
        aggregation_mode: str = "mean",
        improved: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.graph_store = graph_store
        self.max_neighbors = int(max_neighbors)
        self.edge_type = edge_type or ""
        self.aggregation_mode = aggregation_mode or "mean"
        self.improved = bool(improved)

    # ------------------------------------------------------------------
    # Neo4jAbstractFS abstract methods (no-op cache)
    # ------------------------------------------------------------------

    def _get_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> Optional[object]:
        return None

    def _update_cached_value(self, nid: int, value: object, attr: TensorAttr, **kwargs) -> None:
        pass

    def _remove_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Query/decode overrides so the base feature-store flow can be reused.
    # ------------------------------------------------------------------

    @staticmethod
    def _cypher_str(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    def _udp_call(self, *, return_label: bool) -> str:
        node_label = self.node_label or ""
        edge_type = self.edge_type or ""
        target_key = self.target_property if return_label else ""
        if self.aggregation_mode == "mean":
            procedure = "gnnProcedures.aggregation.neighbor.mean"
            improved_arg = ""
        elif self.aggregation_mode == "gcnNorm":
            procedure = "gnnProcedures.aggregation.neighbor.gcnNorm"
            improved_arg = f",{'true' if self.improved else 'false'}"
        elif self.aggregation_mode == "sampledGcnNorm":
            raise ValueError(
                "sampledGcnNorm uses the current sampled subgraph and does not "
                "build a plain neighborhood aggregation query via _udp_call()."
            )
        else:
            raise ValueError(
                "Unsupported Neo4jUDPFeatureStore aggregation_mode "
                f"'{self.aggregation_mode}'. Expected 'mean', 'gcnNorm', or 'sampledGcnNorm'."
            )
        return (
            f"CALL {procedure}("
            "$node_ids,"
            f"{self._cypher_str(self.nodeid_property)},"
            f"{self._cypher_str(self.feature_property)},"
            f"{self._cypher_str(self.feature_property_type)},"
            f"{self._cypher_str(node_label)},"
            f"{self._cypher_str(edge_type)},"
            f"{self.max_neighbors},"
            f"{self._cypher_str(target_key)},"
            "false,"
            f"{'true' if return_label else 'false'}"
            f"{improved_arg}"
            ")"
        )

    @cached_property
    def _query_both(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}{self._udp_call(return_label=True)} "
            "YIELD nodeId, aggregatedFeatures, label "
            "RETURN nodeId AS id, aggregatedFeatures AS feature, label"
        )

    @cached_property
    def _query_x(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}{self._udp_call(return_label=False)} "
            "YIELD nodeId, aggregatedFeatures "
            "RETURN nodeId AS id, aggregatedFeatures AS value"
        )

    @cached_property
    def _query_x_raw(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}UNWIND $node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"RETURN nid AS id, n.{self.feature_property} AS value"
        )

    @cached_property
    def _query_sampled_gcnnorm(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}CALL gnnProcedures.aggregation.neighbor.sampledGcnNorm("
            "$target_ids,"
            "$edge_pairs,"
            f"{self._cypher_str(self.nodeid_property)},"
            f"{self._cypher_str(self.feature_property)},"
            f"{self._cypher_str(self.feature_property_type)},"
            f"{self._cypher_str(self.node_label or '')},"
            f"{'true' if self.improved else 'false'}"
            ") "
            "YIELD nodeId, aggregatedFeatures "
            "RETURN nodeId AS id, aggregatedFeatures AS value"
        )

    def _multi_get_tensor(self, attrs: List[TensorAttr]):
        if self.aggregation_mode == "sampledGcnNorm":
            return [self._get_tensor(attr) for attr in attrs]
        return super()._multi_get_tensor(attrs)

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:
        if self.aggregation_mode != "sampledGcnNorm" or attr.attr_name != "x":
            return super()._get_value_from_db(nids, attr, **kwargs)

        if self.graph_store is None:
            raise ValueError("Neo4jUDPFeatureStore sampledGcnNorm mode requires graph_store")

        subgraph = self.graph_store.get_last_sampled_subgraph()
        hop_depths = getattr(self.graph_store, "last_sampled_hop_depths", {})
        max_depth = getattr(self.graph_store, "last_sampled_max_depth", 0)
        if subgraph is None or max_depth <= 0:
            return self._get_raw_features(nids)

        raw_nids = [nid for nid in nids if hop_depths.get(int(nid), 0) < max_depth]
        deepest_nids = [nid for nid in nids if hop_depths.get(int(nid), 0) == max_depth]
        target_nids = [nid for nid in nids if hop_depths.get(int(nid), 0) == max_depth - 1]

        result_map = self._get_raw_features(raw_nids)
        preagg_map = self._fetch_sampled_gcnnorm(target_nids, subgraph.get("edge_pairs") or [])
        self.graph_store.last_hybrid_preagg = preagg_map

        feature_dim = None
        if result_map:
            feature_dim = next(iter(result_map.values())).shape[0]
        elif preagg_map:
            feature_dim = next(iter(preagg_map.values())).shape[0]
        elif self._feature_dim is not None:
            feature_dim = self._feature_dim

        if feature_dim is None:
            raise ValueError("Could not infer feature dimension for sampledGcnNorm feature assembly")

        self._feature_dim = feature_dim
        zero = np.zeros(feature_dim, dtype=np.float32)
        for nid in deepest_nids:
            result_map[int(nid)] = zero.copy()

        return result_map

    def _get_raw_features(self, nids: List[int]) -> Dict[int, np.ndarray]:
        if not nids:
            return {}

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(self._query_x_raw, node_ids=nids)
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        total_feat_fetch_ms = (t_all_records - t_send) * 1000
        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_feat_fetch_ms)
            self._log_extra_feature_fetch_metrics(records, total_feat_fetch_ms, is_label=False, paired=False)
        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "feat_x", t_send, t_all_records)

        result_map: Dict[int, np.ndarray] = {}
        if not records:
            return result_map

        feat_matrix = self._decode_feature_matrix(records, "value")
        for i, rec in enumerate(records):
            result_map[int(rec["id"])] = feat_matrix[i]
        return result_map

    def _fetch_sampled_gcnnorm(self, target_nids: List[int], edge_pairs: List[List[int]]) -> Dict[int, np.ndarray]:
        if not target_nids:
            return {}

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            result = session.run(
                self._query_sampled_gcnnorm,
                target_ids=target_nids,
                edge_pairs=edge_pairs,
            )
            records = list(result)

        if not records:
            return {}

        feat_matrix = np.asarray([rec["value"] for rec in records], dtype=np.float32)
        return {int(rec["id"]): feat_matrix[i] for i, rec in enumerate(records)}

    def _decode_feature_matrix(self, records: List[object], field_name: str) -> np.ndarray:
        return np.asarray([rec[field_name] for rec in records], dtype=np.float32)

    def _log_extra_feature_fetch_metrics(
        self,
        records: List[object],
        total_fetch_ms: float,
        *,
        is_label: bool,
        paired: bool,
    ) -> None:
        if self.measurer is None or is_label:
            return
        self.measurer.log_event("udp_agg_ms", total_fetch_ms)
        self.measurer.log_event("udp_records", len(records))
