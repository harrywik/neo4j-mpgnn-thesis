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
        max_neighbors: int = 10,
        edge_type: str = "",
        aggregation_mode: str = "mean",
        improved: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.max_neighbors = int(max_neighbors)
        self.edge_type = edge_type or ""
        self.aggregation_mode = aggregation_mode or "mean"
        self.improved = bool(improved)
        self._last_hybrid_preagg: Dict[int, np.ndarray] = {}
        self._last_hybrid_targets: set[int] = set()
        if self.aggregation_mode == "sampledGcnNorm" and self.sampler is not None:
            self.sampler.return_frontier = True

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
    def _query_sampled_gcnnorm_fetch(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}CALL gnnProcedures.aggregation.neighbor.sampledGcnNormFetchBatch("
            "$node_ids,"
            "$raw_node_ids,"
            "$target_ids,"
            "$edge_pairs,"
            "$frontier_ids,"
            f"{self._cypher_str(self.nodeid_property)},"
            f"{self._cypher_str(self.feature_property)},"
            f"{self._cypher_str(self.feature_property_type)},"
            f"{self._cypher_str(self.node_label or '')},"
            f"{self._cypher_str(self.target_property)},"
            "true,"
            f"{'true' if self.improved else 'false'}"
            ") "
            "YIELD rawNodeIds, rawNodeFeatures, labelNodeIds, labels, targetNodeIds, aggregatedFeatures "
            "RETURN rawNodeIds, rawNodeFeatures, labelNodeIds, labels, targetNodeIds, aggregatedFeatures"
        )

    @cached_property
    def _query_sampled_gcnnorm_fetch_no_label(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}CALL gnnProcedures.aggregation.neighbor.sampledGcnNormFetchBatch("
            "$node_ids,"
            "$raw_node_ids,"
            "$target_ids,"
            "$edge_pairs,"
            "$frontier_ids,"
            f"{self._cypher_str(self.nodeid_property)},"
            f"{self._cypher_str(self.feature_property)},"
            f"{self._cypher_str(self.feature_property_type)},"
            f"{self._cypher_str(self.node_label or '')},"
            f"{self._cypher_str(self.target_property)},"
            "false,"
            f"{'true' if self.improved else 'false'}"
            ") "
            "YIELD rawNodeIds, rawNodeFeatures, labelNodeIds, labels, targetNodeIds, aggregatedFeatures "
            "RETURN rawNodeIds, rawNodeFeatures, labelNodeIds, labels, targetNodeIds, aggregatedFeatures"
        )

    def _get_both_from_db(self, nids: List[int], x_attr: TensorAttr):
        if self.aggregation_mode != "sampledGcnNorm":
            return super()._get_both_from_db(nids, x_attr)

        if self.sampler is not None:
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=getattr(self.sampler, "last_frontier_nodes", None),
            )

        node_ids = [int(nid) for nid in nids]
        frontier_set = {int(nid) for nid in (self._current_frontier_nodes or [])}
        edge_pairs = self._current_sampled_edge_pairs or []
        target_nid_set = {
            int(pair[1])
            for pair in edge_pairs
            if pair is not None and len(pair) >= 2 and int(pair[0]) in frontier_set
        }
        raw_nids = [nid for nid in node_ids if nid not in frontier_set]
        target_nids = [nid for nid in node_ids if nid in target_nid_set]

        feature_map, label_map, preagg_map = self._fetch_sampled_gcnnorm_bundle(
            node_ids=node_ids,
            raw_node_ids=raw_nids if frontier_set else node_ids,
            target_ids=target_nids,
            edge_pairs=edge_pairs,
            frontier_ids=list(frontier_set),
            include_label=True,
        )
        if not feature_map and not node_ids:
            empty_feats = np.empty((0, self._feature_dim or 0), dtype=np.float32)
            return [], empty_feats, np.empty(0, dtype=np.int64)

        feature_dim = None
        if feature_map:
            feature_dim = next(iter(feature_map.values())).shape[0]
        elif preagg_map:
            feature_dim = next(iter(preagg_map.values())).shape[0]
        elif self._feature_dim is not None:
            feature_dim = self._feature_dim
        if feature_dim is None:
            raise ValueError("Could not infer feature dimension for sampledGcnNorm feature assembly")

        self._feature_dim = feature_dim
        zero = np.zeros(feature_dim, dtype=np.float32)
        feat_matrix = np.empty((len(node_ids), feature_dim), dtype=np.float32)
        y_array = np.asarray([label_map[nid] for nid in node_ids], dtype=np.int64)

        for i, nid in enumerate(node_ids):
            feat_matrix[i] = feature_map.get(nid, zero)

        self._last_hybrid_preagg = preagg_map
        self._last_hybrid_targets = target_nid_set
        return node_ids, feat_matrix, y_array

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:
        if self.aggregation_mode != "sampledGcnNorm" or attr.attr_name != "x":
            return super()._get_value_from_db(nids, attr, **kwargs)

        if self.sampler is not None:
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=getattr(self.sampler, "last_frontier_nodes", None),
            )

        frontier_set = {int(nid) for nid in (self._current_frontier_nodes or [])}
        edge_pairs = self._current_sampled_edge_pairs or []
        target_nid_set = {
            int(pair[1])
            for pair in edge_pairs
            if pair is not None and len(pair) >= 2 and int(pair[0]) in frontier_set
        }

        node_ids = [int(nid) for nid in nids]
        raw_nids = [nid for nid in node_ids if nid not in frontier_set]
        deepest_nids = [nid for nid in node_ids if nid in frontier_set]
        target_nids = [nid for nid in node_ids if nid in target_nid_set]

        result_map, _, preagg_map = self._fetch_sampled_gcnnorm_bundle(
            node_ids=node_ids,
            raw_node_ids=raw_nids if frontier_set else node_ids,
            target_ids=target_nids,
            edge_pairs=edge_pairs,
            frontier_ids=list(frontier_set),
            include_label=False,
        )

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

        self._last_hybrid_preagg = preagg_map
        self._last_hybrid_targets = target_nid_set

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

    def _fetch_sampled_gcnnorm_bundle(
        self,
        node_ids: List[int],
        raw_node_ids: List[int],
        target_ids: List[int],
        edge_pairs: List[List[int]],
        frontier_ids: List[int],
        include_label: bool,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, np.ndarray]]:
        if not node_ids:
            return {}, {}, {}

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(
                self._query_sampled_gcnnorm_fetch if include_label else self._query_sampled_gcnnorm_fetch_no_label,
                node_ids=node_ids,
                raw_node_ids=raw_node_ids,
                target_ids=target_ids,
                edge_pairs=edge_pairs,
                frontier_ids=frontier_ids,
            )
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        if not records:
            return {}, {}, {}

        total_fetch_ms = (t_all_records - t_send) * 1000
        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_fetch_ms)
            self._log_extra_feature_fetch_metrics(records, total_fetch_ms, is_label=False, paired=include_label)
        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "feat_x", t_send, t_all_records)

        record = records[0]

        raw_ids = [int(nid) for nid in (record["rawNodeIds"] or [])]
        raw_features = record["rawNodeFeatures"] or []
        feature_map: Dict[int, np.ndarray] = {}
        if raw_ids:
            raw_matrix = self._decode_packed_feature_rows(raw_features)
            for i, nid in enumerate(raw_ids):
                feature_map[nid] = raw_matrix[i]

        label_map: Dict[int, int] = {}
        if include_label:
            label_ids = [int(nid) for nid in (record["labelNodeIds"] or [])]
            labels = record["labels"] or []
            for nid, label in zip(label_ids, labels):
                if isinstance(label, str):
                    if label not in self._labels:
                        self._labels[label] = len(self._labels)
                    label_map[nid] = self._labels[label]
                elif label is not None:
                    label_map[nid] = int(label)

        preagg_map: Dict[int, np.ndarray] = {}
        target_ids_out = [int(nid) for nid in (record["targetNodeIds"] or [])]
        aggregated_features = record["aggregatedFeatures"] or []
        if target_ids_out:
            agg_matrix = self._decode_packed_feature_rows(aggregated_features)
            for i, nid in enumerate(target_ids_out):
                preagg_map[nid] = agg_matrix[i]

        return feature_map, label_map, preagg_map

    def _decode_packed_feature_rows(self, rows: List[object]) -> np.ndarray:
        if not rows:
            feature_dim = self._feature_dim or 0
            return np.empty((0, feature_dim), dtype=np.float32)

        first_row = rows[0]
        if isinstance(first_row, (bytes, bytearray, memoryview)):
            return np.stack([
                np.frombuffer(memoryview(row), dtype=np.float32)
                for row in rows
            ])

        return np.asarray(rows, dtype=np.float32)

    def _decode_feature_matrix(self, records: List[object], field_name: str) -> np.ndarray:
        first_value = records[0][field_name]
        if isinstance(first_value, (bytes, bytearray, memoryview)):
            return np.stack([
                np.frombuffer(memoryview(rec[field_name]), dtype=np.float32)
                for rec in records
            ])
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
