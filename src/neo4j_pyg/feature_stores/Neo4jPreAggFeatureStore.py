"""Neo4jPreAggFeatureStore — feature store that returns server-side pre-aggregated features.

Usage
-----
Use a regular topology sampler to produce the sampled node IDs and edge_index,
and let this feature store retrieve pre-aggregated features on demand::

    sampler = Neo4jJavaNeighborSampler(graph_store, ...)
    feature_store = Neo4jPreAggFeatureStore(graph_store, ...)

For the ``x`` attribute, ``_multi_get_tensor`` uses the requested node IDs from
PyG and calls ``gnnProcedures.aggregation.neighbor.sampledGcnNormFetchBatch`` to
return server-side GCN-normalised pre-aggregated features for those nodes.

For paired ``x``/``y`` requests, the same procedure call also returns labels so
the feature store can avoid a second DB round-trip.
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

from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS


class Neo4jPreAggFeatureStore(Neo4jFS):
    """Feature store that retrieves server-side pre-aggregated node features.

    Instead of fetching raw features for every sampled node, this store
    delegates aggregation to Neo4j stored procedures
    (``gnnProcedures.aggregation.neighbor.*``), returning one pre-aggregated
    feature vector per requested node ready for the GNN's linear transform.

    Parameters
    ----------
    improved:
        Use the improved GCN self-loop weight of ``2`` instead of ``1``.
    All remaining keyword arguments are forwarded verbatim to
    :class:`Neo4jFS`.
    """

    def __init__(
        self,
        sampler=None,
        improved: bool = False,
        **kwargs,
    ):
        kwargs.pop("aggregation_mode", None)
        kwargs.pop("max_neighbors", None)
        kwargs.pop("edge_type", None)
        kwargs.pop("use_java_preagg", None)
        super().__init__(**kwargs)
        self.sampler = sampler
        self.improved = bool(improved)

    # ------------------------------------------------------------------
    # Query/decode overrides so the base feature-store flow can be reused.
    # ------------------------------------------------------------------

    @staticmethod
    def _cypher_str(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    @cached_property
    def _query_sampled_gcnnorm_fetch(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        return (
            f"{prefix}CALL gnnProcedures.aggregation.neighbor.sampledGcnNormFetchBatch("
            "$nodes_by_hop,"
            "$edge_pairs,"
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
            "$nodes_by_hop,"
            "$edge_pairs,"
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
        if self.sampler is not None:
            nodes_by_hop = getattr(self.sampler, "last_nodes_by_hop", None) or []
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=nodes_by_hop[-1] if nodes_by_hop else None,
            )
        else:
            nodes_by_hop = []

        node_ids = [int(nid) for nid in nids]
        edge_pairs = self._current_sampled_edge_pairs or []

        feature_map, label_map, preagg_map = self._fetch_sampled_gcnnorm_bundle(
            nodes_by_hop=nodes_by_hop,
            edge_pairs=edge_pairs,
            include_label=True,
        )

        # DB call is done — start timing pure Python assembly.
        self.t_feat_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

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
        y_array = np.asarray([label_map.get(nid, -1) for nid in node_ids], dtype=np.int64)

        for i, nid in enumerate(node_ids):
            feat_matrix[i] = feature_map.get(nid, zero)

        self._last_hybrid_preagg = preagg_map
        return node_ids, feat_matrix, y_array

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:
        if attr.attr_name != "x":
            return super()._get_value_from_db(nids, attr, **kwargs)

        if self.sampler is not None:
            nodes_by_hop = getattr(self.sampler, "last_nodes_by_hop", None) or []
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=nodes_by_hop[-1] if nodes_by_hop else None,
            )
        else:
            nodes_by_hop = []

        frontier_set = {int(nid) for nid in (nodes_by_hop[-1] if nodes_by_hop else [])}
        edge_pairs = self._current_sampled_edge_pairs or []

        node_ids = [int(nid) for nid in nids]
        deepest_nids = [nid for nid in node_ids if nid in frontier_set]

        result_map, _, preagg_map = self._fetch_sampled_gcnnorm_bundle(
            nodes_by_hop=nodes_by_hop,
            edge_pairs=edge_pairs,
            include_label=False,
        )

        # DB call is done — start timing pure Python assembly.
        self.t_feat_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

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

        return result_map

    def _fetch_sampled_gcnnorm_bundle(
        self,
        nodes_by_hop: List[List[int]],
        edge_pairs: List[List[int]],
        include_label: bool,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, np.ndarray]]:
        if not nodes_by_hop:
            return {}, {}, {}

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(
                self._query_sampled_gcnnorm_fetch if include_label else self._query_sampled_gcnnorm_fetch_no_label,
                nodes_by_hop=nodes_by_hop,
                edge_pairs=edge_pairs,
            )
            if self.measurer is not None:
                self.measurer.log_event("start_deserialise", 1)
                self.measurer.set_phase("deserialise")
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()
            if self.measurer is not None:
                self.measurer.log_event("end_deserialise", 1)
                self.measurer.set_phase("etl")
                self.measurer.log_event("start_etl", 1)

        if not records:
            if self.measurer is not None:
                self.measurer.log_event("end_etl", 1)
                self.measurer.set_phase("db_wait")
            return {}, {}, {}

        total_fetch_ms = (t_all_records - t_send) * 1000
        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_fetch_ms)
            self.measurer.log_event("udp_agg_ms", total_fetch_ms)
            self.measurer.log_event("udp_records", len(records))
        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "feat_x", t_send, t_all_records)

        record = records[0]

        raw_ids = [int(nid) for nid in (record["rawNodeIds"] or [])]
        raw_features = record["rawNodeFeatures"]
        feature_map: Dict[int, np.ndarray] = {}
        raw_matrix: Optional[np.ndarray] = None
        if raw_ids:
            raw_matrix = self._decode_packed_feature_rows(raw_features, len(raw_ids))
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
        aggregated_features = record["aggregatedFeatures"]
        agg_matrix: Optional[np.ndarray] = None
        if target_ids_out:
            agg_matrix = self._decode_packed_feature_rows(aggregated_features, len(target_ids_out))
            for i, nid in enumerate(target_ids_out):
                preagg_map[nid] = agg_matrix[i]

        if self.measurer is not None:
            raw_bytes = int(raw_matrix.nbytes) if raw_matrix is not None else 0
            agg_bytes = int(agg_matrix.nbytes) if agg_matrix is not None else 0
            self.measurer.log_event("feat_bytes", raw_bytes + agg_bytes)
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("db_wait")

        return feature_map, label_map, preagg_map

    def _decode_packed_feature_rows(self, flat_bytes, n: int) -> np.ndarray:
        if not flat_bytes or n == 0:
            feature_dim = self._feature_dim or 0
            return np.empty((0, feature_dim), dtype=np.float32)
        # New format: single flat byte[] from Java (one Bolt bytes value).
        if isinstance(flat_bytes, (bytes, bytearray, memoryview)):
            return np.frombuffer(memoryview(flat_bytes), dtype=np.float32).reshape(n, -1)
        # Fallback: old List<byte[]> format (plugin not yet rebuilt).
        return np.stack([np.frombuffer(memoryview(row), dtype=np.float32) for row in flat_bytes])


