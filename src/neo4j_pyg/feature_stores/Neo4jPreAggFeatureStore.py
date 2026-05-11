"""Neo4jPreAggFeatureStore — feature store that returns server-side pre-aggregated features.

Usage
-----
Use a regular topology sampler to produce the sampled node IDs and edge_index,
and let this feature store retrieve pre-aggregated features on demand::

    sampler = Neo4jJavaNeighborSampler(graph_store, ...)
    feature_store = Neo4jPreAggFeatureStore(graph_store, max_neighbors=10, ...)

For the ``x`` attribute, ``_multi_get_tensor`` uses the requested node IDs from
PyG and calls a ``gnnProcedures.aggregation.neighbor.*`` stored procedure to
return server-side pre-aggregated features for those nodes.

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
    max_neighbors:
        Maximum number of neighbours to sample per requested node during
        server-side aggregation.
    edge_type:
        Relationship type to aggregate across. Empty string means any type.
    aggregation_mode:
        Server-side aggregation to use. Supported values are ``"mean"``,
        ``"gcnNorm"``, and ``"sampledGcnNorm"``.
    improved:
        When ``aggregation_mode="gcnNorm"``, use the improved GCN self-loop
        weight of ``2`` instead of ``1``.
    All remaining keyword arguments are forwarded verbatim to
    :class:`Neo4jFS`.
    """

    def __init__(
        self,
        sampler=None,
        max_neighbors: int = 10,
        edge_type: str = "",
        aggregation_mode: str = "mean",
        improved: bool = False,
        use_java_preagg: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.max_neighbors = int(max_neighbors)
        self.edge_type = edge_type or ""
        self.aggregation_mode = aggregation_mode or "mean"
        self.improved = bool(improved)
        self.use_java_preagg = bool(use_java_preagg)
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
                "Unsupported Neo4jPreAggFeatureStore aggregation_mode "
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

    @cached_property
    def _query_sampled_gcnnorm_fetch_cypher(self) -> str:
        """Pure-Cypher equivalent of sampledGcnNormFetchBatch (with label).

        Fetches raw features for ``$raw_node_ids`` and labels for ``$node_ids``
        using UNWIND/MATCH.  ``targetNodeIds`` and ``aggregatedFeatures`` are
        always empty — GCN-norm aggregation is not performed server-side.
        """
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}"
            f"UNWIND $raw_node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"WITH collect(nid) AS rawNodeIds, collect(n.{self.feature_property}) AS rawNodeFeatures "
            f"UNWIND $node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"WITH rawNodeIds, rawNodeFeatures, "
            f"collect(nid) AS labelNodeIds, collect(n.{self.target_property}) AS labels "
            f"RETURN rawNodeIds, rawNodeFeatures, labelNodeIds, labels, "
            f"[] AS targetNodeIds, [] AS aggregatedFeatures"
        )

    @cached_property
    def _query_sampled_gcnnorm_fetch_cypher_no_label(self) -> str:
        """Pure-Cypher equivalent of sampledGcnNormFetchBatch (without label).

        Fetches raw features for ``$raw_node_ids`` using UNWIND/MATCH.
        ``labelNodeIds``, ``labels``, ``targetNodeIds``, and ``aggregatedFeatures``
        are always empty.
        """
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}"
            f"UNWIND $raw_node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"WITH collect(nid) AS rawNodeIds, collect(n.{self.feature_property}) AS rawNodeFeatures "
            f"RETURN rawNodeIds, rawNodeFeatures, "
            f"[] AS labelNodeIds, [] AS labels, [] AS targetNodeIds, [] AS aggregatedFeatures"
        )

    def _get_both_from_db(self, nids: List[int], x_attr: TensorAttr):
        if self.aggregation_mode != "sampledGcnNorm":
            return super()._get_both_from_db(nids, x_attr)

        if self.sampler is not None:
            nodes_by_hop = getattr(self.sampler, "last_nodes_by_hop", None) or []
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=None,
            )
        else:
            nodes_by_hop = []

        node_ids = [int(nid) for nid in nids]
        edge_pairs = self._current_sampled_edge_pairs or []
        target_nid_set = {int(nid) for nid in (nodes_by_hop[-2] if len(nodes_by_hop) >= 2 else [])}

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
        self._last_hybrid_targets = target_nid_set
        return node_ids, feat_matrix, y_array

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:
        if self.aggregation_mode != "sampledGcnNorm" or attr.attr_name != "x":
            return super()._get_value_from_db(nids, attr, **kwargs)

        if self.sampler is not None:
            nodes_by_hop = getattr(self.sampler, "last_nodes_by_hop", None) or []
            self.set_sampled_subgraph_context(
                sampled_nodes=getattr(self.sampler, "last_sampled_nodes", None),
                edge_pairs=getattr(self.sampler, "last_sampled_edge_pairs", None),
                frontier_nodes=None,
            )
        else:
            nodes_by_hop = []

        frontier_set = {int(nid) for nid in (nodes_by_hop[-1] if nodes_by_hop else [])}
        target_nid_set = {int(nid) for nid in (nodes_by_hop[-2] if len(nodes_by_hop) >= 2 else [])}
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
        self._last_hybrid_targets = target_nid_set

        return result_map

    def _fetch_sampled_gcnnorm_bundle(
        self,
        nodes_by_hop: List[List[int]],
        edge_pairs: List[List[int]],
        include_label: bool,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, np.ndarray]]:
        if not nodes_by_hop:
            return {}, {}, {}
        if not self.use_java_preagg:
            return self._fetch_sampled_gcnnorm_bundle_cypher(nodes_by_hop, include_label)

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

    def _fetch_sampled_gcnnorm_bundle_cypher(
        self,
        nodes_by_hop: List[List[int]],
        include_label: bool,
    ) -> tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, np.ndarray]]:
        """Cypher fallback: fetch raw features for all nodes in nodes_by_hop.

        No server-side aggregation is performed; preagg_map is always empty.
        PyG's GCNConv uses the sampled edge_index to aggregate locally.
        """
        all_nodes = list({int(nid) for hop in nodes_by_hop for nid in hop})
        if not all_nodes:
            return {}, {}, {}

        lf = f":{self.node_label}" if self.node_label else ""
        prefix = "PROFILE " if self.profile else ""
        if include_label:
            query = (
                f"{prefix}UNWIND $node_ids AS nid "
                f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
                f"RETURN nid AS id, n.{self.feature_property} AS feature, "
                f"n.{self.target_property} AS label"
            )
        else:
            query = (
                f"{prefix}UNWIND $node_ids AS nid "
                f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
                f"RETURN nid AS id, n.{self.feature_property} AS feature"
            )

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(query, node_ids=all_nodes)
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

        feat_matrix = self._decode_feature_matrix(records, "feature")
        if self.measurer is not None:
            self.measurer.log_event("feat_bytes", feat_matrix.nbytes)

        feature_map: Dict[int, np.ndarray] = {}
        label_map: Dict[int, int] = {}
        for i, rec in enumerate(records):
            nid = int(rec["id"])
            feature_map[nid] = feat_matrix[i]
            if include_label:
                lbl = rec["label"]
                if isinstance(lbl, str):
                    if lbl not in self._labels:
                        self._labels[lbl] = len(self._labels)
                    label_map[nid] = self._labels[lbl]
                elif lbl is not None:
                    label_map[nid] = int(lbl)

        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("db_wait")

        return feature_map, label_map, {}  # empty preagg_map — aggregation done by PyG

    def _decode_packed_feature_rows(self, flat_bytes, n: int) -> np.ndarray:
        if not flat_bytes or n == 0:
            feature_dim = self._feature_dim or 0
            return np.empty((0, feature_dim), dtype=np.float32)
        # New format: single flat byte[] from Java (one Bolt bytes value).
        if isinstance(flat_bytes, (bytes, bytearray, memoryview)):
            return np.frombuffer(memoryview(flat_bytes), dtype=np.float32).reshape(n, -1)
        # Fallback: old List<byte[]> format (plugin not yet rebuilt).
        return np.stack([np.frombuffer(memoryview(row), dtype=np.float32) for row in flat_bytes])

    def _decode_feature_matrix(self, records: List[object], field_name: str) -> np.ndarray:
        first_value = records[0][field_name]
        if isinstance(first_value, (bytes, bytearray, memoryview)):
            return np.stack([
                np.frombuffer(memoryview(rec[field_name]), dtype=np.float32)
                for rec in records
            ])
        return np.asarray([rec[field_name] for rec in records], dtype=np.float32)


