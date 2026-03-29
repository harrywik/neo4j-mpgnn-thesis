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
from typing import List, Optional

import numpy as np
from torch_geometric.data.feature_store import TensorAttr

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS


class Neo4jUDPFeatureStore(Neo4jAbstractFS):
    """Feature store backed by the ``gnnProcedures.aggregation.neighbor.mean`` UDP.

    Parameters
    ----------
    max_neighbors:
        Maximum number of neighbours to aggregate per requested node.
    edge_type:
        Relationship type to aggregate across. Empty string means any type.
    All remaining keyword arguments are forwarded verbatim to
    :class:`Neo4jAbstractFS`.
    """

    def __init__(self, sampler=None, max_neighbors: int = 10, edge_type: str = "", **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.max_neighbors = int(max_neighbors)
        self.edge_type = edge_type or ""

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
        return (
            "CALL gnnProcedures.aggregation.neighbor.mean("
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

    def _decode_feature_matrix(self, records: List[object], field_name: str) -> np.ndarray:
        return np.asarray([rec[field_name] for rec in records], dtype=np.float32)

