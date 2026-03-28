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

For the ``y`` attribute, the standard DB round-trip is used (labels are tiny
compared with feature vectors so fetching them separately is insignificant).
"""

from pathlib import Path
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

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
    # Core override: serve x from UDP, y from DB
    # ------------------------------------------------------------------

    def _multi_get_tensor(self, attrs: List[TensorAttr]) -> List[Optional[FeatureTensorType]]:
        """Serve UDP-aggregated ``x`` and fetch ``y`` from DB."""
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        if x_attr is None or y_attr is None:
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()
        agg_map = self._get_aggregated_x_from_db(
            node_ids,
            max_neighbors=self.max_neighbors,
            edge_type=self.edge_type,
        )
        label_map: Dict[int, int] = self._get_value_from_db(node_ids, y_attr)

        self.t_feat_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

        x_rows: List[np.ndarray] = []
        feat_dim: Optional[int] = self._feature_dim
        for nid in node_ids:
            vec = agg_map.get(nid)
            if vec is not None:
                if feat_dim is None:
                    feat_dim = vec.shape[0]
                    self._feature_dim = feat_dim
                x_rows.append(vec)
            else:
                x_rows.append(np.zeros(feat_dim or 1, dtype=np.float32))

        x_tensor = torch.from_numpy(np.stack(x_rows))
        y_array = np.array([label_map.get(nid, 0) for nid in node_ids], dtype=np.int64)
        y_tensor = torch.from_numpy(y_array)

        # Return in the same order as `attrs` was received.
        result: List[Optional[FeatureTensorType]] = []
        for attr in attrs:
            if attr.attr_name == "x":
                result.append(x_tensor)
            elif attr.attr_name == "y":
                result.append(y_tensor)
            else:
                result.append(self._get_tensor(attr))

        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")
            self.measurer.log_event("feat_x_etl_ms", (time.monotonic() - self.t_feat_etl_start) * 1000)
        return result

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        if attr.attr_name != "x":
            return super()._get_tensor(attr)

        node_ids: List[int] = attr.index.tolist()
        agg_map = self._get_aggregated_x_from_db(
            node_ids,
            max_neighbors=self.max_neighbors,
            edge_type=self.edge_type,
        )

        self.t_feat_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

        feat_dim: Optional[int] = self._feature_dim
        rows: List[np.ndarray] = []
        for nid in node_ids:
            vec = agg_map.get(nid)
            if vec is not None:
                if feat_dim is None:
                    feat_dim = vec.shape[0]
                    self._feature_dim = feat_dim
                rows.append(vec)
            else:
                rows.append(np.zeros(feat_dim or 1, dtype=np.float32))

        x_tensor = torch.from_numpy(np.stack(rows))
        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")
            self.measurer.log_event("feat_x_etl_ms", (time.monotonic() - self.t_feat_etl_start) * 1000)
        return x_tensor
