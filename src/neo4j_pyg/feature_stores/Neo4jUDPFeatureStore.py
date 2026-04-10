"""Neo4jUDPFeatureStore — feature store that reads pre-aggregated vectors
from a companion Neo4jAggregationSampler instead of hitting Neo4j a second time.

Usage
-----
Wire the sampler and the feature store together by passing the same sampler
instance to both the ``NodeLoader`` and this class::

    sampler = Neo4jAggregationSampler(graph_store, ...)
    feature_store = Neo4jUDPFeatureStore(graph_store, sampler=sampler, ...)

For the ``x`` attribute, ``_multi_get_tensor`` reads directly from
``sampler.pending_agg`` — a ``{nodeId: np.ndarray}`` dict that the sampler
populates on every call to ``sample_from_nodes``.

For the ``y`` attribute, the standard DB round-trip is used (labels are tiny
compared with feature vectors so fetching them separately is insignificant).
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS


class Neo4jUDPFeatureStore(Neo4jFS):
    """Feature store backed by the ``custom.gcn.aggregateNeighbors`` UDP.

    Parameters
    ----------
    sampler:
        The ``Neo4jAggregationSampler`` instance whose ``pending_agg`` dict
        will be consumed for each mini-batch.
    All remaining keyword arguments are forwarded verbatim to
    :class:`Neo4jFS`.
    """

    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler

    # ------------------------------------------------------------------
    # Cache overrides (no-op)
    # ------------------------------------------------------------------

    def _update_cached_value(self, nid: int, value: object, attr: TensorAttr, **kwargs) -> None:
        pass

    def _remove_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> None:
        pass

    # ------------------------------------------------------------------
    # Core override: serve x from UDP, y from DB
    # ------------------------------------------------------------------

    def _multi_get_tensor(self, attrs: List[TensorAttr]) -> List[Optional[FeatureTensorType]]:
        """Serve ``x`` from the sampler's pre-aggregated cache; fetch ``y`` from DB."""
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        if x_attr is None or y_attr is None:
            # Unexpected attr combination — fall back to the parent implementation.
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()
        pending = self.sampler.pending_agg

        # Build x tensor from the UDP results buffered by the sampler.
        x_rows: List[np.ndarray] = []
        feat_dim: Optional[int] = self._feature_dim
        for nid in node_ids:
            vec = pending.get(nid)
            if vec is not None:
                if feat_dim is None:
                    feat_dim = vec.shape[0]
                    self._feature_dim = feat_dim
                x_rows.append(vec)
            else:
                # Node missing from UDP result — fill with zeros so downstream
                # code doesn't crash.  This should not happen in practice.
                x_rows.append(np.zeros(feat_dim or 1, dtype=np.float32))

        x_tensor = torch.from_numpy(np.stack(x_rows))

        # Fetch labels (y) via the normal DB path.
        label_map: Dict[int, int] = self._get_value_from_db(node_ids, y_attr)
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
        return result
