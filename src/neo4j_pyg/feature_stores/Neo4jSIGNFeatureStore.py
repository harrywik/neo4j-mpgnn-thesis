"""Neo4jSIGNFeatureStore — feature store that reads SIGN pre-aggregated hop
vectors from a companion Neo4jSIGNSampler and concatenates them.

For each seed node the sampler stores:
    pending_sign[nodeId] = [array_hop0, array_hop1, ..., array_hopK]

This class concatenates those arrays to build the SIGN input tensor:
    x = [x_0 || x_1 || ... || x_k]   shape: (N, feature_dim * (hops + 1))

Labels (y) are fetched from Neo4j via the normal DB path.
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

from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS


class Neo4jSIGNFeatureStore(Neo4jAbstractFS):
    """Feature store backed by the ``custom.gcn.signAggregate`` UDP.

    Parameters
    ----------
    sampler:
        The ``Neo4jSIGNSampler`` whose ``pending_sign`` dict will be consumed.
    hops:
        Must match the ``hops`` value used in the sampler.
    All remaining keyword arguments are forwarded to :class:`Neo4jAbstractFS`.
    """

    def __init__(self, sampler, hops: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.sampler = sampler
        self.hops = hops

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
    # Core override: concatenate hop aggregations for x; fetch y from DB
    # ------------------------------------------------------------------

    def _multi_get_tensor(self, attrs: List[TensorAttr]) -> List[Optional[FeatureTensorType]]:
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        if x_attr is None or y_attr is None:
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()
        pending = self.sampler.pending_sign

        # Build concatenated SIGN input for each seed.
        x_rows: List[np.ndarray] = []
        concat_dim: Optional[int] = self._feature_dim  # total dim after concat
        for nid in node_ids:
            hop_arrays = pending.get(nid)
            if hop_arrays is not None and len(hop_arrays) == self.hops + 1:
                vec = np.concatenate(hop_arrays)
                if concat_dim is None:
                    concat_dim = vec.shape[0]
                    self._feature_dim = concat_dim
                x_rows.append(vec)
            else:
                # Missing — fill with zeros.
                x_rows.append(np.zeros(concat_dim or 1, dtype=np.float32))

        x_tensor = torch.from_numpy(np.stack(x_rows))

        # Fetch labels via normal DB path.
        label_map: Dict[int, int] = self._get_value_from_db(node_ids, y_attr)
        y_array = np.array([label_map.get(nid, 0) for nid in node_ids], dtype=np.int64)
        y_tensor = torch.from_numpy(y_array)

        result: List[Optional[FeatureTensorType]] = []
        for attr in attrs:
            if attr.attr_name == "x":
                result.append(x_tensor)
            elif attr.attr_name == "y":
                result.append(y_tensor)
            else:
                result.append(self._get_tensor(attr))
        return result
