"""GPU-resident static feature cache backed by Neo4j.

Implements the PaGraph-style caching strategy:
- Nodes are ranked by out-degree (no GDS required).
- Top-K node features are loaded once into a contiguous GPU tensor.
- At batch time, a vectorized split separates cached from uncached nodes,
  avoiding per-node Python-dict lookups on the hot path.

Cache size is either:
- Fixed via ``cache_size_GB``  (default, no CUDA required), or
- Auto-sized from available GPU memory when ``auto_size=True``.

References
----------
PaGraph: Scaling GNN Training on Large Graphs via Computation-aware Caching
Lin et al., SoCC 2020.
"""

from __future__ import annotations

import atexit
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neo4j import Driver

from neo4j_pyg.feature_caches.Neo4jAbstractCache import Neo4jAbstractCache

# Bytes reserved for model/activations so we don't OOM during training.
_RESERVED_GPU_BYTES = 1 * (1024 ** 3)  # 1 GB


class Neo4jGPUCache(Neo4jAbstractCache):
    """Static GPU-resident feature cache ranked by node out-degree.

    Parameters
    ----------
    device:
        PyTorch device string (e.g. ``"cuda:0"``).  Falls back to ``"cpu"``
        when CUDA is unavailable.
    auto_size:
        If ``True``, cache capacity is derived from free GPU memory after a
        1 GB safety margin, ignoring ``cache_size_GB``.
    cache_size_GB:
        Explicit cache budget in gigabytes (used when ``auto_size=False``).
    reserved_gb:
        GPU memory reserved for model/activations (only used when
        ``auto_size=True``).  Default: 1 GB.
    """

    def __init__(
        self,
        driver: Optional[Driver] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        database_name: Optional[str] = None,
        nodeid_property: str = "nodeId",
        feature_property: str = "features",
        target_property: str = "category",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict[str, int]] = None,
        cache_size_GB: float = 0.5,
        device: str = "cuda",
        auto_size: bool = False,
        reserved_gb: float = 1.0,
        prefill: bool = True,
    ) -> None:
        self._cache_size_GB = cache_size_GB
        self._auto_size = auto_size
        self._reserved_bytes = int(reserved_gb * (1024 ** 3))

        # Resolve device before calling super().__init__ so compute_cache_size
        # can access it. Fall back to CPU if CUDA not available.
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        super().__init__(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            database_name=database_name,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            feature_property_type=feature_property_type,
            label_map=label_map,
            cache_size_GB=cache_size_GB,
        )

        # GPU tensors — populated in prefill_hot_cache.
        self.cached_features: Optional[torch.Tensor] = None  # [N, feat_dim]
        self.cached_labels: Optional[torch.Tensor] = None    # [N]
        self._nid_to_cache_idx: Dict[int, int] = {}
        self._cached_nids_set: frozenset = frozenset()
        self._feat_dim: Optional[int] = None

        if prefill:
            self.prefill_hot_cache(graph_name="", k=self.cache_size)

    # ------------------------------------------------------------------
    # Cache-size computation
    # ------------------------------------------------------------------

    def compute_cache_size(self, cache_size_GB: float) -> int:
        """Return a provisional cache size; final size is set in prefill."""
        # We don't know feat_dim yet, so return a placeholder.
        # The real value is computed in prefill_hot_cache once we know feat_dim.
        return 0

    def _compute_capacity(self, feat_dim: int) -> int:
        """Compute how many nodes fit given feat_dim and sizing strategy."""
        bytes_per_node = feat_dim * 4 + 8  # float32 features + int64 label
        if self._auto_size and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            available = max(0, free_bytes - self._reserved_bytes)
        else:
            available = int(self._cache_size_GB * (1024 ** 3))
        return max(0, available // bytes_per_node)

    # ------------------------------------------------------------------
    # Abstract interface — compatibility path (per-node, not vectorized)
    # ------------------------------------------------------------------

    def _split_key(self, key) -> Tuple[str, int]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Cache keys must be (attr_name, nid) tuples")
        attr_name, nid = key
        if attr_name not in {"x", "y"}:
            raise KeyError(f"Unsupported attr_name {attr_name!r}; expected 'x' or 'y'")
        return attr_name, int(nid)

    def get(self, key):
        """Return cached value for *key*, or ``None`` if not cached.

        Returns a CPU numpy array for ``"x"`` and a plain Python int for
        ``"y"`` to stay compatible with ``Neo4jAbstractFS._get_cached_value``.
        The hot path uses :meth:`fetch_batch` instead.
        """
        attr_name, nid = self._split_key(key)
        idx = self._nid_to_cache_idx.get(nid)
        if idx is None:
            return None
        if attr_name == "x":
            return self.cached_features[idx].cpu().numpy()
        return int(self.cached_labels[idx].item())

    def set(self, key, value) -> None:
        """No-op: static cache is not updated at runtime."""

    def delete(self, key) -> None:
        """No-op: static cache cannot be modified after prefill."""

    def __delitem__(self, key) -> None:
        self.delete(key)

    def clear(self) -> None:
        self.cached_features = None
        self.cached_labels = None
        self._nid_to_cache_idx = {}
        self._cached_nids_set = frozenset()

    def __contains__(self, key) -> bool:
        try:
            _, nid = self._split_key(key)
        except KeyError:
            return False
        return nid in self._cached_nids_set

    # ------------------------------------------------------------------
    # Vectorized batch fetch — the fast path used by Neo4jGPUCachedFS
    # ------------------------------------------------------------------

    def fetch_batch(
        self, nids: List[int]
    ) -> Tuple[
        Optional[torch.Tensor],  # cached features  [n_cached, feat_dim] on device
        Optional[torch.Tensor],  # cached labels     [n_cached]          on device
        List[int],               # output positions for cached entries
        List[int],               # node IDs that were NOT cached (need DB)
        List[int],               # output positions for uncached entries
    ]:
        """Split *nids* into GPU-cached and uncached in one pass.

        Returns
        -------
        cached_x, cached_y, cached_pos, uncached_nids, uncached_pos
            ``cached_x`` and ``cached_y`` are GPU tensors (or ``None`` if
            nothing was cached).  ``cached_pos`` and ``uncached_pos`` are
            integer lists for scatter-assignment into the output tensor.
        """
        if self.cached_features is None:
            # Cache not yet filled — everything is a miss.
            return None, None, [], nids, list(range(len(nids)))

        cached_indices: List[int] = []
        cached_pos: List[int] = []
        uncached_nids: List[int] = []
        uncached_pos: List[int] = []

        for pos, nid in enumerate(nids):
            idx = self._nid_to_cache_idx.get(nid)
            if idx is not None:
                cached_indices.append(idx)
                cached_pos.append(pos)
            else:
                uncached_nids.append(nid)
                uncached_pos.append(pos)

        if not cached_indices:
            return None, None, [], uncached_nids, uncached_pos

        idx_t = torch.tensor(cached_indices, dtype=torch.long, device=self.device)
        cached_x = self.cached_features[idx_t]  # [n_cached, feat_dim]
        cached_y = self.cached_labels[idx_t]    # [n_cached]

        return cached_x, cached_y, cached_pos, uncached_nids, uncached_pos

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill_hot_cache(self, graph_name: str, k: int) -> None:
        """Fill GPU cache with top-K nodes ranked by out-degree.

        ``graph_name`` is accepted for API compatibility but ignored — no GDS
        projection is required.
        """
        # Out-degree query: count outgoing edges per node.
        # Nodes with high out-degree are sampled as neighbors most often.
        out_degree_query = (
            f"MATCH (n)-[r]->()\n"
            f"WITH n, count(r) AS out_deg\n"
            f"ORDER BY out_deg DESC\n"
            f"LIMIT $limit\n"
            f"RETURN n.{self.nodeid_property} AS id,\n"
            f"       n.{self.feature_property} AS feature,\n"
            f"       n.{self.target_property}  AS label"
        )

        label_query = (
            f"MATCH (n) WHERE n.{self.nodeid_property} IN $node_ids "
            f"RETURN n.{self.nodeid_property} AS id, "
            f"n.{self.target_property} AS label"
        )

        # Use a large initial limit so we can compute capacity from feat_dim
        # after seeing the first record.  We'll slice to actual capacity later.
        initial_limit = max(k, 1) if k > 0 else 100_000

        nids_list: List[int] = []
        feats_list: List[np.ndarray] = []
        labels_list: List[int] = []

        with self._get_driver().session(database=self.database_name) as session:
            for record in session.run(out_degree_query, limit=initial_limit):
                raw_feat = record["feature"]
                if raw_feat is None:
                    continue
                feat = self._normalize_feature_value(raw_feat)

                # Compute true capacity once we know feat_dim.
                if self._feat_dim is None:
                    self._feat_dim = feat.shape[0]
                    capacity = self._compute_capacity(self._feat_dim)
                    self.cache_size = capacity

                nid = int(record["id"])
                nids_list.append(nid)
                feats_list.append(feat)

                raw_label = record["label"]
                labels_list.append(self._normalize_label_value(raw_label))

                if len(nids_list) >= self.cache_size:
                    break

        if not nids_list:
            return

        feat_matrix = np.stack(feats_list, axis=0)  # [N, feat_dim]
        label_array = np.array(labels_list, dtype=np.int64)

        self.cached_features = torch.from_numpy(feat_matrix).to(self.device)
        self.cached_labels = torch.from_numpy(label_array).to(self.device)
        self._nid_to_cache_idx = {nid: i for i, nid in enumerate(nids_list)}
        self._cached_nids_set = frozenset(nids_list)

    # ------------------------------------------------------------------
    # Pickle safety (DataLoader workers)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_driver"] = None
        # Move tensors to CPU before pickling so workers can access them.
        if state.get("cached_features") is not None:
            state["cached_features"] = state["cached_features"].cpu()
        if state.get("cached_labels") is not None:
            state["cached_labels"] = state["cached_labels"].cpu()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None
        # Restore tensors to the configured device.
        if self.cached_features is not None:
            self.cached_features = self.cached_features.to(self.device)
        if self.cached_labels is not None:
            self.cached_labels = self.cached_labels.to(self.device)
