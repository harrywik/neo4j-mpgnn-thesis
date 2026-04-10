"""GPU-resident static feature cache.

Implements the PaGraph-style caching strategy:
- Nodes are ranked by out-degree (no GDS required).
- Top-K node features are loaded once into a contiguous GPU tensor.
- At batch time, a vectorized split separates cached from uncached nodes,
  avoiding per-node Python-dict lookups on the hot path.

Cache size is either:
- Fixed via ``cache_size_GB``  (default, no CUDA required), or
- Auto-sized from available GPU memory when ``auto_size=True``.

The cache itself has no DB dependency.  Use :func:`prefill_gpu_cache_from_neo4j`
to populate it from a Neo4j driver, then pass the result to the feature store.

References
----------
PaGraph: Scaling GNN Training on Large Graphs via Computation-aware Caching
Lin et al., SoCC 2020.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from neo4j import Driver

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class Neo4jGPUCache(Neo4jCache):
    """Static GPU-resident feature cache.

    Parameters
    ----------
    device:
        PyTorch device string (e.g. ``"cuda:0"``).  Falls back to ``"cpu"``
        when CUDA is unavailable.
    auto_size:
        If ``True``, cache capacity is derived from free GPU memory after a
        safety margin, ignoring ``cache_size_GB``.
    cache_size_GB:
        Explicit cache budget in gigabytes (used when ``auto_size=False``).
    reserved_gb:
        GPU memory reserved for model/activations (only used when
        ``auto_size=True``).  Default: 1 GB.
    """

    def __init__(
        self,
        device: str = "cuda",
        auto_size: bool = False,
        cache_size_GB: float = 0.5,
        reserved_gb: float = 1.0,
    ) -> None:
        self._cache_size_GB = cache_size_GB
        self._auto_size = auto_size
        self._reserved_bytes = int(reserved_gb * (1024 ** 3))

        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # GPU tensors — populated via load().
        self.cached_features: Optional[torch.Tensor] = None  # [N, feat_dim]
        self.cached_labels: Optional[torch.Tensor] = None    # [N]
        self._nid_to_cache_idx: Dict[int, int] = {}
        self._cached_nids_set: frozenset = frozenset()
        self._feat_dim: Optional[int] = None
        self.cache_size: int = 0

    # ------------------------------------------------------------------
    # Capacity computation
    # ------------------------------------------------------------------

    def compute_capacity(self, feat_dim: int) -> int:
        """Compute how many nodes fit given *feat_dim* and sizing strategy."""
        bytes_per_node = feat_dim * 4 + 8  # float32 features + int64 label
        if self._auto_size and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            available = max(0, free_bytes - self._reserved_bytes)
        else:
            available = int(self._cache_size_GB * (1024 ** 3))
        return max(0, available // bytes_per_node)

    # ------------------------------------------------------------------
    # Loading data into the cache
    # ------------------------------------------------------------------

    def load(
        self,
        nids: List[int],
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Populate the GPU cache with pre-fetched data.

        Parameters
        ----------
        nids:
            Node IDs in the same order as the rows of *features*.
        features:
            ``(N, feat_dim)`` float32 array of node features.
        labels:
            ``(N,)`` int64 array of node labels.
        """
        self.cached_features = torch.from_numpy(features).to(self.device)
        self.cached_labels = torch.from_numpy(labels).to(self.device)
        self._nid_to_cache_idx = {nid: i for i, nid in enumerate(nids)}
        self._cached_nids_set = frozenset(nids)
        self._feat_dim = features.shape[1] if features.ndim == 2 else None
        self.cache_size = len(nids)

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    @staticmethod
    def _split_key(key) -> Tuple[str, int]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Cache keys must be (attr_name, nid) tuples")
        attr_name, nid = key
        if attr_name not in {"x", "y"}:
            raise KeyError(f"Unsupported attr_name {attr_name!r}; expected 'x' or 'y'")
        return attr_name, int(nid)

    def get(self, key):
        """Return cached value, or ``None`` on miss.

        Returns a CPU numpy array for ``"x"`` and a plain int for ``"y"``
        to stay compatible with ``Neo4jFS._get_cached_value``.
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
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        List[int],
        List[int],
        List[int],
    ]:
        """Split *nids* into GPU-cached and uncached in one pass.

        Returns ``(cached_x, cached_y, cached_pos, uncached_nids, uncached_pos)``.
        """
        if self.cached_features is None:
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
        cached_x = self.cached_features[idx_t]
        cached_y = self.cached_labels[idx_t]

        return cached_x, cached_y, cached_pos, uncached_nids, uncached_pos

    # ------------------------------------------------------------------
    # Pickle safety (DataLoader workers)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("cached_features") is not None:
            state["cached_features"] = state["cached_features"].cpu()
        if state.get("cached_labels") is not None:
            state["cached_labels"] = state["cached_labels"].cpu()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.cached_features is not None:
            self.cached_features = self.cached_features.to(self.device)
        if self.cached_labels is not None:
            self.cached_labels = self.cached_labels.to(self.device)


# ---------------------------------------------------------------------------
# Factory: populate a GPU cache from Neo4j
# ---------------------------------------------------------------------------

def prefill_gpu_cache_from_neo4j(
    driver: Driver,
    database_name: str,
    nodeid_property: str = "nodeId",
    feature_property: str = "features",
    target_property: str = "category",
    feature_property_type: str = "f64[]",
    label_map: Optional[Dict[str, int]] = None,
    device: str = "cuda",
    auto_size: bool = False,
    cache_size_GB: float = 0.5,
    reserved_gb: float = 1.0,
) -> Neo4jGPUCache:
    """Create and populate a :class:`Neo4jGPUCache` from Neo4j out-degree ranking."""
    cache = Neo4jGPUCache(
        device=device,
        auto_size=auto_size,
        cache_size_GB=cache_size_GB,
        reserved_gb=reserved_gb,
    )

    labels: Dict[str, int] = dict(label_map) if label_map else {}

    def _normalize_feature(raw) -> np.ndarray:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return np.frombuffer(bytes(raw), dtype=np.float32).copy()
        return np.asarray(raw, dtype=np.float32)

    def _normalize_label(raw) -> int:
        if raw is None:
            return 0
        if isinstance(raw, str):
            if raw not in labels:
                labels[raw] = len(labels)
            return labels[raw]
        return int(raw)

    out_degree_query = (
        f"MATCH (n)-[r]->()\n"
        f"WITH n, count(r) AS out_deg\n"
        f"ORDER BY out_deg DESC\n"
        f"LIMIT $limit\n"
        f"RETURN n.{nodeid_property} AS id,\n"
        f"       n.{feature_property} AS feature,\n"
        f"       n.{target_property}  AS label"
    )

    initial_limit = 100_000
    nids_list: List[int] = []
    feats_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with driver.session(database=database_name) as session:
        for record in session.run(out_degree_query, limit=initial_limit):
            raw_feat = record["feature"]
            if raw_feat is None:
                continue
            feat = _normalize_feature(raw_feat)

            if cache._feat_dim is None:
                cache._feat_dim = feat.shape[0]
                cache.cache_size = cache.compute_capacity(cache._feat_dim)

            nids_list.append(int(record["id"]))
            feats_list.append(feat)
            labels_list.append(_normalize_label(record["label"]))

            if len(nids_list) >= cache.cache_size:
                break

    if nids_list:
        cache.load(
            nids_list,
            np.stack(feats_list, axis=0),
            np.array(labels_list, dtype=np.int64),
        )

    return cache
