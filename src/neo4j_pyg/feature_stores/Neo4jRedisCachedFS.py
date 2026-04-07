"""Redis-cached Neo4j feature store.

Uses :class:`Neo4jRedisCache` (L1 per-worker LRU + L2 shared Redis) and
overrides ``_multi_get_tensor`` to call ``get_many``/``set_many`` so that
the entire batch is resolved in one Redis MGET and one pipeline write —
no per-node Python loops on the hot path.

All sampling workers share the same Redis backend, so a node fetched once
by any worker is available to all others from that point on.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from neo4j import Driver
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

from benchmarking_tools import Measurer
from neo4j_pyg.feature_caches.Neo4jRedisCache import Neo4jRedisCache
from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS


class Neo4jRedisCachedFS(Neo4jAbstractFS):
    """Feature store with L1 per-worker LRU + L2 shared Redis cache.

    Parameters
    ----------
    feature_dim:
        Dimensionality of node feature vectors.  Required for Redis value
        decoding.  Must match the ``feature_dim`` stored in the dataset config.
    redis_url:
        Redis connection URL.  Use ``"unix:///tmp/redis.sock?db=0"`` for
        a Unix socket (lowest latency when Redis is on the same machine).
    redis_key_prefix:
        Namespace prefix for all Redis keys.
    redis_ttl_seconds:
        Optional TTL for cached entries.  ``None`` → keys never expire.
    l1_maxsize:
        Max entries in the per-worker in-process LRU.  Set to 0 to disable.
    """

    def __init__(
        self,
        driver: Optional[Driver] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        measurer: Optional[Measurer] = None,
        database_name: Optional[str] = None,
        dataset_name: str = "neo4j",
        feature_property: str = "features",
        target_property: str = "category",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict] = None,
        feature_dim: int = 128,
        redis_url: str = "redis://localhost:6379/0",
        redis_key_prefix: str = "fs",
        redis_ttl_seconds: Optional[int] = None,
        l1_maxsize: int = 10_000,
    ) -> None:
        cache = Neo4jRedisCache(
            feature_dim=feature_dim,
            redis_url=redis_url,
            key_prefix=redis_key_prefix,
            ttl_seconds=redis_ttl_seconds,
            l1_maxsize=l1_maxsize,
            label_map=label_map,
        )
        super().__init__(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            measurer=measurer,
            database_name=database_name,
            dataset_name=dataset_name,
            feature_property=feature_property,
            target_property=target_property,
            split_property_name=split_property_name,
            split_property_type=split_property_type,
            nodeid_property=nodeid_property,
            feature_property_type=feature_property_type,
            cache=cache,
        )
        self._redis_cache: Neo4jRedisCache = cache
        self._feature_dim = feature_dim
        # Build a deterministic label map so the integer encoding stored in
        # Redis is stable across runs.  Without this, the lazy _labels dict
        # can map the same string label to different integers in different runs,
        # causing silently wrong labels for cache-hit nodes.
        self._init_label_map()

    def _init_label_map(self) -> None:
        """Query all distinct labels once and fix a deterministic mapping.

        Only runs when labels are strings (e.g. Cora).  Integer labels
        (OGB datasets) need no mapping and are skipped.
        """
        query = (
            f"MATCH (n) WHERE n.{self.target_property} IS NOT NULL "
            f"RETURN DISTINCT n.{self.target_property} AS label LIMIT 1"
        )
        with self._get_driver().session(database=self.database_name) as session:
            sample = session.run(query).single()

        if sample is None or not isinstance(sample["label"], str):
            return  # integer labels — no mapping needed

        all_labels_query = (
            f"MATCH (n) WHERE n.{self.target_property} IS NOT NULL "
            f"RETURN DISTINCT n.{self.target_property} AS label ORDER BY label"
        )
        with self._get_driver().session(database=self.database_name) as session:
            records = list(session.run(all_labels_query))

        label_map = {rec["label"]: i for i, rec in enumerate(records)}
        # Overwrite _labels on both the FS and the cache (they share the same
        # dict via Neo4jAbstractFS.__init__, but set both to be safe).
        self._labels.update(label_map)
        self._redis_cache._labels.update(label_map)

    def _prefill_hot_cache(self, graph_name: str, k: int = 0, **kwargs) -> None:
        """No-op: Redis cache warms up on-demand during training."""

    # ------------------------------------------------------------------
    # Vectorised hot path
    # ------------------------------------------------------------------

    def _multi_get_tensor(
        self, attrs: List[TensorAttr]
    ) -> List[Optional[FeatureTensorType]]:
        """Batch cache lookup: one MGET for the whole batch, one pipeline write for misses.

        Steps
        -----
        1. Build all ``("x", nid)`` and ``("y", nid)`` keys.
        2. ``get_many`` resolves L1 hits locally and L2 hits via a single MGET.
        3. Remaining misses are fetched from Neo4j in one round-trip.
        4. ``set_many`` writes all misses back to Redis in a single pipeline.
        5. Assemble numpy arrays → tensors.
        """
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        if x_attr is None or y_attr is None:
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()
        n = len(node_ids)
        nid_to_pos: Dict[int, int] = {nid: i for i, nid in enumerate(node_ids)}

        # --- batch cache lookup ---
        x_keys = [("x", nid) for nid in node_ids]
        y_keys = [("y", nid) for nid in node_ids]
        all_keys = x_keys + y_keys

        cached = self._redis_cache.get_many(all_keys)

        cached_x: Dict = {k: v for k, v in cached.items() if k[0] == "x"}
        cached_y: Dict = {k: v for k, v in cached.items() if k[0] == "y"}

        missing: List[int] = [
            nid for nid in node_ids
            if ("x", nid) not in cached_x or ("y", nid) not in cached_y
        ]

        n_hit = n - len(missing)
        if self.measurer is not None:
            self.measurer.log_event("cache_hit", n_hit)
            self.measurer.log_event("cache_miss", len(missing))

        # --- fetch misses from Neo4j ---
        if missing:
            if self.measurer is not None:
                self.measurer.log_event("remote_feature_fetch", 1)

            fetched_nids, feat_matrix, y_array = self._get_both_from_db(missing, x_attr)

            if self._feature_dim is None and len(feat_matrix):
                self._feature_dim = feat_matrix.shape[1]

            # Write misses back to Redis in one pipeline.
            to_cache: dict = {}
            for i, nid in enumerate(fetched_nids):
                to_cache[("x", nid)] = feat_matrix[i]
                to_cache[("y", nid)] = int(y_array[i])
            self._redis_cache.set_many(to_cache)

            # Merge fetched data into local dicts for assembly below.
            for i, nid in enumerate(fetched_nids):
                cached_x[("x", nid)] = feat_matrix[i]
                cached_y[("y", nid)] = int(y_array[i])

        self.t_feat_etl_start = time.monotonic()

        # --- assemble output tensors ---
        feat_dim = self._feature_dim
        if feat_dim is None and cached_x:
            feat_dim = next(iter(cached_x.values())).shape[0]
            self._feature_dim = feat_dim

        feat_out = np.empty((n, feat_dim), dtype=np.float32)
        y_out = np.empty(n, dtype=np.int64)

        for nid in node_ids:
            pos = nid_to_pos[nid]
            feat_out[pos] = cached_x[("x", nid)]
            y_out[pos] = cached_y[("y", nid)]

        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")
            self.measurer.log_event(
                "feat_x_etl_ms",
                (time.monotonic() - self.t_feat_etl_start) * 1000,
            )

        x_tensor = torch.from_numpy(feat_out)
        y_tensor = torch.from_numpy(y_out)

        return [
            x_tensor if a.attr_name == "x" else y_tensor
            for a in attrs
        ]
