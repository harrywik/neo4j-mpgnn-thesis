"""Redis-backed shared feature cache with per-worker LRU L1.

Architecture
------------
L1 — per-worker in-process LRU (``cachetools.LRUCache``)
     Fastest repeated hits inside one worker. Never crosses process boundary.

L2 — Redis (shared by all sampling workers on the machine)
     Workers read/write via MGET/pipeline for low round-trip overhead.
     Use a Unix socket for lowest IPC latency when Redis is co-located.

L3 — Neo4j (handled by the feature store, not this class)

Key format:  ``{prefix}:x:{nid}``  and  ``{prefix}:y:{nid}``
Values:
  - features (x) → raw float32 bytes (C-contiguous)
  - labels   (y) → little-endian int64 (8 bytes)

Pickle safety
-------------
The live Redis client is excluded from ``__getstate__`` so that PyTorch
DataLoader workers can unpickle the object and lazily reconnect.
"""

from __future__ import annotations

import atexit
import struct
from typing import Dict, List, Optional

import psutil

import numpy as np

try:
    import redis as _redis_mod
except ImportError as exc:  # pragma: no cover
    raise ImportError("redis is required: pip install redis") from exc

try:
    from cachetools import LRUCache as _LRUCache
except ImportError as exc:  # pragma: no cover
    raise ImportError("cachetools is required: pip install cachetools") from exc

from neo4j_pyg.feature_caches.Neo4jAbstractCache import Neo4jAbstractCache


# ---------------------------------------------------------------------------
# Encoding helpers (module-level so they can be imported by tests)
# ---------------------------------------------------------------------------

def encode_feature(arr: np.ndarray) -> bytes:
    arr = np.asarray(arr, dtype=np.float32, order="C")
    return arr.tobytes()


def decode_feature(buf: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(buf, dtype=np.float32, count=dim).copy()


def encode_label(y: int) -> bytes:
    return struct.pack("<q", int(y))


def decode_label(buf: bytes) -> int:
    return struct.unpack("<q", buf)[0]


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class Neo4jRedisCache(Neo4jAbstractCache):
    """Two-level cache: per-worker LRU (L1) + shared Redis (L2).

    Parameters
    ----------
    feature_dim:
        Number of float32 elements per node feature vector.  Required for
        decoding raw bytes back into numpy arrays.
    redis_url:
        Redis connection URL.  Use ``"unix:///tmp/redis.sock?db=0"`` for
        Unix socket (lowest latency when Redis is on the same machine).
    key_prefix:
        Namespace prefix for all Redis keys (avoids collisions with other
        applications sharing the same Redis instance).
    ttl_seconds:
        Optional TTL for every Redis key.  ``None`` means keys never expire.
    l1_maxsize:
        Maximum number of entries in the per-worker LRU.
        Set to 0 to disable L1.
    memory_GB:
        Maximum RAM to use for cached feature vectors.  When ``None`` (default)
        the limit is computed lazily on the first ``set_many`` call by measuring
        available system RAM at that point (after the first training batch has
        allocated its working set) and leaving a 1 GB safety margin.  When set
        to a float the limit is computed immediately from the given value.
    """

    def __init__(
        self,
        feature_dim: int,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "fs",
        ttl_seconds: Optional[int] = None,
        l1_maxsize: int = 10_000,
        # Neo4jAbstractCache requires these but they are not used by Redis
        driver=None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        database_name: Optional[str] = None,
        nodeid_property: str = "nodeId",
        feature_property: str = "features",
        target_property: str = "category",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict] = None,
        cache_size_GB: float = 0.0,
        memory_GB: Optional[float] = None,
    ) -> None:
        # Do not call super().__init__() — the abstract base queries Neo4j to
        # estimate cache size, which is not needed for Redis.  Set the required
        # attributes manually instead.
        self.driver = driver
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.database_name = database_name or "neo4j"
        self.nodeid_property = nodeid_property
        self.feature_property = feature_property
        self.target_property = target_property
        self.feature_property_type = feature_property_type
        self._labels: Dict[str, int] = dict(label_map) if label_map else {}
        self._driver = None
        self.cache_size = 0  # not meaningful for Redis; size is managed by Redis

        self.feature_dim = feature_dim
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.l1_maxsize = l1_maxsize
        self.memory_GB = memory_GB

        # Capacity tracking: how many nodes (x+y pairs) may be written to Redis.
        # None means unlimited; set via _init_capacity() on first set_many call.
        self._node_limit: Optional[int] = None
        self._capacity_initialized: bool = False
        self._nodes_cached: int = 0

        if memory_GB is not None:
            self._init_capacity()

        # L1: per-process LRU — not shared across workers, but avoids Redis
        # round-trips for repeated accesses within the same worker.
        self._l1: Optional[_LRUCache] = _LRUCache(maxsize=l1_maxsize) if l1_maxsize > 0 else None
        # Lazy Redis client — created on first use, excluded from pickling.
        self._client: Optional[_redis_mod.Redis] = None

    # ------------------------------------------------------------------
    # Capacity management
    # ------------------------------------------------------------------

    def _init_capacity(self) -> None:
        """Compute and store the node limit from memory_GB or available RAM."""
        if self.memory_GB is not None:
            bytes_available = int(self.memory_GB * (1024 ** 3))
        else:
            available = psutil.virtual_memory().available
            # Leave a 1 GB safety margin so the OS and other processes stay healthy.
            bytes_available = max(0, available - 1 * (1024 ** 3))
        bytes_per_node = self.feature_dim * 4  # float32
        self._node_limit = max(0, bytes_available // bytes_per_node)
        self._capacity_initialized = True

    # ------------------------------------------------------------------
    # Redis client lifecycle
    # ------------------------------------------------------------------

    def _get_client(self) -> "_redis_mod.Redis":
        if self._client is None:
            self._client = _redis_mod.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=5,
                socket_timeout=5,
                decode_responses=False,
            )
            atexit.register(self.close)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    # ------------------------------------------------------------------
    # Key encoding
    # ------------------------------------------------------------------

    def _redis_key(self, key) -> str:
        attr_name, nid = key
        return f"{self.key_prefix}:{attr_name}:{nid}"

    # ------------------------------------------------------------------
    # Abstract interface — single-key path (used internally by get_many)
    # ------------------------------------------------------------------

    def get(self, key):
        attr_name, nid = key
        # L1
        if self._l1 is not None and key in self._l1:
            return self._l1[key]
        # L2 Redis
        raw = self._get_client().get(self._redis_key(key))
        if raw is None:
            return None
        value = self._decode(attr_name, raw)
        if self._l1 is not None:
            self._l1[key] = value
        return value

    def set(self, key, value) -> None:
        attr_name, nid = key
        raw = self._encode(attr_name, value)
        if self.ttl_seconds is None:
            self._get_client().set(self._redis_key(key), raw)
        else:
            self._get_client().set(self._redis_key(key), raw, ex=self.ttl_seconds)
        if self._l1 is not None:
            self._l1[key] = value

    def delete(self, key) -> None:
        self._get_client().delete(self._redis_key(key))
        if self._l1 is not None:
            self._l1.pop(key, None)

    def clear(self) -> None:
        if self._l1 is not None:
            self._l1.clear()
        # We do NOT flush Redis — other workers/trainers may be sharing it.

    # ------------------------------------------------------------------
    # Batch interface — the fast path used by Neo4jRedisCachedFS
    # ------------------------------------------------------------------

    def get_many(self, keys) -> dict:
        """Return {key: value} for all keys present in L1 or Redis.

        Uses a single MGET for L2 misses after checking L1.
        """
        result: dict = {}
        l2_keys: List = []  # keys that missed L1

        for k in keys:
            if self._l1 is not None and k in self._l1:
                result[k] = self._l1[k]
            else:
                l2_keys.append(k)

        if not l2_keys:
            return result

        redis_keys = [self._redis_key(k) for k in l2_keys]
        raws = self._get_client().mget(redis_keys)

        for k, raw in zip(l2_keys, raws):
            if raw is None:
                continue
            attr_name, _ = k
            value = self._decode(attr_name, raw)
            result[k] = value
            if self._l1 is not None:
                self._l1[k] = value

        return result

    def set_many(self, items: dict) -> None:
        """Write all key→value pairs to Redis in a single pipeline.

        Enforces the node capacity limit derived from ``memory_GB`` (or
        available RAM when ``memory_GB`` is ``None``).  Initialises capacity
        lazily on the first call so that the first training batch has already
        allocated its working set before we measure free memory.
        """
        if not items:
            return

        if not self._capacity_initialized:
            self._init_capacity()

        if self._node_limit == 0:
            return  # no memory allocated for cache

        # Count how many new nodes (not yet counted) are in this batch.
        # Each node contributes one "x" key; "y" keys ride along for free.
        new_node_count = sum(1 for attr, _ in items if attr == "x")

        if self._node_limit is not None and self._nodes_cached >= self._node_limit:
            return  # cache full — skip write

        # If writing this batch would exceed the limit, trim to what fits.
        if self._node_limit is not None:
            remaining = self._node_limit - self._nodes_cached
            if new_node_count > remaining:
                # Keep only the first `remaining` nodes (x + y pairs).
                trimmed: dict = {}
                kept = 0
                seen_nids: set = set()
                for k, value in items.items():
                    attr, nid = k
                    if nid not in seen_nids:
                        if kept >= remaining:
                            continue
                        seen_nids.add(nid)
                        kept += 1
                    trimmed[k] = value
                items = trimmed
                new_node_count = remaining

        pipe = self._get_client().pipeline(transaction=False)
        for k, value in items.items():
            attr_name, _ = k
            raw = self._encode(attr_name, value)
            if self.ttl_seconds is None:
                pipe.set(self._redis_key(k), raw)
            else:
                pipe.set(self._redis_key(k), raw, ex=self.ttl_seconds)
        pipe.execute()
        self._nodes_cached += new_node_count

        if self._l1 is not None:
            for k, value in items.items():
                self._l1[k] = value

    # ------------------------------------------------------------------
    # Codec helpers
    # ------------------------------------------------------------------

    def _encode(self, attr_name: str, value) -> bytes:
        if attr_name == "x":
            return encode_feature(value)
        if attr_name == "y":
            return encode_label(int(value))
        raise ValueError(f"Unsupported attr_name: {attr_name!r}")

    def _decode(self, attr_name: str, raw: bytes):
        if attr_name == "x":
            return decode_feature(raw, self.feature_dim)
        if attr_name == "y":
            return decode_label(raw)
        raise ValueError(f"Unsupported attr_name: {attr_name!r}")

    # ------------------------------------------------------------------
    # Unused abstract methods (Redis cache has no PageRank prefill)
    # ------------------------------------------------------------------

    def prefill_hot_cache(self, graph_name: str, k: int) -> None:
        """No-op: Redis cache is filled on-demand during training."""

    # ------------------------------------------------------------------
    # Pickle safety — exclude live client and L1 (L1 is per-process)
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        # L1 is per-process; workers start with an empty L1 and warm it up.
        state["_l1"] = _LRUCache(maxsize=self.l1_maxsize) if self.l1_maxsize > 0 else None
        # _node_limit and _nodes_cached are preserved so workers share the same
        # capacity decision without re-measuring RAM.  Workers do NOT write back
        # _nodes_cached to the main process, but that is acceptable — the limit
        # is a soft bound, not a hard quota.
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
