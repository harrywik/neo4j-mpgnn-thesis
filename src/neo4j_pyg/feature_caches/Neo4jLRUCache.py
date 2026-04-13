"""Redis-backed LRU cache.

Writable at runtime — ``static = False`` so :class:`TieredCache` writes
through to it.  LRU eviction is handled by Redis: configure the server with::

    maxmemory <budget>
    maxmemory-policy allkeys-lru

The optional ``max_entries`` parameter adds a soft per-process cap — writes
are dropped once the local counter reaches the limit.  This is approximate
across workers (each worker tracks its own counter) but prevents any single
worker from flooding Redis.

Values use the same type-aware codec as :class:`Neo4jStaticCache`:

    'N' + np.save() bytes   → numpy array
    'I' + 8-byte LE int64   → int
    'F' + 8-byte LE float64 → float
    'P' + pickle bytes      → anything else
"""

from __future__ import annotations

import atexit
from typing import Optional

try:
    import redis as _redis_mod
except ImportError as exc:  # pragma: no cover
    raise ImportError("redis is required: pip install redis") from exc

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache
from neo4j_pyg.feature_caches.Neo4jStaticCache import _encode, _decode


class Neo4jLRUCache(Neo4jCache):
    """Redis-backed LRU cache.

    Parameters
    ----------
    redis_url:
        Redis connection URL.
    key_prefix:
        Namespace prefix for all Redis keys.
    ttl_seconds:
        Optional TTL per key.  ``None`` means keys never expire.
    max_entries:
        Soft per-process entry cap.  ``None`` means unlimited.
    maxmemory:
        Redis memory budget as a string (e.g. ``"2gb"``, ``"512mb"``).
        When set, ``maxmemory`` and ``maxmemory-policy allkeys-lru`` are
        configured on the Redis server on first connect.  ``None`` leaves
        Redis config unchanged.
    static:
        If ``True``, :class:`TieredCache` will not write through to this tier.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "lru",
        ttl_seconds: Optional[int] = None,
        max_entries: Optional[int] = None,
        maxmemory: Optional[str] = None,
        static: bool = False,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.maxmemory = maxmemory
        self.static = static
        self._entries_cached: int = 0
        self._client: Optional[_redis_mod.Redis] = None

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
            if self.maxmemory is not None:
                self._client.config_set("maxmemory", self.maxmemory)
                self._client.config_set("maxmemory-policy", "allkeys-lru")
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
        return f"{self.key_prefix}:{repr(key)}"

    # ------------------------------------------------------------------
    # Cache interface
    # ------------------------------------------------------------------

    def get(self, key):
        raw = self._get_client().get(self._redis_key(key))
        return None if raw is None else _decode(raw)

    def set(self, key, value) -> None:
        if self.max_entries is not None and self._entries_cached >= self.max_entries:
            return
        raw = _encode(value)
        if self.ttl_seconds is None:
            self._get_client().set(self._redis_key(key), raw)
        else:
            self._get_client().set(self._redis_key(key), raw, ex=self.ttl_seconds)
        self._entries_cached += 1

    def delete(self, key) -> None:
        self._get_client().delete(self._redis_key(key))

    def clear(self) -> None:
        client = self._get_client()
        keys = client.keys(f"{self.key_prefix}:*")
        if keys:
            client.delete(*keys)
        self._entries_cached = 0

    # ------------------------------------------------------------------
    # Batch interface
    # ------------------------------------------------------------------

    def get_many(self, keys) -> dict:
        keys = list(keys)
        raws = self._get_client().mget([self._redis_key(k) for k in keys])
        return {k: _decode(raw) for k, raw in zip(keys, raws) if raw is not None}

    def set_many(self, items: dict) -> None:
        if not items:
            return
        remaining = (
            self.max_entries - self._entries_cached
            if self.max_entries is not None
            else len(items)
        )
        if remaining <= 0:
            return
        pipe = self._get_client().pipeline(transaction=False)
        written = 0
        for k, value in items.items():
            if written >= remaining:
                break
            raw = _encode(value)
            if self.ttl_seconds is None:
                pipe.set(self._redis_key(k), raw)
            else:
                pipe.set(self._redis_key(k), raw, ex=self.ttl_seconds)
            written += 1
        pipe.execute()
        self._entries_cached += written

    # ------------------------------------------------------------------
    # Pickle safety
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
