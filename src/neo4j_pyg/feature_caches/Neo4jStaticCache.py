"""Static Redis-backed out-degree cache.

Pre-filled once at startup with the top-K nodes ranked by out-degree.
Read-only at runtime — ``static = True`` so :class:`TieredCache` never
writes through to it.

Values are encoded with a tag byte so the decoder is self-describing:

    'N' + dtype(3B) + ndim(1B) + shape(ndim*4B LE uint32) + raw bytes
         → numpy array  (faster than np.save — no BytesIO, no format header)
    'I' + 8-byte LE int64   → int
    'F' + 8-byte LE float64 → float
    'P' + pickle bytes      → anything else (fallback)

Keys are encoded as ``{prefix}:{repr(key)}``, so any hashable key works.

NOTE: the numpy encoding format changed from np.save() to raw bytes.
Flush Redis (``make redis-flush``) before first run after upgrading.
"""

from __future__ import annotations

import atexit
import pickle
import struct
from typing import Dict, Optional

import numpy as np

try:
    import redis as _redis_mod
except ImportError as exc:  # pragma: no cover
    raise ImportError("redis is required: pip install redis") from exc

from neo4j import GraphDatabase

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

_TAG_NUMPY  = b'N'
_TAG_INT    = b'I'
_TAG_FLOAT  = b'F'
_TAG_PICKLE = b'P'


def _encode(value) -> bytes:
    if isinstance(value, np.ndarray):
        dt = value.dtype.str.encode()          # e.g. b'<f4' — always 3 bytes
        ndim = len(value.shape)
        header = struct.pack(f"<B{ndim}I", ndim, *value.shape)
        return _TAG_NUMPY + dt + header + value.tobytes()
    if isinstance(value, (int, np.integer)):
        return _TAG_INT + struct.pack("<q", int(value))
    if isinstance(value, (float, np.floating)):
        return _TAG_FLOAT + struct.pack("<d", float(value))
    return _TAG_PICKLE + pickle.dumps(value)


def _decode(raw: bytes):
    tag, data = raw[0:1], raw[1:]
    if tag == _TAG_NUMPY:
        dt = data[:3].decode()                 # e.g. '<f4'
        ndim = struct.unpack("<B", data[3:4])[0]
        shape = struct.unpack(f"<{ndim}I", data[4:4 + 4 * ndim])
        arr = np.frombuffer(data[4 + 4 * ndim:], dtype=dt).reshape(shape)
        return arr.copy()                      # copy → writable, releases buffer ref
    if tag == _TAG_INT:
        return struct.unpack("<q", data)[0]
    if tag == _TAG_FLOAT:
        return struct.unpack("<d", data)[0]
    return pickle.loads(data)


# ---------------------------------------------------------------------------
# Cache class
# ---------------------------------------------------------------------------

class Neo4jStaticCache(Neo4jCache):
    """Static Redis-backed cache pre-filled with top-K out-degree nodes.

    ``static = True``: read-only at runtime.  Call :meth:`fill_from_neo4j`
    once before training to populate.

    Values are promoted to an in-process dict on first access so repeated
    reads within the same process are pure in-memory (no Redis round-trip).

    Parameters
    ----------
    redis_url:
        Redis connection URL.
    key_prefix:
        Namespace prefix for all Redis keys.
    ttl_seconds:
        Optional TTL per key.  ``None`` means keys never expire.
    """

    static = True

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        key_prefix: str = "neo4j",
        ttl_seconds: Optional[int] = None,
    ) -> None:
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds
        self._client: Optional[_redis_mod.Redis] = None
        self._local: Dict = {}  # in-process promotion cache

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
        return f"{self.key_prefix}:{repr(key)}"

    # ------------------------------------------------------------------
    # Cache interface — reads only
    # ------------------------------------------------------------------

    def get(self, key):
        val = self._local.get(key)
        if val is not None:
            return val
        raw = self._get_client().get(self._redis_key(key))
        if raw is None:
            return None
        val = _decode(raw)
        self._local[key] = val
        return val

    def get_many(self, keys) -> dict:
        keys = list(keys)
        result: Dict = {}
        misses = []
        for k in keys:
            v = self._local.get(k)
            if v is not None:
                result[k] = v
            else:
                misses.append(k)
        if misses:
            raws = self._get_client().mget([self._redis_key(k) for k in misses])
            for k, raw in zip(misses, raws):
                if raw is not None:
                    val = _decode(raw)
                    self._local[k] = val
                    result[k] = val
        return result

    def set(self, key, value) -> None:
        """No-op: static cache is not updated at runtime."""

    def delete(self, key) -> None:
        """No-op."""

    def clear(self) -> None:
        """No-op: does not flush Redis — other workers share it."""

    # ------------------------------------------------------------------
    # Prefill from Neo4j via GDS PageRank
    # ------------------------------------------------------------------

    def fill_from_neo4j(
        self,
        uri: str,
        user: str,
        pwd: str,
        database: str,
        k: int = 1000,
        nodeid_property: str = "nodeId",
        feature_property: str = "features",
        target_property: str = "category",
        label_map: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        """Populate Redis with top-*k* nodes ranked by out-degree.

        Creates a temporary driver from *uri*/*user*/*pwd*, queries Neo4j,
        writes results to Redis via a single pipeline, then closes the driver.
        The cache remains pickle-safe after this call — no driver reference
        is retained.
        """
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

        query = (
            f"MATCH (n)-[r]->()\n"
            f"WITH n, count(r) AS out_deg\n"
            f"ORDER BY out_deg DESC LIMIT $limit\n"
            f"RETURN n.{nodeid_property} AS id,\n"
            f"       n.{feature_property} AS feature,\n"
            f"       n.{target_property}  AS label"
        )

        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        try:
            with driver.session(database=database) as session:
                pipe = self._get_client().pipeline(transaction=False)
                written = 0

                for record in session.run(query, limit=k):
                    raw_feat = record["feature"]
                    if raw_feat is None:
                        continue
                    nid = int(record["id"])
                    feat = _normalize_feature(raw_feat)
                    label = _normalize_label(record["label"])

                    x_key = self._redis_key(("x", nid))
                    y_key = self._redis_key(("y", nid))
                    x_raw = _encode(feat)
                    y_raw = _encode(label)

                    if self.ttl_seconds is None:
                        pipe.set(x_key, x_raw)
                        pipe.set(y_key, y_raw)
                    else:
                        pipe.set(x_key, x_raw, ex=self.ttl_seconds)
                        pipe.set(y_key, y_raw, ex=self.ttl_seconds)

                    written += 1
                    if written >= k:
                        break

                pipe.execute()
        finally:
            driver.close()

    # ------------------------------------------------------------------
    # Pickle safety
    # ------------------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None
        state["_local"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
