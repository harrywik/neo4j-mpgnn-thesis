"""Tests for Neo4jRedisCache and Neo4jRedisCachedFS.

Run with:
    .venv/bin/python -m pytest tests/test_redis_cache.py -v

The mock-based tests work without a running Redis server.
The integration tests are skipped automatically when Redis is unavailable.
"""

import struct
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from neo4j_pyg.feature_caches.Neo4jRedisCache import (
    Neo4jRedisCache,
    encode_feature, decode_feature,
    encode_label, decode_label,
)


# ---------------------------------------------------------------------------
# Codec round-trip tests — no Redis needed
# ---------------------------------------------------------------------------

def test_feature_encode_decode_roundtrip():
    arr = np.array([1.0, 2.5, -0.3, 0.0], dtype=np.float32)
    assert np.allclose(decode_feature(encode_feature(arr), dim=4), arr)


def test_label_encode_decode_roundtrip():
    for y in [0, 1, 42, -1, 2**31]:
        assert decode_label(encode_label(y)) == y


def test_decode_feature_returns_copy():
    arr = np.array([1.0, 2.0], dtype=np.float32)
    buf = encode_feature(arr)
    decoded = decode_feature(buf, dim=2)
    decoded[0] = 99.0  # mutating the result should not affect buf
    assert decode_feature(buf, dim=2)[0] == 1.0


# ---------------------------------------------------------------------------
# Neo4jRedisCache unit tests (Redis client mocked)
# ---------------------------------------------------------------------------

@pytest.fixture
def cache():
    """Return a Neo4jRedisCache with a mocked Redis client."""
    c = Neo4jRedisCache(feature_dim=4, redis_url="redis://localhost:6379/0", l1_maxsize=5)
    c._client = MagicMock()
    return c


def _make_feat(val: float, dim: int = 4) -> np.ndarray:
    return np.full(dim, val, dtype=np.float32)


def test_set_then_get_feature(cache):
    feat = _make_feat(1.5)
    cache._client.get.return_value = encode_feature(feat)
    cache.set(("x", 1), feat)
    result = cache.get(("x", 1))
    assert np.allclose(result, feat)


def test_get_returns_none_on_miss(cache):
    cache._client.get.return_value = None
    assert cache.get(("x", 99)) is None


def test_l1_hit_skips_redis(cache):
    feat = _make_feat(3.0)
    # Pre-populate L1 directly.
    cache._l1[("x", 7)] = feat
    result = cache.get(("x", 7))
    assert np.allclose(result, feat)
    cache._client.get.assert_not_called()


def test_get_many_uses_mget(cache):
    feats = {1: _make_feat(1.0), 2: _make_feat(2.0)}
    raws = [encode_feature(feats[1]), encode_feature(feats[2])]
    cache._client.mget.return_value = raws

    keys = [("x", 1), ("x", 2)]
    result = cache.get_many(keys)

    cache._client.mget.assert_called_once()
    assert set(result.keys()) == {("x", 1), ("x", 2)}
    assert np.allclose(result[("x", 1)], feats[1])
    assert np.allclose(result[("x", 2)], feats[2])


def test_get_many_partial_miss(cache):
    feat1 = _make_feat(1.0)
    # key ("x", 1) is in L1, key ("x", 2) is a Redis miss, key ("x", 3) hits Redis
    cache._l1[("x", 1)] = feat1
    feat3 = _make_feat(3.0)
    # MGET returns [None, bytes] for keys ("x", 2) and ("x", 3)
    cache._client.mget.return_value = [None, encode_feature(feat3)]

    result = cache.get_many([("x", 1), ("x", 2), ("x", 3)])

    assert ("x", 1) in result  # L1 hit
    assert ("x", 2) not in result  # total miss
    assert ("x", 3) in result  # Redis hit
    np.allclose(result[("x", 1)], feat1)
    np.allclose(result[("x", 3)], feat3)


def test_set_many_uses_pipeline(cache):
    pipe = MagicMock()
    cache._client.pipeline.return_value = pipe
    items = {
        ("x", 1): _make_feat(1.0),
        ("y", 1): 3,
    }
    cache.set_many(items)
    assert pipe.set.call_count == 2
    pipe.execute.assert_called_once()


def test_get_label(cache):
    cache._client.get.return_value = encode_label(7)
    result = cache.get(("y", 42))
    assert result == 7


def test_set_many_populates_l1(cache):
    pipe = MagicMock()
    cache._client.pipeline.return_value = pipe
    feat = _make_feat(5.0)
    cache.set_many({("x", 10): feat})
    assert ("x", 10) in cache._l1


def test_pickle_safety(cache):
    import pickle
    state = cache.__getstate__()
    assert state["_client"] is None
    # L1 should be a fresh empty cache after unpickling.
    assert len(state["_l1"]) == 0


def test_key_format(cache):
    key = cache._redis_key(("x", 123))
    assert key == "fs:x:123"
    key_y = cache._redis_key(("y", 456))
    assert key_y == "fs:y:456"


# ---------------------------------------------------------------------------
# Integration test — only runs when Redis is reachable
# ---------------------------------------------------------------------------

def _redis_available(url: str = "redis://localhost:6379/0") -> bool:
    try:
        import redis
        r = redis.Redis.from_url(url, socket_connect_timeout=1)
        r.ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="Redis not reachable")
def test_integration_set_get_many():
    cache = Neo4jRedisCache(
        feature_dim=4,
        redis_url="redis://localhost:6379/0",
        key_prefix="test_integration",
        ttl_seconds=60,
        l1_maxsize=0,  # disable L1 so we actually hit Redis
    )
    feat = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    nid = 99_999

    cache.set_many({("x", nid): feat, ("y", nid): 5})
    result = cache.get_many([("x", nid), ("y", nid)])

    assert ("x", nid) in result
    assert ("y", nid) in result
    assert np.allclose(result[("x", nid)], feat)
    assert result[("y", nid)] == 5

    # Cleanup
    cache.delete(("x", nid))
    cache.delete(("y", nid))
