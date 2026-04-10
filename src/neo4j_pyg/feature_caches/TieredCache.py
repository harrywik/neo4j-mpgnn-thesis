from typing import Dict, List

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class TieredCache(Neo4jCache):
    """Check caches in order; promote hits to earlier tiers.

    Usage::

        cache = TieredCache([
            StaticCache(hot_data),   # L1: static hot nodes
            LRUCache(10_000),        # L2: dynamic LRU
        ])
    """

    def __init__(self, tiers: List[Neo4jCache]) -> None:
        if not tiers:
            raise ValueError("Need at least one tier")
        self._tiers = tiers

    def get(self, key):
        for i, tier in enumerate(self._tiers):
            value = tier.get(key)
            if value is not None:
                for earlier in self._tiers[:i]:
                    earlier.set(key, value)
                return value
        return None

    def set(self, key, value):
        for tier in self._tiers:
            tier.set(key, value)

    def delete(self, key):
        for tier in self._tiers:
            tier.delete(key)

    def clear(self):
        for tier in self._tiers:
            tier.clear()

    def get_many(self, keys):
        found: Dict = {}
        remaining = set(keys)
        for tier in self._tiers:
            if not remaining:
                break
            hits = tier.get_many(remaining)
            found.update(hits)
            remaining -= hits.keys()
        return found

    def set_many(self, items):
        for tier in self._tiers:
            tier.set_many(items)
