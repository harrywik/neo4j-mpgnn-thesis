from typing import Dict, List

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache


class TieredCache(Neo4jCache):
    """Read-through, write-through tiered cache.

    Tiers are checked top-to-bottom on reads.  On a hit, the value is
    promoted (written) to all upper tiers that are not ``static``.
    On writes, the value is written to every non-``static`` tier
    (write-through).

    Usage::

        cache = TieredCache([
            LRUCache(10_000),            # L1: fast in-process LRU
            Neo4jStaticCache(...),         # L2: shared Redis
            Neo4jFetchCache(driver, ...), # L3: DB source-of-truth (static)
        ])
    """

    def __init__(self, tiers: List[Neo4jCache]) -> None:
        if not tiers:
            raise ValueError("Need at least one tier")
        self._tiers = tiers

    # ------------------------------------------------------------------
    # Read-through with promotion
    # ------------------------------------------------------------------

    def get(self, key):
        for i, tier in enumerate(self._tiers):
            value = tier.get(key)
            if value is not None:
                # Promote to all upper non-static tiers
                for j in range(i):
                    if not self._tiers[j].static:
                        self._tiers[j].set(key, value)
                return value
        return None

    def get_many(self, keys):
        found: Dict = {}
        remaining = set(keys)

        for i, tier in enumerate(self._tiers):
            if not remaining:
                break
            hits = tier.get_many(remaining)
            if hits:
                found.update(hits)
                # Promote hits to all upper non-static tiers
                for j in range(i):
                    if not self._tiers[j].static:
                        self._tiers[j].set_many(hits)
                remaining -= hits.keys()

        return found

    # ------------------------------------------------------------------
    # Write-through (skip static tiers)
    # ------------------------------------------------------------------

    def set(self, key, value):
        for tier in self._tiers:
            if not tier.static:
                tier.set(key, value)

    def set_many(self, items):
        for tier in self._tiers:
            if not tier.static:
                tier.set_many(items)

    def delete(self, key):
        for tier in self._tiers:
            if not tier.static:
                tier.delete(key)

    def clear(self):
        for tier in self._tiers:
            if not tier.static:
                tier.clear()

    def fill_from_neo4j(self, uri: str, user: str, pwd: str, **kwargs) -> None:
        for tier in self._tiers:
            tier.fill_from_neo4j(uri, user, pwd, **kwargs)
