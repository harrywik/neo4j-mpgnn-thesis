"""Redis-backed LRU Neo4j feature store.

Convenience wrapper around :class:`Neo4jFS` + :class:`Neo4jLRUCache`.
DB misses are written back to Redis automatically via
:meth:`Neo4jLRUCache.set_many`, called by :class:`Neo4jFS`.
"""

from __future__ import annotations

from typing import Dict, Optional

from neo4j import Driver
from benchmarking_tools import Measurer

from neo4j_pyg.feature_caches.Neo4jLRUCache import Neo4jLRUCache
from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS


class Neo4jRedisCachedFS(Neo4jFS):
    """Neo4j feature store backed by a Redis LRU cache.

    DB misses are written back to Redis automatically so subsequent
    requests for the same node are served from cache.

    Parameters
    ----------
    redis_url:
        Redis connection URL.
    redis_key_prefix:
        Namespace prefix for all Redis keys.
    redis_ttl_seconds:
        Optional TTL for cached entries.  ``None`` → keys never expire.
    maxmemory:
        Redis memory budget (e.g. ``"2gb"``).  When set, configures
        ``maxmemory`` and ``maxmemory-policy allkeys-lru`` on the server.
    max_entries:
        Soft per-process entry cap.  ``None`` means unlimited.
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
        redis_url: str = "redis://localhost:6379/0",
        redis_key_prefix: str = "fs",
        redis_ttl_seconds: Optional[int] = None,
        maxmemory: Optional[str] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        cache = Neo4jLRUCache(
            redis_url=redis_url,
            key_prefix=redis_key_prefix,
            ttl_seconds=redis_ttl_seconds,
            maxmemory=maxmemory,
            max_entries=max_entries,
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
        self._init_label_map()

    def _init_label_map(self) -> None:
        """Query all distinct labels once and fix a deterministic mapping.

        Only runs when labels are strings (e.g. Cora).  Integer labels
        (OGB datasets) need no mapping and are skipped.  Required for Redis
        because labels are stored as ints — without a stable mapping, cached
        labels could decode to wrong class indices across runs.
        """
        query = (
            f"MATCH (n) WHERE n.{self.target_property} IS NOT NULL "
            f"RETURN DISTINCT n.{self.target_property} AS label LIMIT 1"
        )
        with self._get_driver().session(database=self.database_name) as session:
            sample = session.run(query).single()

        if sample is None or not isinstance(sample["label"], str):
            return

        all_labels_query = (
            f"MATCH (n) WHERE n.{self.target_property} IS NOT NULL "
            f"RETURN DISTINCT n.{self.target_property} AS label ORDER BY label"
        )
        with self._get_driver().session(database=self.database_name) as session:
            records = list(session.run(all_labels_query))

        self._labels.update({rec["label"]: i for i, rec in enumerate(records)})
