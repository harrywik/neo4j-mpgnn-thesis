from typing import Dict, Optional

from neo4j import Driver

from neo4j_pyg.feature_caches import Neo4jStaticCache, Neo4jLRURAMCache, TieredCache
from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS
from benchmarking_tools import Measurer


class Neo4jCachedFS(Neo4jFS):
    """Cached Neo4j feature store extending :class:`Neo4jFS`.

    Two cache tiers, both in-memory:

    1. **Static hot cache** — pre-filled at startup with the top-*k* nodes
       ranked by out-degree (backed by Redis for multi-process sharing, with
       in-process promotion so repeated reads are pure RAM).
    2. **LRU RAM cache** — dynamic in-memory LRU that captures every node
       fetched from the DB during training, so repeated accesses are served
       from RAM.

    Pickle-safe: pass ``uri``/``user``/``pwd`` instead of a live ``driver``
    when using PyG's multiprocessing ``DataLoader`` workers.
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
        label_map: Optional[Dict[str, int]] = None,
        hot_cache_k: int = 500,
        lru_max_entries: int = 100,
        redis_url: str = "redis://localhost:6379/0",
    ) -> None:
        cache_db = database_name if database_name else dataset_name

        static = Neo4jStaticCache(redis_url=redis_url)
        static.fill_from_neo4j(
            uri=uri,
            user=user,
            pwd=pwd,
            database=cache_db,
            k=hot_cache_k,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            label_map=label_map,
        )
        lru = Neo4jLRURAMCache(max_entries=lru_max_entries)
        cache = TieredCache([lru, static])

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
