from pathlib import Path
from typing import Dict, Optional
import sys

from neo4j import Driver

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_caches import (
    build_two_level_cache,
    prefill_from_pagerank,
)
from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS
from benchmarking_tools import Measurer


class Neo4jCachedFS(Neo4jFS):
    """Cached Neo4j feature store extending :class:`Neo4jFS`.

    Combines two cache layers:

    1. **Hot cache** — static, filled once at construction using GDS PageRank
       to pre-load the top-K most-connected nodes.
    2. **LRU cache** — dynamic :class:`LRUCache` evicting the least-recently-used
       entry when capacity is exceeded.

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
        cache_size_GB: float = 0.00001,
        hot_cache_k: int = 500,
        lru_max_entries: int = 100_000,
    ) -> None:
        cache_db = database_name if database_name else dataset_name

        # Build a driver for the prefill query if only credentials were given.
        if driver is None and uri is not None:
            from neo4j import GraphDatabase
            _drv = GraphDatabase.driver(uri, auth=(user, pwd))
        else:
            _drv = driver

        hot_cache = prefill_from_pagerank(
            driver=_drv,
            database_name=cache_db,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            feature_property_type=feature_property_type,
            k=hot_cache_k,
            label_map=label_map,
        )
        cache = build_two_level_cache(
            lru_max_entries=lru_max_entries,
            hot_cache=hot_cache,
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

