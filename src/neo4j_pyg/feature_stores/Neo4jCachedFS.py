from pathlib import Path
from typing import Dict, Optional
import sys

from neo4j import Driver

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_caches import Neo4jTwoLevelCache
from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS
from benchmarking_tools import Measurer


class Neo4jCachedFS(Neo4jAbstractFS):
    """Cached Neo4j feature store that implements :class:`Neo4jAbstractFS`.

    Combines two cache layers:

    1. **Hot cache** — static, filled once at construction using GDS PageRank
       to pre-load the top-K most-connected nodes.
     2. **LRU cache** — dynamic :class:`~collections.OrderedDict` evicting the
         least-recently-used entry when the GB-based cache budget is exceeded.

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
    ) -> None:
        cache_db = database_name if database_name else dataset_name
        cache = Neo4jTwoLevelCache(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            database_name=cache_db,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            feature_property_type=feature_property_type,
            label_map=label_map,
            cache_size_GB=cache_size_GB,
            prefill=False,
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

