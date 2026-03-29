from pathlib import Path
from typing import Dict, Optional
import sys

from neo4j import Driver
from torch_geometric.data.feature_store import TensorAttr

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
        )

        self._cache = Neo4jTwoLevelCache(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            database_name=self.database_name,
            nodeid_property=self.nodeid_property,
            feature_property=self.feature_property,
            target_property=self.target_property,
            feature_property_type=self.feature_property_type,
            label_map=label_map,
            cache_size_GB=cache_size_GB,
            prefill=False,
        )
        self.cache_size = self._cache.cache_size
        self._labels = self._cache._labels
        self.hot_cache = self._cache.hot_cache
        self.hot_label_cache = self._cache.hot_label_cache
        self.cache = self._cache.cache
        self.label_cache = self._cache.label_cache

        self._prefill_hot_cache(graph_name="hot_cache_projection", k=self._cache.cache_size // 3)

    # ------------------------------------------------------------------
    # Hot-cache prefill (GDS PageRank)
    # ------------------------------------------------------------------

    def _prefill_hot_cache(
        self,
        graph_name: str,
        k: int = 500,
        undirected: bool = True,
        drop_graph: bool = True,
    ) -> None:
        """Fill the static hot cache with the top-K nodes ranked by PageRank."""
        if undirected is not True:
            raise ValueError("Neo4jTwoLevelCache currently only supports undirected hot-cache prefill")
        if drop_graph is not True:
            raise ValueError("Neo4jTwoLevelCache currently always drops temporary hot-cache projections")

        self._cache.prefill_hot_cache(graph_name=graph_name, k=k)
        print(f"Hot cache prefilled with {len(self.hot_cache)} nodes.")

    # ------------------------------------------------------------------
    # Neo4jAbstractFS abstract method implementations
    # ------------------------------------------------------------------

    def _get_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> Optional[object]:
        """Return the cached value for *nid*, or ``None`` if not cached."""
        return self._cache.get((attr.attr_name, nid))

    def _update_cached_value(
        self, nid: int, value: object, attr: TensorAttr, **kwargs
    ) -> None:
        """Insert *value* for *nid* into the LRU cache, evicting the oldest if full."""
        self._cache[(attr.attr_name, nid)] = value

    def _remove_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> None:
        """Remove *nid* from both hot and LRU caches (no-op if absent)."""
        self._cache.delete((attr.attr_name, nid))

