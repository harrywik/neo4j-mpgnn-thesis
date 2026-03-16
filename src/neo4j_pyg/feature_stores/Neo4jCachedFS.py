from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional
import sys

import numpy as np
import torch
from neo4j import Driver
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS
from benchmarking_tools import Measurer


class Neo4jCachedFS(Neo4jAbstractFS):
    """Cached Neo4j feature store that implements :class:`Neo4jAbstractFS`.

    Combines two cache layers:

    1. **Hot cache** — static, filled once at construction using GDS PageRank
       to pre-load the top-K most-connected nodes.
    2. **LRU cache** — dynamic :class:`~collections.OrderedDict` evicting the
       least-recently-used entry when ``cache_size`` is exceeded.

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
        cache_size: int = 3000,
        hot_cache_size: Optional[int] = None,
    ) -> None:
        # Initialise caches before super().__init__() so that _measure_rtt()
        # (called by the base __init__) can safely access instance attributes.
        self._labels: Dict[str, int] = dict(label_map) if label_map else {}

        self.hot_cache: Dict[int, np.ndarray] = {}
        self.hot_label_cache: Dict[int, int] = {}
        self.cache: OrderedDict = OrderedDict()
        self.label_cache: OrderedDict = OrderedDict()
        self.cache_size = cache_size
        self._hot_cache_size = hot_cache_size

        # Base class sets all shared properties and calls _measure_rtt().
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

        resolved_hot = hot_cache_size if hot_cache_size is not None else cache_size // 3
        self._prefill_hot_cache(graph_name="hot_cache_projection", k=resolved_hot)

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
        self.hot_cache.clear()
        self.hot_label_cache.clear()

        orientation = "UNDIRECTED" if undirected else "NATURAL"
        exists_query = "CALL gds.graph.exists($name) YIELD exists"
        project_query = """
        CALL gds.graph.project(
            $name,
            '*',
            {
              ALL: {
                type: '*',
                orientation: $orientation
              }
            }
        )
        YIELD graphName
        """
        # Only fetch features from the PageRank query — never trust GDS for labels,
        # because gds.util.asNode property access can return mismatched label values.
        # Labels are fetched separately via a direct MATCH query after the feature
        # hot-cache is filled, guaranteeing correct property-to-node association.
        pagerank_query = f"""
        CALL gds.pageRank.stream('{graph_name}')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        ORDER BY score DESC LIMIT $limit
        RETURN n.{self.nodeid_property} AS id,
               n.{self.feature_property}  AS embedding
        """
        label_query = (
            f"MATCH (n) WHERE n.{self.nodeid_property} IN $node_ids "
            f"RETURN n.{self.nodeid_property} AS id, n.{self.target_property} AS label"
        )
        drop_query = "CALL gds.graph.drop($name)"

        projected_here = False
        try:
            with self._get_driver().session(database=self.database_name) as session:
                exists = session.run(exists_query, name=graph_name).single()["exists"]
                if not exists:
                    session.run(project_query, name=graph_name, orientation=orientation)
                    projected_here = True

                for record in session.run(pagerank_query, limit=k):
                    nid = record["id"]

                    if self.feature_property_type == "byte[]":
                        feat = np.frombuffer(bytes(record["embedding"]), dtype=np.float32).copy()
                    elif self.feature_property_type == "f64[]":
                        feat = np.asarray(record["embedding"], dtype=np.float32)
                    else:
                        raise ValueError(
                            f"Unsupported feature_property_type: {self.feature_property_type!r}"
                        )
                    self.hot_cache[nid] = feat

                    if len(self.hot_cache) >= k:
                        break

                if drop_graph and projected_here:
                    session.run(drop_query, name=graph_name)

                # Fetch labels directly — one query, correct property association.
                hot_nids = list(self.hot_cache.keys())
                for record in session.run(label_query, node_ids=hot_nids):
                    nid = record["id"]
                    label_val = record["label"]
                    if isinstance(label_val, str):
                        if label_val not in self._labels:
                            self._labels[label_val] = len(self._labels)
                        self.hot_label_cache[nid] = self._labels[label_val]
                    else:
                        self.hot_label_cache[nid] = int(label_val)

        except Exception as exc:
            raise RuntimeError(
                "GDS is unavailable or graph projection failed."
            ) from exc

        print(f"Hot cache prefilled with {len(self.hot_cache)} nodes.")

    # ------------------------------------------------------------------
    # Neo4jAbstractFS abstract method implementations
    # ------------------------------------------------------------------

    def _get_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> Optional[object]:
        """Return the cached value for *nid*, or ``None`` if not cached."""
        is_label = attr.attr_name == "y"
        hot = self.hot_label_cache if is_label else self.hot_cache
        lru = self.label_cache if is_label else self.cache

        if nid in hot:
            return hot[nid]
        if nid in lru:
            lru.move_to_end(nid)
            return lru[nid]
        return None

    def _update_cached_value(
        self, nid: int, value: object, attr: TensorAttr, **kwargs
    ) -> None:
        """Insert *value* for *nid* into the LRU cache, evicting the oldest if full."""
        is_label = attr.attr_name == "y"
        lru = self.label_cache if is_label else self.cache
        lru[nid] = value
        if len(lru) > self.cache_size:
            lru.popitem(last=False)

    def _remove_cached_value(
        self, nid: int, attr: TensorAttr, **kwargs
    ) -> None:
        """Remove *nid* from both hot and LRU caches (no-op if absent)."""
        is_label = attr.attr_name == "y"
        hot = self.hot_label_cache if is_label else self.hot_cache
        lru = self.label_cache if is_label else self.cache
        hot.pop(nid, None)
        lru.pop(nid, None)

