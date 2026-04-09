from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple
import os
import sys

import numpy as np
from neo4j import Driver

FS_DIR = Path(__file__).resolve().parent.parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

from neo4j_pyg.feature_caches.Neo4jAbstractCache import Neo4jAbstractCache


class Neo4jTwoLevelCache(Neo4jAbstractCache):
    """Two-level Neo4j cache matching the behavior of Neo4jCachedFS.

    Keys use the form ``(attr_name, nid)`` where ``attr_name`` is typically
    ``"x"`` for features or ``"y"`` for labels.
    """

    def __init__(
        self,
        driver: Optional[Driver] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        database_name: Optional[str] = None,
        nodeid_property: str = "nodeId",
        feature_property: str = "features",
        target_property: str = "category",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict[str, int]] = None,
        cache_size_GB: float = 0.000001,
        prefill: bool = True,
    ) -> None:
        super().__init__(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            database_name=database_name,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            feature_property_type=feature_property_type,
            label_map=label_map,
            cache_size_GB=cache_size_GB,
        )

        self.hot_cache: Dict[int, np.ndarray] = {}
        self.hot_label_cache: Dict[int, int] = {}
        self.cache: OrderedDict[int, object] = OrderedDict()
        self.label_cache: OrderedDict[int, int] = OrderedDict()

        if prefill:
            # Use a per-process projection name so concurrent DDP workers
            # (each with a different RANK env var set by torchrun) don't race
            # to create the same GDS graph projection.
            rank = os.environ.get("RANK", "")
            graph_name = f"hot_cache_projection_{rank}" if rank else "hot_cache_projection"
            self.prefill_hot_cache(graph_name=graph_name, k=self.cache_size // 3)

    @staticmethod
    def _split_key(key: Tuple[str, int]) -> Tuple[str, int]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("Cache keys must be (attr_name, nid) tuples")
        attr_name, nid = key
        if attr_name not in {"x", "y"}:
            raise KeyError(f"Unsupported attr_name {attr_name!r}; expected 'x' or 'y'")
        return attr_name, int(nid)

    def _pick_stores(self, attr_name: str):
        if attr_name == "y":
            return self.hot_label_cache, self.label_cache
        return self.hot_cache, self.cache

    def prefill_hot_cache(self, graph_name: str, k: int) -> None:
        self.hot_cache.clear()
        self.hot_label_cache.clear()

        orientation = "UNDIRECTED"
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
        pagerank_query = f"""
        CALL gds.pageRank.stream('{graph_name}')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        ORDER BY score DESC LIMIT $limit
        RETURN n.{self.nodeid_property} AS id,
               n.{self.feature_property} AS embedding
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
                    nid = int(record["id"])
                    embedding = record["embedding"]
                    feat = self._normalize_feature_value(embedding)
                    self.hot_cache[nid] = feat
                    if len(self.hot_cache) >= k:
                        break

                if projected_here:
                    session.run(drop_query, name=graph_name)

                hot_nids = list(self.hot_cache.keys())
                if hot_nids:
                    for record in session.run(label_query, node_ids=hot_nids):
                        nid = int(record["id"])
                        self.hot_label_cache[nid] = self._normalize_label_value(record["label"])
        except Exception as exc:
            raise RuntimeError("GDS is unavailable or graph projection failed.") from exc

    def get(self, key):
        attr_name, nid = self._split_key(key)
        hot, lru = self._pick_stores(attr_name)

        if nid in hot:
            return hot[nid]
        if nid in lru:
            lru.move_to_end(nid)
            return lru[nid]
        return None

    def set(self, key, value):
        attr_name, nid = self._split_key(key)
        _, lru = self._pick_stores(attr_name)
        lru[nid] = value
        if len(lru) > self.cache_size:
            lru.popitem(last=False)

    def delete(self, key) -> None:
        attr_name, nid = self._split_key(key)
        hot, lru = self._pick_stores(attr_name)
        hot.pop(nid, None)
        lru.pop(nid, None)

    def __delitem__(self, key):
        self.delete(key)

    def clear(self):
        self.hot_cache.clear()
        self.hot_label_cache.clear()
        self.cache.clear()
        self.label_cache.clear()

    def __contains__(self, key) -> bool:
        return self.get(key) is not None
