"""Two-level cache: static hot tier + LRU spillover.

Built by composing :class:`StaticCache` and :class:`LRUCache` via
:class:`TieredCache`.  The ``prefill_from_pagerank`` helper populates the
hot tier by querying Neo4j GDS PageRank.
"""

import atexit
import os
from typing import Dict, Optional

import numpy as np
from neo4j import Driver, GraphDatabase

from neo4j_pyg.feature_caches.Neo4jCache import Neo4jCache
from neo4j_pyg.feature_caches.LRUCache import LRUCache
from neo4j_pyg.feature_caches.StaticCache import StaticCache
from neo4j_pyg.feature_caches.TieredCache import TieredCache


def prefill_from_pagerank(
    driver: Driver,
    database_name: str,
    nodeid_property: str,
    feature_property: str,
    target_property: str,
    feature_property_type: str,
    k: int,
    graph_name: str | None = None,
    label_map: Dict[str, int] | None = None,
) -> StaticCache:
    """Query Neo4j GDS PageRank and return a frozen :class:`StaticCache`.

    The returned cache contains ``(\"x\", nid)`` and ``(\"y\", nid)`` entries
    for the top-*k* nodes by PageRank score.
    """
    if graph_name is None:
        rank = os.environ.get("RANK", "")
        graph_name = f"hot_cache_projection_{rank}" if rank else "hot_cache_projection"

    labels: Dict[str, int] = dict(label_map) if label_map else {}

    def _normalize_feature(raw) -> np.ndarray:
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return np.frombuffer(bytes(raw), dtype=np.float32).copy()
        return np.asarray(raw, dtype=np.float32)

    def _normalize_label(raw) -> int:
        if raw is None:
            return 0
        if isinstance(raw, str):
            if raw not in labels:
                labels[raw] = len(labels)
            return labels[raw]
        return int(raw)

    exists_query = "CALL gds.graph.exists($name) YIELD exists"
    project_query = """
    CALL gds.graph.project(
        $name, '*',
        { ALL: { type: '*', orientation: $orientation } }
    ) YIELD graphName
    """
    pagerank_query = f"""
    CALL gds.pageRank.stream('{graph_name}')
    YIELD nodeId, score
    WITH gds.util.asNode(nodeId) AS n, score
    ORDER BY score DESC LIMIT $limit
    RETURN n.{nodeid_property} AS id,
           n.{feature_property} AS feature,
           n.{target_property} AS label
    """
    drop_query = "CALL gds.graph.drop($name)"

    data: Dict = {}
    projected_here = False
    try:
        with driver.session(database=database_name) as session:
            exists = session.run(exists_query, name=graph_name).single()["exists"]
            if not exists:
                session.run(project_query, name=graph_name, orientation="UNDIRECTED")
                projected_here = True

            for record in session.run(pagerank_query, limit=k):
                nid = int(record["id"])
                data[("x", nid)] = _normalize_feature(record["feature"])
                data[("y", nid)] = _normalize_label(record["label"])
                if len(data) >= k * 2:
                    break

            if projected_here:
                session.run(drop_query, name=graph_name)
    except Exception as exc:
        raise RuntimeError("GDS is unavailable or graph projection failed.") from exc

    cache = StaticCache(data)
    cache.freeze()
    return cache


def build_two_level_cache(
    lru_max_entries: int,
    hot_cache: StaticCache | None = None,
) -> Neo4jCache:
    """Construct a two-level cache from an optional hot tier + LRU.

    If *hot_cache* is ``None`` the result is a plain :class:`LRUCache`.
    """
    lru = LRUCache(lru_max_entries)
    if hot_cache is None:
        return lru
    return TieredCache([hot_cache, lru])
