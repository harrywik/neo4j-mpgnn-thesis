from pathlib import Path
import sys
from typing import Optional
from benchmarking_tools import Measurer
from benchmarking_tools.QueryProfileAccumulator import QueryProfileAccumulator
from neo4j import GraphDatabase
import atexit

GS_DIR = Path(__file__).resolve().parent.parent
if str(GS_DIR) not in sys.path:
    sys.path.insert(0, str(GS_DIR))

from neo4j_pyg.graph_stores.Neo4jAbstractGS import Neo4jAbstractGS


class Neo4jMultiGS(Neo4jAbstractGS):
    """Graph store that creates a Neo4j driver lazily per process.

    Safe to use with multiple DataLoader workers (``num_workers > 0``):
    each spawned process reconnects independently via ``uri``/``user``/``pwd``.
    Use :class:`Neo4SingleGS` instead when no worker parallelism is needed.
    """

    def __init__(self, uri: str, user: str, pwd: str, measurer: Optional[Measurer] = None, database_name: str = None, dataset_name: str = "neo4j", split_property_name: str = "split", split_property_type: str = "int", nodeid_property: str = "nodeId", profile_accumulator: Optional[QueryProfileAccumulator] = None):
        super().__init__(uri=uri, user=user, pwd=pwd, measurer=measurer, database_name=database_name, dataset_name=dataset_name, split_property_name=split_property_name, split_property_type=split_property_type, nodeid_property=nodeid_property, profile_accumulator=profile_accumulator)
        self._driver = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_driver"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None
    
    
    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver

    def close(self):
        if getattr(self, "_driver", None) is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None
