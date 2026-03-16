from typing import Optional
from neo4j import Driver
from benchmarking_tools import Measurer
from neo4j_pyg.graph_stores.Neo4jAbstractGS import Neo4jAbstractGS


class Neo4SingleGS(Neo4jAbstractGS):
    """Graph store backed by a pre-existing Neo4j driver.

    Intended for single-process use; the injected driver is returned
    directly and is not safe to share across DataLoader workers.
    Use :class:`Neo4jMultiGS` instead when ``num_workers > 0``.
    """

    def __init__(
        self,
        driver: Driver,
        measurer: Optional[Measurer] = None,
        database_name: str = None,
        dataset_name: str = "neo4j",
        feature_property: str = "features",
        target_property: str = "category",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
    ):
        super().__init__(
            measurer=measurer,
            database_name=database_name,
            dataset_name=dataset_name,
            split_property_name=split_property_name,
            split_property_type=split_property_type,
            nodeid_property=nodeid_property,
        )
        self.driver = driver
        self.feature_property = feature_property
        self.target_property = target_property

    def _get_driver(self) -> Driver:
        return self.driver