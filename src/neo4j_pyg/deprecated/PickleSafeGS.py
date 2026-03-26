import torch
from typing import List, Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver
from neo4j import GraphDatabase
import atexit


class PickleSafeGS(GraphStore):
    """Pickle safe implementation of GraphStore that connects to Neo4j on demand and closes connection on exit. Does not cache any graph data, but can be used together with a sampler that does."""
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        dataset_name: str = "neo4j",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
    ):
        super().__init__()
        self._driver = None
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.dataset_name = dataset_name
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property

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

    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def get_split(self, n: int | None = None, offset: int | None = None, split: str = "train", shuffle: bool = False) -> torch.Tensor:   
        shuffle_clause = "ORDER BY rand()" if shuffle else f"ORDER BY n.{self.nodeid_property} ASC"

        assert not shuffle or (shuffle and offset is None), "Offset together with shuffle does not make sense"

        split_map = {"train": 0, "val": 1, "test": 2}
        if self.split_property_type == "int":
            split = split_map[split]
        elif self.split_property_type != "str":
            raise ValueError(f"Unsupported split property type: {self.split_property_type}")

        query = f"""
        MATCH (n {{ {self.split_property_name}: $split }})
        """ + shuffle_clause + f"""
        LIMIT toInteger(coalesce($n, 9223372036854775807))
        SKIP toInteger(coalesce($offset, 0))
        RETURN n.{self.nodeid_property} AS id
        """

        with self._get_driver().session(database=self.dataset_name) as session:
            result = session.run(query, n=n, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, kwargs, query: str):
        with self._get_driver().session(database=self.dataset_name) as session:
            result = session.run(query, **kwargs)
            edges = [[r["src"], r["dst"]] for r in result]

        if len(edges) == 0:
            empty = torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor([], dtype=torch.long), empty

        edge_index_global = torch.tensor(edges, dtype=torch.long).t().contiguous()
        unique_nodes, local_indices = torch.unique(edge_index_global, return_inverse=True)
        edge_index_local = local_indices.view(2, -1)
        return unique_nodes, edge_index_local

    def fetch_ordered_subgraph(self, query: str, kwargs: dict) -> dict | None:
        """Execute a subgraph-sampling query and return its single result record.

        Used by :class:`Neo4jNeighborSampler`, whose Cypher query returns one
        record containing ``ordered_nodes`` (global IDs in encounter order) and
        ``edge_pairs`` (list of ``[src_id, dst_id]``), rather than one row per
        edge as in :meth:`sample_from_nodes`.

        Sub-phase timings (query dispatch, first-record latency, transfer,
        ETL) are logged to ``self.measurer`` when it is set.

        Returns the record dict, or ``None`` if the query produced no rows.
        """
        with self._get_driver().session(database=self.dataset_name) as session:
            result = session.run(query, **kwargs)
            record = result.single()

        return record
            
    
    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass