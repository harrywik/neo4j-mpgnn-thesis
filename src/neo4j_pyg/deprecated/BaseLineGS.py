import time
import torch
from typing import List, Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver
import json
from pathlib import Path
from benchmarking_tools import Measurer
from neo4j_pyg.graph_stores.Neo4jAbstractGS import Neo4jAbstractGS

class BaseLineGS(Neo4jAbstractGS):
    def __init__(self, driver: Driver, database_name:str = None, dataset_name:str = "neo4j", feature_property:str = "features", target_property:str = "category", split_property_name:str = "split", split_property_type:str = "int", nodeid_property:str = "nodeId", measurer:Measurer = None):
        super().__init__()
        self.driver = driver
        self.feature_property = feature_property
        self.target_property = target_property
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property
        self.dataset_name = dataset_name
        self.database_name = database_name if database_name else dataset_name
        # Measurer is set dynamically per-run by the experiment orchestrator
        # (graph store is shared across runs, measurer is per-run).
        self.measurer = measurer


    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def get_split(self, split:str, n: int | None = None, offset: int | None = None, shuffle: bool = False) -> torch.Tensor:   
        shuffle_clause = "ORDER BY rand()" if shuffle else f"ORDER BY n.{self.nodeid_property} ASC"

        assert not shuffle or (shuffle and offset is None), "Offset together with shuffle does not make sense"
        
        split_map = {"train": 0, "val":1, "test":2}
        
        if self.split_property_type == "int":
            split = split_map[split]
        elif self.split_property_type != "str":
            raise ValueError(f"Unsupported split property type: {self.split_property_type}")

        query = """
        MATCH (n { """ + self.split_property_name + """: $split })
        """ + shuffle_clause + f"""
        LIMIT toInteger(coalesce($n, 9223372036854775807))
        SKIP toInteger(coalesce($offset, 0))
        RETURN n.{self.nodeid_property} AS id
        """
                
        with self.driver.session(database=self.database_name) as session:
            result = session.run(query, n=n, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, kwargs, query:str):
        with self.driver.session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, **kwargs)
            t_query_sent = time.monotonic()

            first = result.peek()
            t_first_record = time.monotonic()

            edges = [[r["src"], r["dst"]] for r in result] if first is not None else []
            t_all_records = time.monotonic()

        t_etl_start = time.monotonic()

        if self.measurer is not None:
            self.measurer.log_event("topo_query_sent_ms", (t_query_sent - t_send) * 1000)
            self.measurer.log_event("topo_first_record_ms", (t_first_record - t_query_sent) * 1000)
            self.measurer.log_event("topo_transfer_ms", (t_all_records - t_first_record) * 1000)

        if len(edges) == 0:
            if self.measurer is not None:
                self.measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)
            empty = torch.zeros((2, 0), dtype=torch.long)
            return torch.tensor([], dtype=torch.long), empty

        edge_index_global = torch.tensor(edges, dtype=torch.long).t().contiguous()
        unique_nodes, local_indices = torch.unique(edge_index_global, return_inverse=True)
        edge_index_local = local_indices.view(2, -1)

        if self.measurer is not None:
            self.measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)

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
        with self.driver.session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, **kwargs)
            t_query_sent = time.monotonic()

            first = result.peek()
            t_first_record = time.monotonic()

            record = result.single() if first is not None else None
            t_all_records = time.monotonic()

        if self.measurer is not None:
            self.measurer.log_event("topo_query_sent_ms", (t_query_sent - t_send) * 1000)
            self.measurer.log_event("topo_first_record_ms", (t_first_record - t_query_sent) * 1000)
            self.measurer.log_event("topo_transfer_ms", (t_all_records - t_first_record) * 1000)

        return record

    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass