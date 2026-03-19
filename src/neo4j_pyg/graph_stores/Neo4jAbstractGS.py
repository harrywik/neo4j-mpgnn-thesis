import time
import torch
from typing import Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver
from neo4j import GraphDatabase
import atexit
import sys
from pathlib import Path
from abc import abstractmethod, ABC

_GS_DIR = Path(__file__).resolve().parent.parent
if str(_GS_DIR) not in sys.path:
    sys.path.insert(0, str(_GS_DIR))

from benchmarking_tools import Measurer
from benchmarking_tools.QueryProfileAccumulator import QueryProfileAccumulator


class Neo4jAbstractGS(GraphStore, ABC):
    """Abstract base class for Neo4j graph stores.

    Provides a common interface for Neo4j graph stores.
    """

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        pwd: str = None,
        measurer: Optional[Measurer] = None,
        database_name: str = None,
        dataset_name: str = "neo4j",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
        profile_accumulator: Optional[QueryProfileAccumulator] = None,
    ):
        super().__init__()
        self._driver: Optional[Driver] = None
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.database_name = database_name if database_name else dataset_name
        self.dataset_name = dataset_name
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property
        self.measurer = measurer
        self.profile_accumulator = profile_accumulator
    
    @abstractmethod
    def _get_driver(self):
        pass

    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def get_split(self, limit: int | None = None, offset: int | None = None, split: str = "train", shuffle: bool = False) -> torch.Tensor:   
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
        LIMIT toInteger(coalesce($limit, 9223372036854775807))
        SKIP toInteger(coalesce($offset, 0))
        RETURN n.{self.nodeid_property} AS id
        """

        with self._get_driver().session(database=self.database_name) as session:
            result = session.run(query, limit=limit, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, kwargs, query: str):
        with self._get_driver().session(database=self.database_name) as session:
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

        Sub-phase timings (query dispatch, first-record latency, transfer)
        are logged to ``self.measurer`` when it is set.  When the query was
        issued with the ``PROFILE`` keyword and a
        :class:`~benchmarking_tools.QueryProfileAccumulator` is attached, the
        plan profile is accumulated for later averaging.

        Returns the record dict, or ``None`` if the query produced no rows.
        """
        with self._get_driver().session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, **kwargs)

            # Consume all records first — the driver only populates
            # summary.profile after every record has been received.
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        # Client-side wall time: send → all records received (includes network + DB exec).
        total_topo_fetch_ms = (t_all_records - t_send) * 1000

        if self.measurer is not None:
            self.measurer.log_event("topo_fetch_ms", total_topo_fetch_ms)

        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "sampler", t_send, t_all_records)

        return records[0] if records else None
            
    
    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass