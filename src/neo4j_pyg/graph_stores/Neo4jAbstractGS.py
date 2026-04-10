import time
import numpy as np
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

    def get_split_by_partition(
        self,
        split: str = "train",
        partition_id: int = 0,
        limit: int | None = None,
        partition_property: str = "rankId",
    ) -> torch.Tensor:
        """Return train/val/test node IDs belonging to one METIS partition.

        Each worker calls this independently with its own rank as partition_id,
        so no cross-rank broadcast is needed.
        """
        split_map = {"train": 0, "val": 1, "test": 2}
        split_val = split_map[split] if self.split_property_type == "int" else split

        query = f"""
        MATCH (n {{ {self.split_property_name}: $split, {partition_property}: $partition_id }})
        ORDER BY n.{self.nodeid_property} ASC
        LIMIT toInteger(coalesce($limit, 9223372036854775807))
        RETURN n.{self.nodeid_property} AS id
        """

        with self._get_driver().session(database=self.database_name, fetch_size=-1) as session:
            result = session.run(query, split=split_val, partition_id=partition_id, limit=limit)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)

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

        with self._get_driver().session(database=self.database_name, fetch_size=-1) as session:
            result = session.run(query, limit=limit, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, query: str, kwargs: dict) -> dict | None:
        """Execute a subgraph-sampling query and return its single result record.

        Executes a Cypher query that returns a single record containing
        ``ordered_nodes`` (global IDs in encounter order) and
        ``edge_pairs`` (list of ``[src_id, dst_id]``).

        Sub-phase timings (query dispatch, first-record latency, transfer)
        are logged to ``self.measurer`` when it is set.  When the query was
        issued with the ``PROFILE`` keyword and a
        :class:`~benchmarking_tools.QueryProfileAccumulator` is attached, the
        plan profile is accumulated for later averaging.

        Returns the record dict, or ``None`` if the query produced no rows.
        """
        with self._get_driver().session(database=self.database_name, fetch_size=-1) as session:
            t_send = time.monotonic()
            result = session.run(query, **kwargs)

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


    def build_topo_etl(
        self, record: dict | None, fallback_seeds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a raw ``sample_from_nodes`` record into PyG tensors.

        This is the topology ETL step: pure Python / C++ work that happens
        *after* the Neo4j round-trip completes.  Emits ``start_etl`` /
        ``end_etl`` events and logs ``topo_etl_ms`` so the CPU-timeline plot
        shows topo ETL in the same green band as feature ETL.

        Returns ``(node_tensor, row, col)``.  On an empty result the caller's
        *fallback_seeds* are returned as the node tensor with zero-length
        edge tensors.
        """
        t_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

        if record is None or (not record.get("ordered_nodes") and not record.get("nodes_by_hop")):
            if self.measurer is not None:
                self.measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)
                self.measurer.log_event("end_etl", 1)
                self.measurer.set_phase("sampling")
            empty = torch.zeros(0, dtype=torch.long)
            return fallback_seeds, empty, empty

        if record.get("ordered_nodes"):
            ordered_global_ids = torch.tensor(record["ordered_nodes"], dtype=torch.long)
        else:
            flat = [nid for hop in record["nodes_by_hop"] for nid in hop]
            ordered_global_ids = torch.tensor(flat, dtype=torch.long)
        global_to_local = {
            int(gid): i for i, gid in enumerate(ordered_global_ids.tolist())
        }

        edge_pairs = record.get("edge_pairs") or []
        if edge_pairs:
            row = torch.tensor(
                [global_to_local[e[0]] for e in edge_pairs], dtype=torch.long
            )
            col = torch.tensor(
                [global_to_local[e[1]] for e in edge_pairs], dtype=torch.long
            )
        else:
            row = col = torch.zeros(0, dtype=torch.long)

        if self.measurer is not None:
            self.measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")

        return ordered_global_ids, row, col

    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass