"""Neo4jSIGNSampler — SIGN-style multi-hop aggregation via the Java UDP.

Calls ``gnnProcedures.aggregation.sign.multiHop`` which returns one row per (seed, hop):
  - hop 0: seed's own feature vector
  - hop 1: mean of 1-hop incoming neighbours
  - hop k: mean of k-hop shell

Results are stored in ``self.pending_sign``:
    {nodeId: [array_hop0, array_hop1, ..., array_hopK]}

A companion ``Neo4jSIGNFeatureStore`` reads this dict and concatenates the
hop arrays to form the SIGN input tensor [x_0 || x_1 || ... || x_k].
"""

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput

from benchmarking_tools import Measurer


class Neo4jSIGNSampler(BaseSampler):
    """Calls ``gnnProcedures.aggregation.sign.multiHop`` for each mini-batch.

    Parameters
    ----------
    graph_store:
        A ``Neo4jAbstractGS`` instance (used for its driver and database_name).
    node_id_key:
        Node property holding the application-level integer ID.
    feature_key:
        Feature property name (e.g. ``"embedding_bytes"``).
    feature_type:
        ``"byte[]"`` or ``"f64[]"``.
    node_label:
        Neo4j node label used for the index lookup.
    edge_type:
        Relationship type to traverse.  ``""`` = any type.
    hops:
        Number of hops (k).  Results will have k+1 entries per seed.
    max_neighbors_per_hop:
        Maximum neighbours sampled per node at each hop.  ``-1`` = no limit.
    measurer:
        Optional ``Measurer`` for timing instrumentation.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        node_id_key: str = "id",
        feature_key: str = "embedding_bytes",
        feature_type: str = "byte[]",
        node_label: str = "Paper",
        edge_type: str = "",
        hops: int = 2,
        max_neighbors_per_hop: int = 10,
        measurer: Optional[Measurer] = None,
    ):
        self.graph_store = graph_store
        self.node_id_key = node_id_key
        self.feature_key = feature_key
        self.feature_type = feature_type
        self.node_label = node_label
        self.edge_type = edge_type
        self.hops = hops
        self.max_neighbors_per_hop = max_neighbors_per_hop
        self.measurer = measurer

        # Populated by sample_from_nodes(), consumed by Neo4jSIGNFeatureStore.
        # Structure: {nodeId: [array_hop0, array_hop1, ..., array_hopK]}
        self.pending_sign: Dict[int, List[np.ndarray]] = {}

        self._cypher = (
            "CALL gnnProcedures.aggregation.sign.multiHop("
            "   $seed_ids,"
            "   $node_id_key,"
            "   $feature_key,"
            "   $feature_type,"
            "   $node_label,"
            "   $edge_type,"
            "   $hops,"
            "   $max_neighbors_per_hop"
            ") YIELD nodeId, hop, aggregatedFeatures"
            " RETURN nodeId, hop, aggregatedFeatures"
            " ORDER BY nodeId, hop"
        )

    # ------------------------------------------------------------------
    # BaseSampler interface
    # ------------------------------------------------------------------

    def sample_from_nodes(
        self, index: NodeSamplerInput, **kwargs
    ) -> SamplerOutput:
        seeds: torch.Tensor = index.node

        if self.measurer is not None:
            self.measurer.log_event("start_sampling", 1)
            self.measurer.set_phase("sampling")

        t_start = time.monotonic()

        params = {
            "seed_ids":              seeds.tolist(),
            "node_id_key":           self.node_id_key,
            "feature_key":           self.feature_key,
            "feature_type":          self.feature_type,
            "node_label":            self.node_label,
            "edge_type":             self.edge_type,
            "hops":                  self.hops,
            "max_neighbors_per_hop": self.max_neighbors_per_hop,
        }

        driver = self.graph_store._get_driver()
        db = self.graph_store.database_name

        with driver.session(database=db, fetch_size=-1) as session:
            t_send = time.monotonic()
            result = session.run(self._cypher, **params)
            records = list(result)
            t_recv = time.monotonic()
            result.consume()

        if self.measurer is not None:
            self.measurer.log_event("sign_udp_ms", (t_recv - t_send) * 1000)
            self.measurer.log_event("sign_udp_records", len(records))

        # Group rows by nodeId, ordered by hop.
        hop_data: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        for rec in records:
            nid = int(rec["nodeId"])
            h = int(rec["hop"])
            feats = rec["aggregatedFeatures"]
            if feats:
                hop_data[nid][h] = np.array(feats, dtype=np.float32)

        # Build pending_sign: list of arrays indexed by hop (0..hops).
        self.pending_sign = {}
        for nid, hop_map in hop_data.items():
            arrays: List[np.ndarray] = []
            for h in range(self.hops + 1):
                arr = hop_map.get(h)
                if arr is not None:
                    arrays.append(arr)
            if len(arrays) == self.hops + 1:
                self.pending_sign[nid] = arrays

        if self.measurer is not None:
            self.measurer.log_event("end_sampling", 1)
            self.measurer.log_event("sampling_ms", (time.monotonic() - t_start) * 1000)

        empty_edge = torch.zeros((2, 0), dtype=torch.long)
        return SamplerOutput(
            node=seeds,
            row=empty_edge[0],
            col=empty_edge[1],
            edge=None,
            batch=None,
            metadata=(seeds, None),
        )

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Neo4jSIGNSampler supports node sampling only.")
