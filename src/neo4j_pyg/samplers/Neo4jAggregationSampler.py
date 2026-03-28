"""Neo4jAggregationSampler — server-side 1-hop mean aggregation via Java UDP.

Design (Option D hybrid)
------------------------
A single Cypher call invokes ``gnnProcedures.aggregation.neighbor.mean`` which traverses
the graph in the Neo4j JVM, computes the mean of incoming-neighbour feature
vectors for each seed node, and streams back ``(nodeId, aggregatedFeatures)``.

The sampler stores those pre-aggregated vectors in ``self.pending_agg`` (a plain
dict).  A companion ``Neo4jUDPFeatureStore`` reads from this dict instead of
issuing a separate feature-fetch round-trip.

Because the model (``GCNPostAggregation``) applies weight matrices to the
already-aggregated features using PyTorch, the full autograd graph is preserved
and training works normally.

Topology / edge_index are *not* returned (they are not needed when the
aggregation was done server-side).
"""

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput

from benchmarking_tools import Measurer


class Neo4jAggregationSampler(BaseSampler):
    """Calls the ``gnnProcedures.aggregation.neighbor.mean`` Java UDP for each mini-batch.

    Parameters
    ----------
    graph_store:
        A ``Neo4jAbstractGS`` instance (used for its driver and database_name).
    node_id_key:
        The node property that stores the application-level integer node ID
        (e.g. ``"id"`` for Cora, ``"nodeId"`` for arxiv/products).
    feature_key:
        Name of the node property holding feature vectors
        (e.g. ``"embedding_bytes"`` or ``"features"``).
    feature_type:
        ``"byte[]"`` for packed float32 bytes or ``"f64[]"`` for double arrays.
    node_label:
        Neo4j node label used for the indexed property lookup (e.g. ``"Paper"``).
        An index/uniqueness constraint on ``(nodeLabel {nodeIdKey})`` must exist.
    edge_type:
        Relationship type to traverse.  Empty string means *any* relationship.
    max_neighbors:
        Maximum number of neighbours to sample per seed.  ``-1`` = no limit.
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
        max_neighbors: int = 10,
        measurer: Optional[Measurer] = None,
    ):
        self.graph_store = graph_store
        self.node_id_key = node_id_key
        self.feature_key = feature_key
        self.feature_type = feature_type
        self.node_label = node_label
        self.edge_type = edge_type
        self.max_neighbors = max_neighbors
        self.measurer = measurer

        # Shared state: populated by sample_from_nodes(), consumed by
        # Neo4jUDPFeatureStore.get_tensor().
        self.pending_agg: Dict[int, np.ndarray] = {}

        self._cypher = (
            "CALL gnnProcedures.aggregation.neighbor.mean("
            "   $seed_ids,"
            "   $node_id_key,"
            "   $feature_key,"
            "   $feature_type,"
            "   $node_label,"
            "   $edge_type,"
            "   $max_neighbors"
            ") YIELD nodeId, aggregatedFeatures"
            " RETURN nodeId, aggregatedFeatures"
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
            "seed_ids":    seeds.tolist(),
            "node_id_key": self.node_id_key,
            "feature_key": self.feature_key,
            "feature_type": self.feature_type,
            "node_label":  self.node_label,
            "edge_type":   self.edge_type,
            "max_neighbors": self.max_neighbors,
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
            self.measurer.log_event("udp_agg_ms", (t_recv - t_send) * 1000)
            self.measurer.log_event("udp_records", len(records))

        # Parse results and populate pending_agg cache.
        self.pending_agg = {}
        for rec in records:
            nid = int(rec["nodeId"])
            feats = rec["aggregatedFeatures"]
            if feats:
                self.pending_agg[nid] = np.array(feats, dtype=np.float32)

        if self.measurer is not None:
            self.measurer.log_event("end_sampling", 1)
            total_ms = (time.monotonic() - t_start) * 1000
            self.measurer.log_event("sampling_ms", total_ms)

        # Return empty topology — GCNPostAggregation does not use edge_index.
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
        raise NotImplementedError("Neo4jAggregationSampler supports node sampling only.")
