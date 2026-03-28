from typing import List

import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput


class Neo4jJavaNeighborSampler(BaseSampler):
    """Neighbor sampler backed by Java UDP ``gnnProcedures.sampling.neighbor.sample``.

    This is a homogeneous, incoming-edge sampler that returns topology in the
    same shape as ``Neo4jNeighborSampler``:
      - ordered_nodes (global ids in encounter order)
      - edge_pairs ([[src_global_id, dst_global_id], ...])
    """

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        rel_type: str = None,
        node_label: str = None,
        node_id_key: str = "id",
        random_seed: int = 42,
        profile: bool = False,
    ):
        self.graph_store = graph_store
        self.num_neighbors = [int(k) for k in num_neighbors]
        self.rel_type = rel_type or ""
        self.node_label = node_label
        self.nodeid_property = node_id_key or graph_store.nodeid_property
        self.random_seed = int(random_seed)
        self.profile = profile

        self.query = (
            "CALL gnnProcedures.sampling.neighbor.sample("
            "   $seed_ids,"
            "   $node_id_key,"
            "   $node_label,"
            "   $num_neighbors,"
            "   $edge_type,"
            "   $random_seed"
            ") YIELD ordered_nodes, edge_pairs "
            "RETURN ordered_nodes, edge_pairs"
        )

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seed_time = getattr(ns_input, "time", None)

        record = self.graph_store.fetch_ordered_subgraph(
            self.query,
            {
                "seed_ids": seeds.tolist(),
                "node_id_key": self.nodeid_property,
                "node_label": self.node_label,
                "edge_type": self.rel_type,
                "num_neighbors": self.num_neighbors,
                "random_seed": self.random_seed,
            },
        )

        node, row, col = self.graph_store.build_topo_etl(record, seeds)

        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=None,
            batch=None,
            metadata=(seeds, seed_time),
        )

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Neo4jJavaNeighborSampler supports node sampling only.")
