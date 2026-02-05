from typing import List, Tuple
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore


class Neo4jSampler(BaseSampler):
    """This class defines a samlping method for generating sub-graphs around the seed"""
    def __init__(self, graph_store: GraphStore, num_neighbors: List[int]):
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors # e.g., [10, 5] for 2 hops
        self.query = """
            MATCH (n) WHERE n.id IN $seed_ids
            CALL apoc.path.expandConfig(n, {
                relationshipFilter: "<>",
                minLevel: 1,
                maxLevel: $hops,
                uniqueness: "RELATIONSHIP_PATH",
                limit: $limit
            }) YIELD path
            WITH nodes(path) AS ns
            UNWIND range(0, size(ns)-2) AS i
            RETURN ns[i].id AS src, ns[i+1].id AS dst
            """

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seeds_list = seeds.tolist()
        seed_time = getattr(ns_input, "time", None)
        # For a 2-hop sampler, num_neighbors would be [n, m]
        
        total_hops = len(self.num_neighbors)
        limit = 1
        for n in self.num_neighbors:
            limit *= n

        if hasattr(self.graph_store, "sample_from_nodes"):
            unique_nodes, edge_index_local = self.graph_store.sample_from_nodes(seeds_list, total_hops, limit, self.query)
            return SamplerOutput(
                node=unique_nodes,               # These Global IDs go to the FeatureStore
                row=edge_index_local[0],         # Local source indices (0 to N-1)
                col=edge_index_local[1],         # Local target indices (0 to N-1)
                edge=None,
                batch=None,
                metadata=(seeds, seed_time)
            )
        else:
            raise NotImplementedError("Neo4jSampler.sample_from_nodes: GraphStore lacks 'sample_from_nodes'.")
        
    def sample_from_edges(self, index, neg_sampling = None):
        pass