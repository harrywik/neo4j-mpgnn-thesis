from typing import List, Optional
import torch
from torch import Tensor
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
from torch_geometric.data.graph_store import EdgeLayout

class InMemorySampler(BaseSampler):
    def __init__(self, graph_store, edge_type, layout: EdgeLayout = EdgeLayout.COO, undirected: bool = True):
        super().__init__()
        self.graph_store = graph_store
        self.edge_type = edge_type
        self.layout = layout
        self.undirected = undirected

    def sample_from_nodes(self, index: NodeSamplerInput, k_hops: List[int] = [10]) -> SamplerOutput:
        seeds: Tensor = index.node.to(torch.long)
        seed_time = getattr(index, "time", None)

        total_hops = len(k_hops)
        limit = 1
        for n in k_hops:
            limit *= n

        if hasattr(self.graph_store, "sample_from_nodes"):
            unique_nodes, edge_index_local = self.graph_store.sample_from_nodes(
                seeds.tolist(),
                total_hops,
                limit,
                edge_type=self.edge_type,
                layout=self.layout,
                undirected=self.undirected,
            )
            return SamplerOutput(
                node=unique_nodes,
                row=edge_index_local[0],
                col=edge_index_local[1],
                edge=None,
                batch=None,
                metadata=(seeds, seed_time),
            )

        raise NotImplementedError("GraphStore lacks 'sample_from_nodes'.")

    def sample_from_edges(self, *args, **kwargs):
        raise NotImplementedError