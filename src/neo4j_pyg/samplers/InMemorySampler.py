from typing import List, Optional
import torch
from torch import Tensor
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
from torch_geometric.data.graph_store import EdgeLayout

class InMemorySampler(BaseSampler):
    def __init__(
        self,
        graph_store,
        edge_type,
        num_neighbors: Optional[List[int]] = None,
        layout: EdgeLayout = EdgeLayout.COO,
        undirected: bool = True,
    ):
        super().__init__()
        self.graph_store = graph_store
        self.edge_type = edge_type
        self.layout = layout
        self.undirected = undirected
        self.num_neighbors = num_neighbors if num_neighbors is not None else [10]

    def sample_from_nodes(self, index: NodeSamplerInput, k_hops: Optional[List[int]] = None) -> SamplerOutput:
        seeds: Tensor = index.node.to(torch.long)  # global node ids
        seed_time = getattr(index, "time", None)  # may be None for non-temporal data

        num_neighbors = k_hops if k_hops is not None else self.num_neighbors
        visited = set(seeds.tolist())
        frontier = seeds.tolist()
        edges: List[tuple[int, int]] = []

        for k in num_neighbors:
            if not frontier:
                break
            next_frontier_set = set()
            for src in frontier:
                neigh = self.graph_store.neighbors(
                    src, edge_type=self.edge_type, layout=self.layout, undirected=self.undirected
                )
                if neigh.numel() == 0:
                    continue
                candidates = [v for v in neigh.tolist() if v not in visited]
                if not candidates:
                    continue
                if k > 0 and len(candidates) > k:
                    idx = torch.randperm(len(candidates))[:k]
                    picked = [candidates[i] for i in idx.tolist()]
                else:
                    picked = candidates

                for dst in picked:
                    edges.append((src, dst))
                    if dst not in visited:
                        visited.add(dst)
                        next_frontier_set.add(dst)
            frontier = list(next_frontier_set)

        nodes = torch.tensor(sorted(visited), dtype=torch.long)
        if nodes.numel() > 0:
            mapping = torch.full((int(nodes.max().item()) + 1,), -1, dtype=torch.long)
            mapping[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
        else:
            mapping = torch.empty(0, dtype=torch.long)

        if edges:
            edge_index_global = torch.tensor(edges, dtype=torch.long).t().contiguous()
            row_local = mapping[edge_index_global[0]].contiguous()
            col_local = mapping[edge_index_global[1]].contiguous()
        else:
            row_local = torch.empty(0, dtype=torch.long)
            col_local = torch.empty(0, dtype=torch.long)

        return SamplerOutput(
            node=nodes,
            row=row_local,
            col=col_local,
            edge=None,
            batch=None,
            num_sampled_nodes=None,
            num_sampled_edges=None,
            orig_row=None,
            orig_col=None,
            metadata=(seeds, seed_time),
        )
    def sample_from_edges(self, *args, **kwargs):
        raise NotImplementedError