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
        seeds: Tensor = index.node.to(torch.long)  # global node ids
        seed_time = getattr(index, "time", None)  # may be None for non-temporal data

        node_set = set(seeds.tolist())

        # Expand per hop by querying GraphStore (no direct graph access)
        frontier = seeds.clone()
        for hop_idx, k in enumerate(k_hops, start=1):
            new_nodes = []
            for n in frontier.tolist():
                # get_neighbors returns seed + up-to-hop nodes; request only this hop by k_hop=1 over current frontier
                neigh = self.graph_store.get_neighbors(seed=n, max_neighboor=k, k_hop=1, edge_type=self.edge_type,
                                                       layout=self.layout, undirected=self.undirected)
                for v in neigh.tolist():
                    if v not in node_set:
                        node_set.add(v)
                        new_nodes.append(v)
            if not new_nodes:
                break
            frontier = torch.tensor(new_nodes, dtype=torch.long)

        # Global → local relabeling
        nodes = torch.tensor(sorted(node_set), dtype=torch.long)
        max_id = int(nodes.max().item()) if nodes.numel() > 0 else -1
        mapping = torch.full((max_id + 1,), -1, dtype=torch.long)
        mapping[nodes] = torch.arange(nodes.numel(), dtype=torch.long)

        # Induced edges from GraphStore, relabeled locally
        # After computing `nodes`:
        rc = self.graph_store.get_edge_index(edge_type=self.edge_type, layout=self.layout)
        if rc is None:
            raise ValueError("GraphStore returned no edges for the given edge_type/layout.")
        row_all, col_all = rc

        # Make mapping large enough to index any global node id seen in edges or nodes
        global_max = int(torch.stack([row_all.max(), col_all.max(), nodes.max()]).max().item()) if nodes.numel() > 0 \
             else int(torch.stack([row_all.max(), col_all.max()]).max().item())
        mapping = torch.full((global_max + 1,), -1, dtype=torch.long)
        mapping[nodes] = torch.arange(nodes.numel(), dtype=torch.long)

        # Safe masks and relabel
        src_in = mapping[row_all] != -1
        dst_in = mapping[col_all] != -1
        edge_mask = src_in & dst_in
        row_local = mapping[row_all[edge_mask]].contiguous()
        col_local = mapping[col_all[edge_mask]].contiguous()

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
            metadata=(seeds, seed_time),     # <-- provide both entries
            _seed_node=seeds                 # optional but helpful
        )
    def sample_from_edges(self, *args, **kwargs):
        raise NotImplementedError