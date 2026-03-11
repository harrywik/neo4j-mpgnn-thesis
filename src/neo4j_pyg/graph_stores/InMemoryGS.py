from typing import Dict, List, Optional, Tuple
import torch
from torch_geometric.data import GraphStore, EdgeAttr
from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.typing import EdgeTensorType, EdgeType

class InMemoryGS(GraphStore):
    def __init__(self):
        super().__init__()
        self._edges: Dict[Tuple[EdgeType, EdgeLayout], EdgeTensorType] = {}
        self._adj: Dict[Tuple[EdgeType, EdgeLayout], List[List[int]]] = {}
        self._split_indices: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _key(attr: EdgeAttr) -> Tuple[EdgeType, EdgeLayout]:
        return (attr.edge_type, attr.layout)

    def _put_edge_index(self, edge_index: EdgeTensorType, edge_attr: EdgeAttr) -> bool:
        if edge_attr.layout != EdgeLayout.COO:
            raise ValueError("Only COO layout is supported.")
        row, col = edge_index
        if row.dtype != torch.long or col.dtype != torch.long:
            raise TypeError("row/col must be dtype torch.long")
        if row.dim() != 1 or col.dim() != 1 or row.numel() != col.numel():
            raise ValueError("row/col must be 1-D and of equal length")
        self._edges[self._key(edge_attr)] = (row.contiguous(), col.contiguous())
        # Invalidate adjacency for this key
        self._adj.pop(self._key(edge_attr), None)
        return True

    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[EdgeTensorType]:
        return self._edges.get(self._key(edge_attr))

    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        removed = self._edges.pop(self._key(edge_attr), None) is not None
        if removed:
            self._adj.pop(self._key(edge_attr), None)
        return removed
    def set_split_indices(self, split: str, indices: torch.Tensor) -> None:
        if indices is None:
            return
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long)
        if indices.dtype != torch.long:
            indices = indices.to(torch.long)
        self._split_indices[split] = indices.view(-1).contiguous()

    def set_split_masks(
        self,
        train_mask: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        test_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if train_mask is not None:
            self.set_split_indices("train", torch.nonzero(train_mask, as_tuple=False).view(-1))
        if val_mask is not None:
            self.set_split_indices("val", torch.nonzero(val_mask, as_tuple=False).view(-1))
        if test_mask is not None:
            self.set_split_indices("test", torch.nonzero(test_mask, as_tuple=False).view(-1))

    #    node_ids = graph_store.get_split(N, offset=i, split=split, shuffle=False)
    def get_split(self, n: int | None = None, offset: int | None = None, split: str = "train", shuffle: bool = False) -> torch.Tensor:
        assert not shuffle or (shuffle and offset is None), "Offset together with shuffle does not make sense"

        if split not in self._split_indices:
            raise KeyError(
                f"Split '{split}' not found. Call set_split_indices(...) or set_split_masks(...) first."
            )

        indices = self._split_indices[split]
        if indices.numel() == 0:
            return indices

        if shuffle:
            perm = torch.randperm(indices.numel())
            indices = indices[perm]
        else:
            indices = torch.sort(indices).values

        if offset is not None:
            indices = indices[offset:]
        if n is not None:
            indices = indices[:n]

        return indices
        
    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        out: List[EdgeAttr] = []
        for (edge_type, layout), _ in self._edges.items():
            out.append(self._edge_attr_cls(edge_type=edge_type, layout=layout))
        return out

    # ---- Neighbor APIs for samplers ----
    def _ensure_adj(self, edge_type: EdgeType, layout: EdgeLayout = EdgeLayout.COO, undirected: bool = True) -> List[List[int]]:
        key = (edge_type, layout)
        if key in self._adj:
            return self._adj[key]
        rc = self._edges.get(key)
        if rc is None:
            raise KeyError("No edges stored for the given edge_type/layout")
        row, col = rc
        num_nodes = int(torch.max(torch.stack([row, col])).item()) + 1
        adj: List[List[int]] = [[] for _ in range(num_nodes)]
        for r, c in zip(row.tolist(), col.tolist()):
            adj[r].append(c)
            if undirected:
                adj[c].append(r)
        self._adj[key] = adj
        return adj

    def neighbors(self, node: int, edge_type: EdgeType, layout: EdgeLayout = EdgeLayout.COO, undirected: bool = True) -> torch.Tensor:
        adj = self._ensure_adj(edge_type, layout, undirected)
        if node < 0 or node >= len(adj):
            return torch.empty(0, dtype=torch.long)
        lst = adj[node]
        return torch.tensor(lst, dtype=torch.long) if lst else torch.empty(0, dtype=torch.long)

    def get_neighbors(self, seed: int, max_neighboor: int, k_hop: int, edge_type: EdgeType,
                      layout: EdgeLayout = EdgeLayout.COO, undirected: bool = True) -> torch.Tensor:
        # BFS up to k_hop; cap immediate neighbors per expansion by max_neighboor (<=0 means all)
        visited = set([seed])
        frontier = torch.tensor([seed], dtype=torch.long)
        for hop in range(k_hop + 1):
            if hop == 0:
                # 0-hop → the node itself
                continue
            next_nodes: List[int] = []
            for n in frontier.tolist():
                neigh = self.neighbors(n, edge_type=edge_type, layout=layout, undirected=undirected)
                if neigh.numel() == 0:
                    continue
                if max_neighboor > 0 and max_neighboor < neigh.numel():
                    idx = torch.randperm(neigh.numel())[:max_neighboor]
                    neigh = neigh[idx]
                for v in neigh.tolist():
                    if v not in visited:
                        visited.add(v)
                        next_nodes.append(v)
            if not next_nodes:
                break
            frontier = torch.tensor(next_nodes, dtype=torch.long)
        out = torch.tensor(sorted(visited), dtype=torch.long)
        return out

    @classmethod
    def from_data(cls, data, edge_type: Optional[EdgeType] = None) -> "InMemoryGraphStore":
        store = cls()
        row, col = data.edge_index[0].to(torch.long), data.edge_index[1].to(torch.long)
        store.put_edge_index((row, col), edge_type=edge_type, layout=EdgeLayout.COO)
        return store