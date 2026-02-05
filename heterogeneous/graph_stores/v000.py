import torch
from typing import List, Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver


class Neo4jGraphStore(GraphStore):
    def __init__(self, driver: Driver):
        super().__init__()
        self.driver = driver

    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def get_split(self, n: int | None = None, offset: int | None = None, split: str = "train", shuffle: bool = False) -> torch.Tensor:   
        shuffle_clause = "ORDER BY rand()" if shuffle else "ORDER BY n.id ASC"

        assert not shuffle or (shuffle and offset is None), "Offset together with shuffle does not make sense"

        query = """
        MATCH (n { split: $split })
        """ + shuffle_clause + """
        LIMIT toInteger(coalesce($n, 9223372036854775807))
        SKIP toInteger(coalesce($offset, 0))
        RETURN n.id AS id
        """

        with self.driver.session() as session:
            result = session.run(query, n=n, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, seeds_list:List[int], total_hops:int, limit:int, query:str):
        with self.driver.session() as session:
            result = session.run(query, seed_ids=seeds_list, hops=total_hops, limit=limit)
            
            # Extract edges and format for PyG
            edges = [[r["src"], r["dst"]] for r in result]
            edge_index_global = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        unique_nodes, local_indices = torch.unique(edge_index_global, return_inverse=True)    
        edge_index_local = local_indices.view(2, -1)
        return unique_nodes, edge_index_local
            
    
    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass