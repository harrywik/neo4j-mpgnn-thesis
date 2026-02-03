import torch
from typing import Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from torch_geometric.typing import NodeType
from neo4j import Driver
from typing import Dict, Tuple

class Neo4jGraphStore(GraphStore):
    def __init__(self, driver: Driver):
        super().__init__()
        self.driver = driver

    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def train_val_test_split_db(self, ratios: list[float]) -> None:
        """
        ratios: [train_frac, val_frac, test_frac] with sum eq. 1.0
        """
        
        assert sum(ratios) == 1.0, "ratios must sum to 1.0"
        query = """
        MATCH (n)
        WITH n, rand() AS r
        SET n.split = 
            CASE 
                WHEN r <= $train_tresh THEN 'train'
                WHEN r <= $val_tresh THEN 'val'
                ELSE 'test'
            END
        """

        t, v = ratios[:2]
        v += t

        with self.driver.session() as session:
            session.run(query, train_tresh=t, val_tresh=v)

    def get_random_split(self, n: int, split: str = "train") -> torch.Tensor:
        query = """
        MATCH (n { split: $split })
        ORDER BY rand()
        LIMIT $n
        RETURN n.id AS id
        """

        with self.driver.session() as session:
            result = session.run(query, n=n, split=split)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def _put_edge_index(self):
        pass
    def _remove_edge_index(self):
        pass
    def get_all_edge_attrs(self):
        pass