import torch
from typing import Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver

class Neo4jGraphStore(GraphStore):
    def __init__(self, driver: Driver):
        super().__init__()
        self.driver = driver

    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass
