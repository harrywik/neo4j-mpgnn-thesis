import torch
from typing import List, Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver

from typing import List, Tuple
from torch_geometric.data.graph_store import EdgeAttr, EdgeLayout

class Neo4jHeteroGraphStore(GraphStore):
    def __init__(self, driver, database="dblp"):
        super().__init__()
        self.driver = driver
        self.database = database
        self.meta = self._discover_schema()

    def _discover_schema(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        node_types = set()
        edge_triplets = []

        # This query gets all relationship definitions
        query = """
        CALL apoc.meta.data() 
        YIELD label, property, type, other
        WHERE type = 'RELATIONSHIP'
        RETURN label AS src, property AS rel, other AS dst_list
        """

        with self.driver.session(database=self.database) as session:
            result = session.run(query)
            for record in result:
                src = record["src"]
                rel = record["rel"]
                # 'other' is a list of labels the relationship connects to
                for dst in record["dst_list"]:
                    node_types.add(src)
                    node_types.add(dst)
                    edge_triplets.append((src, rel, dst))
        
        return {
            "node_types": list(node_types),
            "edge_triplets": edge_triplets
        }

    def get_all_edge_attrs(self) -> List[EdgeAttr]:
        return [
            EdgeAttr(edge_type=triplet, layout=EdgeLayout.COO) 
            for triplet in self.meta["edge_triplets"]
        ]
