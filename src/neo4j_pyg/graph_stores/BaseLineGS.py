import torch
from typing import List, Optional
from torch_geometric.data.graph_store import GraphStore, EdgeAttr
from neo4j import Driver
import json
from pathlib import Path

class BaseLineGS(GraphStore):
    def __init__(self, driver: Driver, database_name:str = None, dataset_name:str = "neo4j", feature_property:str = "features", target_property:str = "category", split_property_name:str = "split", split_property_type:str = "int", nodeid_property:str = "nodeId"):
        super().__init__()
        self.driver = driver
        self.feature_property = feature_property
        self.target_property = target_property
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property
        self.dataset_name = dataset_name
        self.database_name = database_name if database_name else dataset_name


    def _get_edge_index(self, attr: EdgeAttr) -> Optional[torch.Tensor]:
        pass

    def get_split(self, split:str, n: int | None = None, offset: int | None = None, shuffle: bool = False) -> torch.Tensor:   
        shuffle_clause = "ORDER BY rand()" if shuffle else f"ORDER BY n.{self.nodeid_property} ASC"

        assert not shuffle or (shuffle and offset is None), "Offset together with shuffle does not make sense"
        
        split_map = {"train": 0, "val":1, "test":2}
        
        if self.split_property_type == "int":
            split = split_map[split]
        elif self.split_property_type != "str":
            raise ValueError(f"Unsupported split property type: {self.split_property_type}")

        query = """
        MATCH (n { """ + self.split_property_name + """: $split })
        """ + shuffle_clause + f"""
        LIMIT toInteger(coalesce($n, 9223372036854775807))
        SKIP toInteger(coalesce($offset, 0))
        RETURN n.{self.nodeid_property} AS id
        """
                
        with self.driver.session(database=self.database_name) as session:
            result = session.run(query, n=n, split=split, offset=offset)
            seed_ids = [record["id"] for record in result]

        return torch.tensor(seed_ids, dtype=torch.int64)
    
    def sample_from_nodes(self, kwargs, query:str):
        with self.driver.session(database=self.database_name) as session:
            result = session.run(query, **kwargs)
            
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