from pathlib import Path
import sys

import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import Driver
from typing import Optional, List, Tuple, Dict
import numpy as np

# Allow running this file directly by adding GNN-implementations to sys.path
GNN_IMPL_DIR = Path(__file__).resolve().parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from benchmarking_tools import Measurer

class NoCacheFeatureStore(FeatureStore):
    def __init__(self, driver: Driver, measurer:Measurer = None, dataset_name:str = "neo4j", feature_property:str = "features", target_property:str = "category", split_property_name:str = "split", split_property_type:str = "int", nodeid_property:str = "nodeId", feature_property_type:str = "f64[]") -> None:
        super().__init__()
        self.driver = driver
        self._feat: Dict[Tuple[Optional[NodeType], str], torch.Tensor] = {}
        self._labels = {}
        self.measurer = measurer
        self.dataset_name = dataset_name
        self.feature_property = feature_property
        self.target_property = target_property
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property
        self.feature_property_type = feature_property_type

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        prop = self.target_property if attr.attr_name == "y" else self.feature_property

        # `subject` is hardcoded as the categorical target for now
        query = f"""
        MATCH (n)
        WHERE n.{self.nodeid_property} IN $node_ids
        RETURN n.{self.nodeid_property} AS id, n.{prop} AS value
        """

        if self.measurer:
            self.measurer.log_event("cache_hit", 0)
            self.measurer.log_event("cache_miss", len(node_ids))
            self.measurer.log_event("remote_feature_fetch", 1)
        with self.driver.session(database=self.dataset_name) as session:
            result = session.run(query, node_ids=node_ids)
            data_map = {}
            if prop == self.feature_property:
                for record in result:
                    raw_blob = record["value"]
                    
                    # Asumption embedding is saved as float32 in DB
                    if self.feature_property_type == "byte[]":
                        data_map[record["id"]] = np.frombuffer(raw_blob, dtype=np.float32).copy()
                    elif self.feature_property_type == "f64[]":
                        data_map[record["id"]] = np.asarray(raw_blob, dtype=np.float32)
                    else:
                        raise ValueError("feat wasn't assigned a value")
            else:
                data_map = {r["id"]: r["value"] for r in result}
        if self.measurer:
            self.measurer.log_event("remote_feature_recieved", 1)     
        # Reconstruct in correct order
        ordered_list = [data_map[i] for i in node_ids]
        if prop == self.feature_property:
            # np.stack is faster than torch.stack for numpy inputs
            final_array = np.stack(ordered_list)
            
            return torch.from_numpy(final_array)
        
        return torch.tensor(ordered_list, dtype=torch.int64)

    @staticmethod
    def _key(attr: TensorAttr) -> Tuple[Optional[NodeType], str]:
        return (attr.group_name, attr.attr_name)

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        pass

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"Tensor not found for {attr}")
        return tuple(out.size())

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        # This tells PyG: "I have features (x) and labels (y) for nodes."
        return [
            TensorAttr(group_name=None, attr_name='x'),
            TensorAttr(group_name=None, attr_name='y')
        ]