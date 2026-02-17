import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import Driver
from typing import Optional, List, Tuple, Dict
import numpy as np

class NoCacheFeatureStore(FeatureStore):
    def __init__(self, driver: Driver) -> None:
        super().__init__()
        self.driver = driver
        self._feat: Dict[Tuple[Optional[NodeType], str], torch.Tensor] = {}
        self._labels = {}

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        prop = "subject" if attr.attr_name == "y" else "embedding"

        # `subject` is hardcoded as the categorical target for now
        query = f"""
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.{prop} AS value
        ORDER BY n.id ASC
        """
        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            data_map = {}
            if prop == "embedding":
                for record in result:
                    raw_blob = record["value"]
                    
                    # Asumption embedding is saved as float32 in DB
                    feat = np.frombuffer(raw_blob, dtype=np.float32).copy()
                    data_map[record["id"]] = feat
            else:
                for record in result:
                    label_str = record["value"]
                    if label_str not in self._labels:
                        self._labels[label_str] = len(self._labels)
                    num_label = self._labels[label_str]
                    data_map[record["id"]] = num_label
                
        # Reconstruct in correct order
        ordered_list = [data_map[i] for i in node_ids]
        if prop == "embedding":
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