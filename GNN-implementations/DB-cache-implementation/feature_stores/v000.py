import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import Driver
from typing import Optional, List, Tuple, Dict

class Neo4jFeatureStore(FeatureStore):
    def __init__(self, driver: Driver) -> None:
        super().__init__()
        self.driver = driver
        self._feat: Dict[Tuple[Optional[NodeType], str], torch.Tensor] = {}
        self._labels = {}

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        prop = "subject" if attr.attr_name == "y" else "embedding"
        dtype = torch.int64 if attr.attr_name == "y" else torch.float

        # `subject` is hardcoded as the categorical target for now
        query = f"""
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.{prop} AS value
        ORDER BY n.id ASC
        """
        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            if prop == "embedding":
                data_map = {r["id"]: r["value"] for r in result}
            else:
                data_map = {}
                for r in result:
                    label_str = r["value"]
                    if label_str not in self._labels:
                        self._labels[label_str] = len(self._labels)
                    num_label = self._labels[label_str]
                    data_map[r["id"]] = num_label
                
        # Reconstruct in correct order
        tensor_data = [data_map[nid] for nid in node_ids]
        return torch.tensor(tensor_data, dtype=dtype)

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