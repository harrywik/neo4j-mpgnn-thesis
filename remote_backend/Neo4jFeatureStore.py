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

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        # subject is hardcoded as the categorical target for now
        query = """
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.features AS feats, n.subject AS label
        ORDER BY n.id ASC
        """

        with self.driver.session() as session:
            result = session.run(query, node_ids=node_ids)
            data_map = {r["id"]: (r["feats"], r["label"]) for r in result}
                
        # Reconstruct in correct order
        features = [data_map[i][0] for i in node_ids]
        labels = [data_map[i][1] for i in node_ids]
        
        # If this specific call is for 'y', return labels; else return features
        if attr.name == 'y':
            return torch.tensor(labels, dtype=torch.long)
        return torch.tensor(features, dtype=torch.float)

    @staticmethod
    def _key(attr: TensorAttr) -> Tuple[Optional[NodeType], str]:
        return (attr.group_name, attr.attr_name)

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        if attr.index is not None:
            raise ValueError("Only full-tensor writes supported (attr.index must be None).")
        self._feat[self._key(attr)] = tensor
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[torch.Tensor]:
        tensor = self._feat.get(self._key(attr))
        if tensor is None:
            return None
        return tensor if attr.index is None else tensor[attr.index]

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        if attr.index is not None:
            raise ValueError("Only full-tensor removals supported (attr.index must be None).")
        return self._feat.pop(self._key(attr), None) is not None

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"Tensor not found for {attr}")
        return tuple(out.size())

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [
            self._tensor_attr_cls(group_name=g, attr_name=a, index=None)
            for (g, a) in self._feat.keys()
        ]