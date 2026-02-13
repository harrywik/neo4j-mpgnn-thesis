from typing import Dict, List, Optional, Tuple
from torch import Tensor
from torch_geometric.data import FeatureStore, TensorAttr
from torch_geometric.typing import NodeType

class InMemoryFeatureStore(FeatureStore):
    def __init__(self):
        super().__init__()
        self._feat: Dict[Tuple[Optional[NodeType], str], Tensor] = {}

    @staticmethod
    def _key(attr: TensorAttr) -> Tuple[Optional[NodeType], str]:
        return (attr.group_name, attr.attr_name)

    def _put_tensor(self, tensor: Tensor, attr: TensorAttr) -> bool:
        if attr.index is not None:
            raise ValueError("Only full-tensor writes supported (attr.index must be None).")
        self._feat[self._key(attr)] = tensor
        return True

    def _get_tensor(self, attr: TensorAttr) -> Optional[Tensor]:
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