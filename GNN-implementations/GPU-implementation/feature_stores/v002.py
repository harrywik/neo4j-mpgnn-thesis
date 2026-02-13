# feature_stores/v001.py
import torch
import numpy as np
from typing import Optional, List, Tuple, Dict
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import GraphDatabase
from neo4j import Driver
import atexit

class Neo4jFeatureStore(FeatureStore):
    """pickle safe implemenation of feature store"""
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        label_map: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None

        # IMPORTANT: must be deterministic across workers
        self._label_map: Dict[str, int] = label_map or {}

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
        return self._driver

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids = attr.index.tolist()
        prop = "subject" if attr.attr_name == "y" else "embedding"

        query = f"""
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.{prop} AS value
        ORDER BY n.id ASC
        """

        with self._get_driver().session() as session:
            result = session.run(query, node_ids=node_ids)
            data_map = {}

            if prop == "embedding":
                for record in result:
                    raw_blob = record["value"]
                    feat = np.frombuffer(raw_blob, dtype=np.float32).copy()
                    data_map[record["id"]] = feat
            else:
                # Use a FIXED mapping. Do NOT create new ids on the fly in workers.
                for record in result:
                    label_str = record["value"]
                    try:
                        data_map[record["id"]] = self._label_map[label_str]
                    except KeyError:
                        raise RuntimeError(
                            f"Label '{label_str}' missing from label_map. "
                            "Build label_map once in the main process and pass it in."
                        )

        ordered = [data_map[i] for i in node_ids]
        if prop == "embedding":
            return torch.from_numpy(np.stack(ordered))
        return torch.tensor(ordered, dtype=torch.int64)
    
    @staticmethod
    def _key(attr: TensorAttr) -> Tuple[Optional[NodeType], str]:
        return (attr.group_name, attr.attr_name)

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        pass

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass
    
    
    def _get_driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver

    def close(self):
        if getattr(self, "_driver", None) is not None:
            try:
                self._driver.close()
            finally:
                self._driver = None
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_driver"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None

    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"Tensor not found for {attr}")
        return tuple(out.size())

    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        return [
            TensorAttr(group_name=None, attr_name="x"),
            TensorAttr(group_name=None, attr_name="y"),
        ]
