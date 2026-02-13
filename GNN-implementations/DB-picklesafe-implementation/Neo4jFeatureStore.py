import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import GraphDatabase
from neo4j import Driver
from typing import Optional, List, Tuple, Dict
from collections import OrderedDict
import numpy as np
import atexit

class Neo4jFeatureStore(FeatureStore):
    """pickle safe implemenation of feature store"""
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        label_map: Optional[Dict[str, int]] = None,
        cache_size: int = 3000
    ) -> None:
        super().__init__()
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self._labels = {}
        # Simple LRU using an OrderedDict
        self.cache = OrderedDict()
        self.cache_size = cache_size
        # IMPORTANT: must be deterministic across workers
        self._label_map: Dict[str, int] = label_map or {}
        

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver

    def close(self) -> None:
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

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        # If looking for a categorical target
        if attr.attr_name == "y":
            # `subject` is hardcoded as the categorical target for now
            query = f"""
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN n.id AS id, n.subject AS value
            ORDER BY n.id ASC
            """
            data_map = {}
            with self._get_driver().session() as session:
                result = session.run(query, node_ids=node_ids)
                for record in result:
                    label_str = record["value"]
                    if label_str not in self._labels:
                        self._labels[label_str] = len(self._labels)
                    num_label = self._labels[label_str]
                    data_map[record["id"]] = num_label
                
            ordered_list = [data_map[i] for i in node_ids]
            return torch.tensor(ordered_list, dtype=torch.int64)

        # else: look for embedding
        # Identify what is already in cache vs. what we need to fetch
        data_map = {}
        missing_indices = []
        
        for nid in node_ids:
            if nid in self.cache:
                # Move to end to maintain LRU order
                self.cache.move_to_end(nid)
                data_map[nid] = self.cache[nid]
            else:
                missing_indices.append(nid)

        if missing_indices:
            query = f"""
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN n.id AS id, n.embedding AS value
            ORDER BY n.id ASC
            """

            with self._get_driver().session() as session:
                result = session.run(query, node_ids=missing_indices)
                for record in result:
                    raw_blob = record["value"]
                    
                    # Asumption embedding is saved as float32 in DB
                    feat = np.frombuffer(raw_blob, dtype=np.float32).copy()
                    data_map[record["id"]] = feat
                    # Store in map and update LRU cache
                    self.cache[record["id"]] = feat
                    
                    # Evict oldest if cache is full
                    if len(self.cache) > self.cache_size:
                        self.cache.popitem(last=False)
                
        # Reconstruct in correct order
        ordered_list = [data_map[i] for i in node_ids]
        # np.stack is faster than torch.stack for numpy inputs
        final_array = np.stack(ordered_list)

        return torch.from_numpy(final_array)

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