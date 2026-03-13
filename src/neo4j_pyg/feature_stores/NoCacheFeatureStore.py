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
    def __init__(self, driver: Driver, measurer:Measurer = None, database_name = None, dataset_name:str = "neo4j", feature_property:str = "features", target_property:str = "category", split_property_name:str = "split", split_property_type:str = "int", nodeid_property:str = "nodeId", feature_property_type:str = "f64[]") -> None:
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
        self.database_name = database_name if database_name else dataset_name

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        prop = self.target_property if attr.attr_name == "y" else self.feature_property

        # Sort node IDs to align with the query's ORDER BY ASC, then restore
        # the original order at the end via a single numpy fancy-index.
        # This eliminates the data_map dict, the ordered-list comprehension,
        # and per-record numpy allocations.
        node_ids_arr = np.asarray(attr.index, dtype=np.int64)
        sort_perm = np.argsort(node_ids_arr, kind="stable")
        inv_perm = np.argsort(sort_perm, kind="stable")
        sorted_ids = node_ids_arr[sort_perm].tolist()

        query = f"""
        MATCH (n)
        WHERE n.{self.nodeid_property} IN $node_ids
        RETURN n.{self.nodeid_property} AS id, n.{prop} AS value
        ORDER BY n.{self.nodeid_property} ASC
        """

        if self.measurer:
            self.measurer.log_event("cache_hit", 0)
            self.measurer.log_event("cache_miss", len(sorted_ids))
            self.measurer.log_event("remote_feature_fetch", 1)

        with self.driver.session(database=self.database_name) as session:
            result = session.run(query, node_ids=sorted_ids)
            records = list(result)

        if self.measurer:
            self.measurer.log_event("remote_feature_recieved", 1)

        n = len(records)
        if n == 0:
            if prop == self.feature_property:
                return torch.zeros((0, 0), dtype=torch.float32)
            return torch.zeros(0, dtype=torch.int64)

        if prop == self.feature_property:
            # Pre-allocate the full feature matrix and fill row-by-row.
            # Records arrive in sorted order (ORDER BY matches sorted_ids),
            # so no dict is needed — inv_perm restores the original order.
            if self.feature_property_type == "byte[]":
                feat_dim = len(np.frombuffer(bytes(records[0]["value"]), dtype=np.float32))
                final_array = np.empty((n, feat_dim), dtype=np.float32)
                for i, rec in enumerate(records):
                    final_array[i] = np.frombuffer(bytes(rec["value"]), dtype=np.float32)
            elif self.feature_property_type == "f64[]":
                feat_dim = len(records[0]["value"])
                final_array = np.empty((n, feat_dim), dtype=np.float32)
                for i, rec in enumerate(records):
                    final_array[i] = rec["value"]
            else:
                raise ValueError(f"Unsupported feature_property_type: {self.feature_property_type!r}")

            return torch.from_numpy(final_array[inv_perm])

        # Labels: collect in sorted order, encode strings, restore order.
        label_ints = np.empty(n, dtype=np.int64)
        for i, rec in enumerate(records):
            v = rec["value"]
            if isinstance(v, str):
                if v not in self._labels:
                    self._labels[v] = len(self._labels)
                v = self._labels[v]
            label_ints[i] = v
        return torch.from_numpy(label_ints[inv_perm])

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