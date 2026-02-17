import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import Driver
from typing import Optional, List, Tuple, Dict
from collections import OrderedDict
import numpy as np

class Neo4jFeatureStore(FeatureStore):
    """Neo4j feature store with caching and deterministic label mapping, but without pickle safety.
    Caches both X and y unlike simple cache feature store."""
    def __init__(
        self,
        driver: Driver,
        cache_size: int = 3000,
        label_cache_size: Optional[int] = None,
        batch_cache_size: int = 64,
        db_batch_size: int = 1000,
    ) -> None:
        super().__init__()
        self.driver = driver
        self._labels = {}
        # Simple LRU using an OrderedDict
        self.cache = OrderedDict()
        self.label_cache = OrderedDict()
        self.batch_cache = OrderedDict()
        self.batch_label_cache = OrderedDict()
        self.cache_size = cache_size
        self.label_cache_size = label_cache_size or cache_size
        self.batch_cache_size = batch_cache_size
        self.db_batch_size = db_batch_size

    @staticmethod
    def _chunked(values: List[int], size: int) -> List[List[int]]:
        if size <= 0:
            return [values]
        return [values[i : i + size] for i in range(0, len(values), size)]

    @staticmethod
    def _batch_key(node_ids: List[int]) -> Tuple[int, ...]:
        return tuple(node_ids)

    def _get_cached_batch(self, cache: OrderedDict, key: Tuple[int, ...]):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def _set_cached_batch(self, cache: OrderedDict, key: Tuple[int, ...], value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.batch_cache_size:
            cache.popitem(last=False)

    def prewarm(self, node_ids: List[int], include_embeddings: bool = True, include_labels: bool = True) -> None:
        ids = list(node_ids)
        if include_embeddings:
            _ = self._get_tensor(TensorAttr(group_name=None, attr_name="x", index=torch.tensor(ids)))
        if include_labels:
            _ = self._get_tensor(TensorAttr(group_name=None, attr_name="y", index=torch.tensor(ids)))

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()
        batch_key = self._batch_key(node_ids)

        # If looking for a categorical target
        if attr.attr_name == "y":
            cached_batch = self._get_cached_batch(self.batch_label_cache, batch_key)
            if cached_batch is not None:
                return cached_batch
            # `subject` is hardcoded as the categorical target for now
            query = f"""
            MATCH (n)
            WHERE n.id IN $node_ids
            RETURN n.id AS id, n.subject AS value
            ORDER BY n.id ASC
            """
            data_map = {}
            missing_indices = []

            for nid in node_ids:
                if nid in self.label_cache:
                    self.label_cache.move_to_end(nid)
                    data_map[nid] = self.label_cache[nid]
                else:
                    missing_indices.append(nid)

            if missing_indices:
                missing_indices.sort()
                with self.driver.session() as session:
                    for chunk in self._chunked(missing_indices, self.db_batch_size):
                        result = session.run(query, node_ids=chunk)
                        for record in result:
                            label_str = record["value"]
                            if label_str not in self._labels:
                                self._labels[label_str] = len(self._labels)
                            num_label = self._labels[label_str]
                            data_map[record["id"]] = num_label
                            self.label_cache[record["id"]] = num_label
                            if len(self.label_cache) > self.label_cache_size:
                                self.label_cache.popitem(last=False)
                
            ordered_list = [data_map[i] for i in node_ids]
            out = torch.tensor(ordered_list, dtype=torch.int64)
            self._set_cached_batch(self.batch_label_cache, batch_key, out)
            return out

        # else: look for embedding
        cached_batch = self._get_cached_batch(self.batch_cache, batch_key)
        if cached_batch is not None:
            return cached_batch

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

            missing_indices.sort()
            with self.driver.session() as session:
                for chunk in self._chunked(missing_indices, self.db_batch_size):
                    result = session.run(query, node_ids=chunk)
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
        final_array = np.stack(ordered_list)
        out = torch.from_numpy(final_array)
        self._set_cached_batch(self.batch_cache, batch_key, out)
        return out

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