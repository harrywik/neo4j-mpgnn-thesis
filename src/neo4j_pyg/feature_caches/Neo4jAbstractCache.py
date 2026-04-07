from abc import ABC, abstractmethod
import atexit
from collections import OrderedDict
import sys
from typing import Dict, Optional

import numpy as np
from neo4j import Driver, GraphDatabase


class Neo4jAbstractCache(ABC):
    def __init__(
        self,
        driver: Optional[Driver] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        database_name: Optional[str] = None,
        nodeid_property: str = "nodeId",
        feature_property: str = "features",
        target_property: str = "category",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict[str, int]] = None,
        cache_size_GB: float = 0.5,
    ):
        self.driver = driver
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self.database_name = database_name or "neo4j"
        self.nodeid_property = nodeid_property
        self.feature_property = feature_property
        self.target_property = target_property
        self.feature_property_type = feature_property_type
        self._labels: Dict[str, int] = dict(label_map) if label_map else {}
        self._driver: Optional[Driver] = None
        self.cache_size = self.compute_cache_size(cache_size_GB)

    def compute_cache_size(self, cache_size_GB: float) -> int:
        bytes_per_entry = self._estimate_cached_entry_size_bytes()
        if bytes_per_entry <= 0:
            raise ValueError("Estimated cache entry size must be positive")
        entries_per_GB = (1024 ** 3) // bytes_per_entry
        return int(cache_size_GB * entries_per_GB)

    def _estimate_cached_entry_size_bytes(self) -> int:
        query = (
            f"MATCH (n) WHERE n.{self.feature_property} IS NOT NULL "
            f"RETURN n.{self.nodeid_property} AS id, "
            f"n.{self.feature_property} AS feature, n.{self.target_property} AS label "
            "LIMIT 1"
        )

        with self._get_driver().session(database=self.database_name) as session:
            record = session.run(query).single()

        if record is None:
            raise RuntimeError(
                f"Could not estimate cache entry size: no node with property {self.feature_property!r} was found"
            )

        node_id = int(record["id"])
        feature_value = record["feature"]
        label_value = record["label"]
        cached_feature = self._normalize_feature_value(feature_value)
        cached_label = self._normalize_label_value(label_value)
        return self._estimate_ordered_dict_entry_size(node_id, cached_feature) + self._estimate_ordered_dict_entry_size(node_id, cached_label)

    @staticmethod
    def _infer_feature_property_type(feature_value: object) -> str:
        if isinstance(feature_value, (bytes, bytearray, memoryview)):
            return "byte[]"
        if isinstance(feature_value, (list, tuple)):
            return "f64[]"
        raise ValueError(
            f"Unsupported feature property value type {type(feature_value).__name__!r}"
        )

    def _normalize_feature_value(self, feature_value: object) -> np.ndarray:
        feature_type = self._infer_feature_property_type(feature_value)
        self.feature_property_type = feature_type
        if feature_type == "byte[]":
            return np.frombuffer(bytes(feature_value), dtype=np.float32).copy()
        return np.asarray(feature_value, dtype=np.float32)

    def _normalize_label_value(self, label_value: object) -> int:
        if label_value is None:
            return 0
        if isinstance(label_value, str):
            if label_value not in self._labels:
                self._labels[label_value] = len(self._labels)
            return self._labels[label_value]
        return int(label_value)

    def _estimate_ordered_dict_entry_size(self, node_id: int, value: object, sample_size: int = 256) -> int:
        base_key = int(node_id)
        empty = OrderedDict()
        filled = OrderedDict((base_key + i, None) for i in range(sample_size))
        container_bytes = (sys.getsizeof(filled) - sys.getsizeof(empty)) / sample_size
        key_bytes = sys.getsizeof(base_key)
        value_bytes = sys.getsizeof(value)
        return int(round(container_bytes + key_bytes + value_bytes))

    def _get_driver(self) -> Driver:
        if self.driver is not None:
            return self.driver
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

    @abstractmethod
    def prefill_hot_cache(self, graph_name: str, k: int) -> None:
        """Pre-fill the hot cache with the top-K most-connected nodes."""
        pass

    @abstractmethod
    def get(self, key):
        pass

    def get_many(self, keys) -> dict:
        """Return a dict of {key: value} for all cached keys.

        Default implementation calls :meth:`get` in a loop.  Override for
        batch-aware backends (Redis MGET, etc.).
        """
        result = {}
        for k in keys:
            v = self.get(k)
            if v is not None:
                result[k] = v
        return result

    @abstractmethod
    def set(self, key, value):
        pass

    def set_many(self, items: dict) -> None:
        """Write all key→value pairs to the cache.

        Default implementation calls :meth:`set` in a loop.  Override for
        batch-aware backends (Redis pipeline, etc.).
        """
        for k, v in items.items():
            self.set(k, v)

    def __getitem__(self, key):
        value = self.get(key)
        if value is None:
            raise KeyError(key)
        return value

    def __setitem__(self, key, value):
        self.set(key, value)

    @abstractmethod
    def delete(self, key) -> None:
        pass

    @abstractmethod
    def clear(self):
        pass
