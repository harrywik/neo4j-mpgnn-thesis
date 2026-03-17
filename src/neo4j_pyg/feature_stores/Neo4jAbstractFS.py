from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType
from neo4j import Driver
from benchmarking_tools import Measurer
from benchmarking_tools.QueryProfileAccumulator import QueryProfileAccumulator
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np
import time
import atexit
from neo4j import GraphDatabase
from abc import abstractmethod, ABC

class Neo4jAbstractFS(FeatureStore, ABC):
    def __init__(self, driver: Driver | None = None, uri: str = None, user: str = None, pwd: str = None, measurer: Measurer = None, database_name: str = None, dataset_name: str = "neo4j", feature_property: str = "features", target_property: str = "category", split_property_name: str = "split", split_property_type: str = "int", nodeid_property: str = "nodeId", feature_property_type: str = "f64[]", profile: bool = False, profile_accumulator: Optional[QueryProfileAccumulator] = None):
        super().__init__()
        self.driver = driver
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self.measurer = measurer
        self.database_name = database_name if database_name else dataset_name
        self.dataset_name = dataset_name
        self.feature_property = feature_property
        self.target_property = target_property
        self.split_property_name = split_property_name
        self.split_property_type = split_property_type
        self.nodeid_property = nodeid_property
        self.feature_property_type = feature_property_type
        self.profile = profile
        self.profile_accumulator = profile_accumulator
        self._labels: Dict[str, int] = {}

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()
        data_map: dict = {}
        missing_indices = []
        is_label = (attr.attr_name == "y")

        for nid in node_ids:
            cached = self._get_cached_value(nid, attr)
            if cached is not None:
                data_map[nid] = cached
            else:
                missing_indices.append(nid)

        if self.measurer is not None:
            self.measurer.log_event("cache_hit", len(data_map))
            self.measurer.log_event("cache_miss", len(missing_indices))
            self.measurer.log_event("remote_feature_fetch", 1)

        if missing_indices:
            fetched = self._get_value_from_db(missing_indices, attr)
            for nid, val in fetched.items():
                data_map[nid] = val
                self._update_cached_value(nid, val, attr)

        ordered_list = [data_map[i] for i in node_ids]
        if is_label:
            return torch.tensor(ordered_list, dtype=torch.int64)
        return torch.from_numpy(np.stack(ordered_list))

    @abstractmethod
    def _get_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> FeatureTensorType:
        """
        abstract method to get a cached value for a given node ID. Must be implemented by the subclass.
        """
        pass
    
    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:
        """Fetch *nids* from Neo4j and return a ``{nid: processed_value}`` dict.

        Each record is matched to its node via ``record["id"]`` — no positional
        assumptions are made about query result ordering. Measurer micro-timers
        (``feat_x_*`` / ``feat_y_*``) are logged when a measurer is attached.
        When ``self.profile`` is ``True`` the query is prefixed with
        ``PROFILE`` and the plan is forwarded to ``self.profile_accumulator``.

        Subclasses may override this to customise the query or processing logic.
        """
        is_label = attr.attr_name == "y"
        prop = self.target_property if is_label else self.feature_property
        phase = "feat_y_" if is_label else "feat_x_"
        profile_prefix = "PROFILE " if self.profile else ""

        query = (
            f"{profile_prefix}MATCH (n) WHERE n.{self.nodeid_property} IN $node_ids "
            f"RETURN n.{self.nodeid_property} AS id, n.{prop} AS value"
        )

        with self._get_driver().session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, node_ids=nids)

            # Consume all records first — the driver only populates
            # summary.profile after every record has been received.
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        # Client-side wall time: send → all records received (includes network + DB exec).
        total_feat_fetch_ms = (t_all_records - t_send) * 1000

        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_feat_fetch_ms)

        if self.profile_accumulator is not None:
            source = "feat_y" if is_label else "feat_x"
            self.profile_accumulator.add(summary, source, t_send, t_all_records)

        result_map: Dict[int, object] = {}
        t_etl_start = time.monotonic()

        if not records:
            if self.measurer:
                self.measurer.log_event(f"{phase}etl_ms", (time.monotonic() - t_etl_start) * 1000)
            return result_map

        for rec in records:
            nid = rec["id"]
            val = rec["value"]
            if is_label:
                if isinstance(val, str):
                    if val not in self._labels:
                        self._labels[val] = len(self._labels)
                    val = self._labels[val]
                result_map[nid] = int(val)
            else:
                if self.feature_property_type == "byte[]":
                    result_map[nid] = np.frombuffer(bytes(val), dtype=np.float32).copy()
                elif self.feature_property_type == "f64[]":
                    result_map[nid] = np.asarray(val, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported feature_property_type: {self.feature_property_type!r}")

        if self.measurer is not None:
            self.measurer.log_event(f"{phase}etl_ms", (time.monotonic() - t_etl_start) * 1000)
        return result_map
    
    @abstractmethod
    def _update_cached_value(self, nid: int, value: FeatureTensorType, attr: TensorAttr, **kwargs) -> None:
        """
        abstract method to update a cached value for a given node ID. Must be implemented by the subclass.
        """
        pass
    
    @abstractmethod
    def _remove_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> None:
        """
        abstract method to remove a cached value for a given node ID. Must be implemented by the subclass.
        """
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_driver"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._driver = None

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
        self._driver = None

    @staticmethod
    def _key(attr: TensorAttr) -> Tuple[Optional[str], str]:
        return (None, attr.attr_name)
    
    def get_all_tensor_attrs(self) -> List[TensorAttr]:
        # This tells PyG: "I have features (x) and labels (y) for nodes."
        return [
            TensorAttr(group_name=None, attr_name='x'),
            TensorAttr(group_name=None, attr_name='y')
        ]
    
    def _get_tensor_size(self, attr: TensorAttr) -> Tuple[int, ...]:
        out = self._get_tensor(attr)
        if out is None:
            raise KeyError(f"Tensor not found for {attr}")
        return tuple(out.size())

    def _put_tensor(self, tensor: torch.Tensor, attr: TensorAttr) -> bool:
        pass

    def _remove_tensor(self, attr: TensorAttr) -> bool:
        pass
