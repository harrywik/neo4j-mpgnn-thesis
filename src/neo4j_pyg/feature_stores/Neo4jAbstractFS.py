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
    def __init__(self, driver: Driver | None = None, uri: str = None, user: str = None, pwd: str = None, measurer: Measurer = None, database_name: str = None, dataset_name: str = "neo4j", feature_property: str = "features", target_property: str = "category", split_property_name: str = "split", split_property_type: str = "int", nodeid_property: str = "nodeId", feature_property_type: str = "f64[]", profile: bool = False, profile_accumulator: Optional[QueryProfileAccumulator] = None, node_label: str = None):
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
        self.node_label = node_label
        self._labels: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Multi-get: combined x+y fetch in one DB round-trip
    # ------------------------------------------------------------------

    def _multi_get_tensor(self, attrs: List[TensorAttr]) -> List[Optional[FeatureTensorType]]:
        """Override the base-class default to fetch features and labels together.

        When the attr list contains both ``x`` and ``y`` for the same node set
        (the normal training case), a single Cypher query retrieves both columns
        in one round-trip.  Any other combination falls back to the sequential
        :meth:`_get_tensor` loop.
        """
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        if x_attr is None or y_attr is None:
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()

        # Check per-attr caches (LRU in CachedFS; always-miss in NoCacheFS).
        x_map: Dict[int, object] = {}
        y_map: Dict[int, object] = {}
        missing: List[int] = []

        for nid in node_ids:
            xc = self._get_cached_value(nid, x_attr)
            yc = self._get_cached_value(nid, y_attr)
            if xc is not None:
                x_map[nid] = xc
            if yc is not None:
                y_map[nid] = yc
            if xc is None or yc is None:
                missing.append(nid)

        if self.measurer is not None:
            self.measurer.log_event("cache_hit", len(x_map) + len(y_map))
            self.measurer.log_event("cache_miss", len(missing))
            self.measurer.log_event("remote_feature_fetch", 1)

        if missing:
            fetched_x, fetched_y = self._get_both_from_db(missing, x_attr)
            for nid, val in fetched_x.items():
                x_map[nid] = val
                self._update_cached_value(nid, val, x_attr)
            for nid, val in fetched_y.items():
                y_map[nid] = val
                self._update_cached_value(nid, val, y_attr)

        x_tensor = torch.from_numpy(np.stack([x_map[nid] for nid in node_ids]))
        y_tensor = torch.tensor([y_map[nid] for nid in node_ids], dtype=torch.int64)

        result = []
        for attr in attrs:
            if attr.attr_name == "x":
                result.append(x_tensor)
            elif attr.attr_name == "y":
                result.append(y_tensor)
            else:
                raise ValueError(f"Unsupported attribute: {attr.attr_name}")
        return result

    def _get_both_from_db(
        self, nids: List[int], x_attr: TensorAttr
    ) -> Tuple[Dict[int, object], Dict[int, object]]:
        """Fetch both feature vector and label for *nids* in one Cypher query.

        Returns a pair ``(x_map, y_map)`` where keys are node IDs and values
        are the processed feature array and integer label respectively.
        """
        profile_prefix = "PROFILE " if self.profile else ""
        label_filter = f":{self.node_label}" if self.node_label else ""
        query = (
            f"{profile_prefix}MATCH (n{label_filter}) WHERE n.{self.nodeid_property} IN $node_ids "
            f"RETURN n.{self.nodeid_property} AS id, "
            f"n.{self.feature_property} AS feature, "
            f"n.{self.target_property} AS label"
        )

        with self._get_driver().session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, node_ids=nids)
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        total_ms = (t_all_records - t_send) * 1000

        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_ms)

        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "feat_x", t_send, t_all_records)

        x_map: Dict[int, object] = {}
        y_map: Dict[int, object] = {}
        t_etl = time.monotonic()

        if records:
            fpt = self.feature_property_type
            if fpt == "byte[]":
                feat_matrix = np.stack([
                    np.frombuffer(bytes(rec["feature"]), dtype=np.float32)
                    for rec in records
                ])
            elif fpt == "f64[]":
                feat_matrix = np.array(
                    [rec["feature"] for rec in records], dtype=np.float32
                )
            else:
                raise ValueError(f"Unsupported feature_property_type: {fpt!r}")

            for i, rec in enumerate(records):
                nid = rec["id"]
                x_map[nid] = feat_matrix[i]

                raw_label = rec["label"]
                if isinstance(raw_label, str):
                    if raw_label not in self._labels:
                        self._labels[raw_label] = len(self._labels)
                    y_map[nid] = self._labels[raw_label]
                else:
                    y_map[nid] = int(raw_label)

        if self.measurer is not None:
            self.measurer.log_event("feat_x_etl_ms", (time.monotonic() - t_etl) * 1000)

        return x_map, y_map

    # ------------------------------------------------------------------
    # Single-attr get (fallback / non x+y cases)
    # ------------------------------------------------------------------

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
        Implement this method to get a cached value for a given node ID.
        """
        return None

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:

        """Fetch *nids* from Neo4j for a single attr and return a ``{nid: value}`` dict.

        This is the fallback path used by :meth:`_get_tensor` when only one of
        ``x`` or ``y`` is requested.  The normal training path goes through
        :meth:`_get_both_from_db` instead.
        """
        is_label = attr.attr_name == "y"
        phase = "feat_y_" if is_label else "feat_x_"
        profile_prefix = "PROFILE " if self.profile else ""
        prop = self.target_property if is_label else self.feature_property
        label_filter = f":{self.node_label}" if self.node_label else ""

        query = (
            f"{profile_prefix}MATCH (n{label_filter}) WHERE n.{self.nodeid_property} IN $node_ids "
            f"RETURN n.{self.nodeid_property} AS id, n.{prop} AS value"
        )

        with self._get_driver().session(database=self.database_name) as session:
            t_send = time.monotonic()
            result = session.run(query, node_ids=nids)
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

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
            if self.measurer is not None:
                self.measurer.log_event(f"{phase}etl_ms", (time.monotonic() - t_etl_start) * 1000)
            return result_map

        if is_label:
            for rec in records:
                nid = rec["id"]
                val = rec["value"]
                if isinstance(val, str):
                    if val not in self._labels:
                        self._labels[val] = len(self._labels)
                    result_map[nid] = self._labels[val]
                else:
                    result_map[nid] = int(val)
        else:
            fpt = self.feature_property_type
            if fpt == "byte[]":
                feat_matrix = np.stack([
                    np.frombuffer(bytes(rec["value"]), dtype=np.float32)
                    for rec in records
                ])
            elif fpt == "f64[]":
                feat_matrix = np.array(
                    [rec["value"] for rec in records], dtype=np.float32
                )
            else:
                raise ValueError(f"Unsupported feature_property_type: {fpt!r}")

            for i, rec in enumerate(records):
                result_map[rec["id"]] = feat_matrix[i]

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
