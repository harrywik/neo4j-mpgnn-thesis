from neo4j_pyg.feature_caches.Neo4jAbstractCache import Neo4jAbstractCache
from torch_geometric.data.feature_store import FeatureStore
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType
from neo4j import Driver
from benchmarking_tools import Measurer
from benchmarking_tools.QueryProfileAccumulator import QueryProfileAccumulator
from typing import Optional, Dict, List, Tuple
from functools import cached_property
import torch
import numpy as np
import time
import atexit
from neo4j import GraphDatabase
from abc import abstractmethod, ABC

class Neo4jAbstractFS(FeatureStore, ABC):
    def __init__(self, driver: Driver | None = None, uri: str = None, user: str = None, pwd: str = None, cache: Neo4jAbstractCache = None, measurer: Measurer = None, database_name: str = None, dataset_name: str = "neo4j", feature_property: str = "features", target_property: str = "category", split_property_name: str = "split", split_property_type: str = "int", nodeid_property: str = "nodeId", feature_property_type: str = "f64[]", profile: bool = False, profile_accumulator: Optional[QueryProfileAccumulator] = None, node_label: str = None):
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
        self._feature_dim: Optional[int] = None
        self.t_feat_etl_start: Optional[float] = None
        self._cache: Optional[Neo4jAbstractCache] = cache
        if self._cache is not None:
            self._labels = self._cache._labels
            self.cache_size = self._cache.cache_size
            for attr in ("hot_cache", "hot_label_cache", "cache", "label_cache"):
                if hasattr(self._cache, attr):
                    setattr(self, attr, getattr(self._cache, attr))
            self._prefill_hot_cache(
                graph_name="hot_cache_projection",
                k=self._cache.cache_size // 3,
            )

    def _prefill_hot_cache(
        self,
        graph_name: str,
        k: int = 500,
        undirected: bool = True,
        drop_graph: bool = True,
    ) -> None:
        """Fill the static hot cache with the top-K nodes ranked by PageRank."""
        if self._cache is None:
            return
        if undirected is not True:
            raise ValueError("Neo4jTwoLevelCache currently only supports undirected hot-cache prefill")
        if drop_graph is not True:
            raise ValueError("Neo4jTwoLevelCache currently always drops temporary hot-cache projections")
        self._cache.prefill_hot_cache(graph_name=graph_name, k=k)
        hot_cache = getattr(self._cache, "hot_cache", {})
        print(f"Hot cache prefilled with {len(hot_cache)} nodes.")

    @cached_property
    def _query_both(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}UNWIND $node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"RETURN nid AS id, "
            f"n.{self.feature_property} AS feature, "
            f"n.{self.target_property} AS label"
        )

    @cached_property
    def _query_x(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}UNWIND $node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"RETURN nid AS id, n.{self.feature_property} AS value"
        )

    @cached_property
    def _query_y(self) -> str:
        prefix = "PROFILE " if self.profile else ""
        lf = f":{self.node_label}" if self.node_label else ""
        return (
            f"{prefix}UNWIND $node_ids AS nid "
            f"MATCH (n{lf} {{{self.nodeid_property}: nid}}) "
            f"RETURN nid AS id, n.{self.target_property} AS value"
        )

    def _query_both_params(self, nids: List[int], x_attr: TensorAttr) -> Dict[str, object]:
        return {"node_ids": nids}

    def _query_value_params(self, nids: List[int], attr: TensorAttr) -> Dict[str, object]:
        return {"node_ids": nids}

    def _decode_feature_matrix(self, records: List[object], field_name: str) -> np.ndarray:
        fpt = self.feature_property_type
        if fpt == "byte[]":
            return np.stack([
                np.frombuffer(memoryview(rec[field_name]), dtype=np.float32)
                for rec in records
            ])
        if fpt == "f64[]":
            return np.asarray([rec[field_name] for rec in records], dtype=np.float32)
        raise ValueError(f"Unsupported feature_property_type: {fpt!r}")

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
        n = len(node_ids)
        nid_to_pos: Dict[int, int] = {nid: i for i, nid in enumerate(node_ids)}

        # Check per-attr caches (LRU in CachedFS; always-miss in NoCacheFS).
        cached_x: Dict[int, object] = {}
        cached_y: Dict[int, object] = {}
        missing: List[int] = []

        for nid in node_ids:
            xc = self._get_cached_value(nid, x_attr)
            yc = self._get_cached_value(nid, y_attr)
            if xc is not None:
                cached_x[nid] = xc
            if yc is not None:
                cached_y[nid] = yc
            if xc is None or yc is None:
                missing.append(nid)

        if self.measurer is not None:
            self.measurer.log_event("cache_hit", len(cached_x) + len(cached_y))
            self.measurer.log_event("cache_miss", len(missing))

        if missing and self.measurer is not None:
            self.measurer.log_event("remote_feature_fetch", 1)

        feat_dim = self._feature_dim
        if feat_dim is None and cached_x:
            feat_dim = next(iter(cached_x.values())).shape[0]
            self._feature_dim = feat_dim

        if missing:
            fetched_nids, feat_matrix, y_array = self._get_both_from_db(missing, x_attr)

            if feat_dim is None and len(feat_matrix):
                feat_dim = feat_matrix.shape[1]
                self._feature_dim = feat_dim

            feat_out = np.empty((n, feat_dim), dtype=np.float32)
            y_out = np.empty(n, dtype=np.int64)

            for nid, val in cached_x.items():
                feat_out[nid_to_pos[nid]] = val
            for nid, val in cached_y.items():
                y_out[nid_to_pos[nid]] = val

            pos_array = np.fromiter(
                (nid_to_pos[nid] for nid in fetched_nids),
                dtype=np.int64, count=len(fetched_nids),
            )
            feat_out[pos_array] = feat_matrix
            y_out[pos_array] = y_array

            for i, nid in enumerate(fetched_nids):
                self._update_cached_value(nid, feat_matrix[i], x_attr)
                self._update_cached_value(nid, int(y_array[i]), y_attr)
        else:
            # All nodes cached — no DB call, ETL starts immediately.
            self.t_feat_etl_start = time.monotonic()
            if self.measurer is not None:
                self.measurer.log_event("start_etl", 1)
                self.measurer.set_phase("etl")

            feat_out = np.empty((n, feat_dim), dtype=np.float32)
            y_out = np.empty(n, dtype=np.int64)
            for nid, val in cached_x.items():
                feat_out[nid_to_pos[nid]] = val
            for nid, val in cached_y.items():
                y_out[nid_to_pos[nid]] = val

        x_tensor = torch.from_numpy(feat_out)
        y_tensor = torch.from_numpy(y_out)

        result = []
        for attr in attrs:
            if attr.attr_name == "x":
                result.append(x_tensor)
            elif attr.attr_name == "y":
                result.append(y_tensor)
            else:
                raise ValueError(f"Unsupported attribute: {attr.attr_name}")

        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")
            self.measurer.log_event("feat_x_etl_ms", (time.monotonic() - self.t_feat_etl_start) * 1000)

        return result

    def _get_both_from_db(
        self, nids: List[int], x_attr: TensorAttr
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Fetch both feature vector and label for *nids* in one Cypher query.

        Returns ``(fetched_nids, feat_matrix, y_array)`` where *fetched_nids*
        is the list of node IDs in DB-returned order, *feat_matrix* is shape
        ``(len(fetched_nids), feature_dim)`` float32, and *y_array* is shape
        ``(len(fetched_nids),)`` int64.  The caller is responsible for placing
        rows into the correct output positions via ``nid_to_pos``.
        """
        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(self._query_both, **self._query_both_params(nids, x_attr))
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        total_ms = (t_all_records - t_send) * 1000

        # DB call is done — start timing pure Python assembly.
        self.t_feat_etl_start = time.monotonic()
        if self.measurer is not None:
            self.measurer.log_event("start_etl", 1)
            self.measurer.set_phase("etl")

        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_ms)
            self._log_extra_feature_fetch_metrics(records, total_ms, is_label=False, paired=True)

        if self.profile_accumulator is not None:
            self.profile_accumulator.add(summary, "feat_x", t_send, t_all_records)

        t_etl = time.monotonic()

        if not records:
            if self.measurer is not None:
                self.measurer.log_event("feat_x_etl_parse_ms", 0.0)
            empty_feats = np.empty((0, self._feature_dim or 0), dtype=np.float32)
            return [], empty_feats, np.empty(0, dtype=np.int64)

        feat_matrix = self._decode_feature_matrix(records, "feature")

        fetched_nids: List[int] = []
        y_array = np.empty(len(records), dtype=np.int64)

        first_label = records[0]["label"]
        if isinstance(first_label, str):
            for i, rec in enumerate(records):
                fetched_nids.append(rec["id"])
                lbl = rec["label"]
                if lbl not in self._labels:
                    self._labels[lbl] = len(self._labels)
                y_array[i] = self._labels[lbl]
        else:
            for i, rec in enumerate(records):
                fetched_nids.append(rec["id"])
                y_array[i] = int(rec["label"])

        if self.measurer is not None:
            self.measurer.log_event("feat_x_etl_parse_ms", (time.monotonic() - t_etl) * 1000)

        return fetched_nids, feat_matrix, y_array

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        self.t_feat_etl_start = time.monotonic()
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

        if missing_indices and self.measurer is not None:
            self.measurer.log_event("remote_feature_fetch", 1)

        if missing_indices:
            fetched = self._get_value_from_db(missing_indices, attr)
            for nid, val in fetched.items():
                data_map[nid] = val
                self._update_cached_value(nid, val, attr)

        ordered_list = [data_map[i] for i in node_ids]
        if is_label:
            result = torch.tensor(ordered_list, dtype=torch.int64)
        else:
            result = torch.from_numpy(np.stack(ordered_list))

        if self.measurer is not None:
            metric = "feat_y_etl_ms" if is_label else "feat_x_etl_ms"
            self.measurer.log_event(metric, (time.monotonic() - self.t_feat_etl_start) * 1000)

        return result

    def _get_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> Optional[object]:
        """Return the cached value for *nid*, or ``None`` if not cached."""
        if self._cache is None:
            return None
        return self._cache.get((attr.attr_name, nid))

    def _get_value_from_db(self, nids: list, attr: TensorAttr, **kwargs) -> Dict[int, object]:

        """Fetch *nids* from Neo4j for a single attr and return a ``{nid: value}`` dict.

        This is the fallback path used by :meth:`_get_tensor` when only one of
        ``x`` or ``y`` is requested.  The normal training path goes through
        :meth:`_get_both_from_db` instead.
        """
        is_label = attr.attr_name == "y"
        phase = "feat_y_" if is_label else "feat_x_"
        query = self._query_y if is_label else self._query_x

        with self._get_driver().session(database=self.database_name, fetch_size=1000) as session:
            t_send = time.monotonic()
            result = session.run(query, **self._query_value_params(nids, attr))
            records = list(result)
            t_all_records = time.monotonic()
            summary = result.consume()

        total_feat_fetch_ms = (t_all_records - t_send) * 1000

        if self.measurer is not None:
            self.measurer.log_event("remote_feature_recieved", 1)
            self.measurer.log_event("feat_fetch_ms", total_feat_fetch_ms)
            self._log_extra_feature_fetch_metrics(records, total_feat_fetch_ms, is_label=is_label, paired=False)

        if self.profile_accumulator is not None:
            source = "feat_y" if is_label else "feat_x"
            self.profile_accumulator.add(summary, source, t_send, t_all_records)

        result_map: Dict[int, object] = {}
        t_etl_start = time.monotonic()

        if not records:
            if self.measurer is not None:
                self.measurer.log_event(f"{phase}etl_parse_ms", (time.monotonic() - t_etl_start) * 1000)
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
            feat_matrix = self._decode_feature_matrix(records, "value")

            for i, rec in enumerate(records):
                result_map[rec["id"]] = feat_matrix[i]

        if self.measurer is not None:
            self.measurer.log_event(f"{phase}etl_parse_ms", (time.monotonic() - t_etl_start) * 1000)
        return result_map

    def _update_cached_value(self, nid: int, value: object, attr: TensorAttr, **kwargs) -> None:
        """Insert *value* for *nid* into the cache, evicting the oldest if full."""
        if self._cache is None:
            return
        self._cache[(attr.attr_name, nid)] = value

    def _remove_cached_value(self, nid: int, attr: TensorAttr, **kwargs) -> None:
        """Remove *nid* from the cache (no-op if absent or no cache configured)."""
        if self._cache is None:
            return
        self._cache.delete((attr.attr_name, nid))

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
