from aiohttp import ClientError
import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr, NodeType
from neo4j import GraphDatabase
from neo4j import Driver
from typing import Optional, List, Tuple, Dict
from collections import OrderedDict
import numpy as np
import atexit

class CachedPickleSafeFS(FeatureStore):
    """pickle safe implemenation of feature store with simple LRU cache"""
    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        label_map: Optional[Dict[str, int]] = None,
        cache_size: int = 3000,
        hot_cache_size: int = None,
        dataset_name: str = "neo4j",
        feature_property: str = "features",
        target_property: str = "category",
        nodeid_property: str = "nodeId",
        feature_property_type: str = "f64[]",
    ) -> None:
        super().__init__()
        self.uri = uri
        self.user = user
        self.pwd = pwd
        self._driver: Optional[Driver] = None
        self._labels = {}
        
        # IMPORTANT: must be deterministic across workers
        self._label_map: Dict[str, int] = label_map or {}
        self.dataset_name = dataset_name
        self.feature_property = feature_property
        self.target_property = target_property
        self.nodeid_property = nodeid_property
        self.feature_property_type = feature_property_type
        
        # 1. Static "Hot" Cache for PageRank-important nodes
        self.hot_cache = {}
        self.hot_label_cache = {} 
        hot_cache_size = hot_cache_size or int(cache_size // 3)  
        self._prefill_hot_cache(graph_name="hot_cache_projection", k=hot_cache_size)  
        
        # 2. Dynamic LRU Cache for recently accessed nodes
        self.cache = OrderedDict()
        self.label_cache = OrderedDict()
        self.cache_size = cache_size
        

    def _get_driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            atexit.register(self.close)
        return self._driver
    
    def _prefill_hot_cache(
        self,
        graph_name: str,
        k: int = 500,
        undirected: bool = True,
        drop_graph: bool = True,
    ):
        """
        Runs PageRank in GDS and fills the static hot_cache with the top K nodes.
        Call this before starting your training loop.
        """
        self.hot_cache.clear()
        self.hot_label_cache.clear()
        orientation = "UNDIRECTED" if undirected else "NATURAL"
        exists_query = "CALL gds.graph.exists($name) YIELD exists"
        project_query = f"""
        CALL gds.graph.project(
            $name,
            '*',
            {{
              ALL: {{
                type: '*',
                orientation: $orientation
              }}
            }}
        )
        YIELD graphName
        """
        pagerank_query = f"""
        CALL gds.pageRank.stream('{graph_name}')
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS n, score
        ORDER BY score DESC LIMIT $limit
        RETURN n.{self.nodeid_property} AS id, n.{self.feature_property} AS embedding, n.{self.target_property} AS label
        """
        drop_query = "CALL gds.graph.drop($name)"

        projected_here = False
        try:
            with self._get_driver().session(database=self.dataset_name) as session:
                exists = session.run(exists_query, name=graph_name).single()["exists"]
                if not exists:
                    session.run(
                        project_query,
                        name=graph_name,
                        orientation=orientation,
                    )
                    projected_here = True

                result = session.run(pagerank_query, limit=k)
                for record in result:
                    nid = record["id"]
                    # Process Embedding
                    if self.feature_property_type == "byte[]":
                        feat = np.frombuffer(record["embedding"], dtype=np.float32).copy()
                    elif self.feature_property_type == "f64[]":
                        feat = np.asarray(record["embedding"], dtype=np.float32)
                    else:
                        raise ValueError("feat wasn't assigned a value")
                    self.hot_cache[nid] = feat
                    
                    # Process Label
                    label_str = record["label"]
                    if label_str not in self._labels:
                        self._labels[label_str] = len(self._labels)
                    self.hot_label_cache[nid] = self._labels[label_str]

                    if len(self.hot_cache) >= k:
                        break

                if drop_graph and projected_here:
                    session.run(drop_query, name=graph_name)
        except ClientError as exc:
            raise RuntimeError("GDS is unavailable or graph projection failed.") from exc
        
        print(f"Hot cache prefilled with {len(self.hot_cache)} nodes.")


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
        data_map = {}
        missing_indices = []

        is_label = (attr.attr_name == "y")
        
        # Determine which caches to check
        target_hot_cache = self.hot_label_cache if is_label else self.hot_cache
        target_lru_cache = self.label_cache if is_label else self.cache

        for nid in node_ids:
            # CHECK 1: Static Hot Cache
            if nid in target_hot_cache:
                data_map[nid] = target_hot_cache[nid]
            
            # CHECK 2: Dynamic LRU Cache
            elif nid in target_lru_cache:
                target_lru_cache.move_to_end(nid)
                data_map[nid] = target_lru_cache[nid]
            
            # FALLBACK: Mark for DB Fetch
            else:
                missing_indices.append(nid)

        if missing_indices:
            # DB Fetch logic (optimized for labels vs embeddings)
            val_col = self.target_property if is_label else self.feature_property
            query = f"MATCH (n) WHERE n.{self.nodeid_property} IN $node_ids RETURN n.{self.nodeid_property} AS id, n.{val_col} AS value"

            with self._get_driver().session(database=self.dataset_name) as session:
                result = session.run(query, node_ids=missing_indices)
                for record in result:
                    nid, val = record["id"], record["value"]
                    
                    if is_label:
                        if val not in self._labels: self._labels[val] = len(self._labels)
                        processed_val = self._labels[val]
                    else:
                        if self.feature_property_type == "byte[]":
                            processed_val = np.frombuffer(val, dtype=np.float32).copy()
                        elif self.feature_property_type == "f64[]":
                            processed_val = np.asarray(val, dtype=np.float32)
                        else:
                            raise ValueError("feat wasn't assigned a value")

                    data_map[nid] = processed_val
                    
                    # Update LRU cache only
                    target_lru_cache[nid] = processed_val
                    if len(target_lru_cache) > self.cache_size:
                        target_lru_cache.popitem(last=False)

        # Reconstruct result
        ordered_list = [data_map[i] for i in node_ids]
        if is_label:
            return torch.tensor(ordered_list, dtype=torch.int64)
        return torch.from_numpy(np.stack(ordered_list))

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