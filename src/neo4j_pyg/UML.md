# neo4j_pyg — ASCII UML Diagram

---

## 1. Inheritance Overview

```
[torch_geometric]                   [PyTorch]                 [Python stdlib]
FeatureStore  GraphStore  BaseSampler  DataLoader  nn.Module     ABC
     │              │           │           │           │          │
     │              │           │           │           │          │
     ▼              ▼           │           │           │          ▼
Neo4jAbstractFS  Neo4jAbstractGS│           │           │   Neo4jAbstractCache
  (abstract)       (abstract)   │           │           │      (abstract)
     │                  │       │           │           │          │
     ├─ Neo4jNoCacheFS  │       │           │           │          └─ Neo4jTwoLevelCache
     ├─ Neo4jCachedFS   │       │           │           │
     └─ Neo4jUDPFS      ├─ Neo4jMultiGS     │           ├─ GCN
                        └─ Neo4jSingleGS    │           ├─ GCNPostAggregation
                                            │           └─ MLPPostAggregation
                                     ┌──────┴──────────────────────────────────┐
                                     │              BaseSampler                │
                                     ├─ Neo4jSampler                           │
                                     ├─ Neo4jNeighborSampler                   │
                                     ├─ Neo4jEdgeModeSampler                   │
                                     ├─ Neo4jJavaNeighborSampler               │
                                     ├─ Neo4jAggregationSampler                │
                                     └─ OldNeighborSampler                     │
                                                                               ░│
                                    DataLoader                                 ░│
                                     └─ Neo4jGraphSAINTSampler (abstract)       │
                                           └─ Neo4jGraphSAINTRandomWalkSampler  │
```

---

## 2. Package: `feature_caches`

```
┌──────────────────────────────────────────────────────────────────────────┐
│  <<abstract>>  Neo4jAbstractCache                                        │
│  (inherits: ABC)                                                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  driver          : Optional[Driver]                                      │
│  uri             : Optional[str]                                         │
│  user            : Optional[str]                                         │
│  pwd             : Optional[str]                                         │
│  database_name   : Optional[str]                                         │
│  nodeid_property : str              = "nodeId"                           │
│  feature_property: str              = "features"                         │
│  target_property : str              = "category"                         │
│  feature_property_type : str        = "f64[]"                            │
│  label_map       : Optional[Dict[str,int]]                               │
│  cache_size      : int              (computed from cache_size_GB)        │
│  _labels         : Dict                                                  │
│  _driver         : Driver                                                │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + compute_cache_size(cache_size_GB: float) -> int                       │
│  + _estimate_cached_entry_size_bytes() -> int                            │
│  + _normalize_feature_value(v) -> np.ndarray                             │
│  + _normalize_label_value(v) -> int                                      │
│  + _get_driver() -> Driver                                               │
│  + close()                                                               │
│  + __getitem__(key)                                                      │
│  + __setitem__(key, value)                                               │
│  + __getstate__() / __setstate__()   [pickle-safe]                       │
│  + prefill_hot_cache(graph_name, k)  <<abstract>>                        │
│  + get(key)                          <<abstract>>                        │
│  + set(key, value)                   <<abstract>>                        │
│  + clear()                           <<abstract>>                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │ inherits
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jTwoLevelCache                                                      │
│  (inherits: Neo4jAbstractCache)                                          │
├──────────────────────────────────────────────────────────────────────────┤
│  EXTRA ATTRIBUTES                                                        │
│  ─────────────────────────────────────────────────────────────────────  │
│  cache_size_GB   : float            = 0.000001                           │
│  prefill         : bool             = True                               │
│  hot_cache       : Dict[int, ndarray]   (static PageRank top-k)          │
│  hot_label_cache : Dict[int, int]                                        │
│  cache           : OrderedDict[int, obj]  (LRU eviction)                 │
│  label_cache     : OrderedDict[int, int]                                 │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + prefill_hot_cache(graph_name, k)  [GDS PageRank → hot_cache]          │
│  + get(key: Tuple[str,int]) -> value | None                              │
│  + set(key: Tuple[str,int], value)   [LRU insert, evict oldest if full]  │
│  + clear()                                                               │
│  - _split_key(key) -> (attr_name, nid)                                   │
│  - _pick_stores(attr_name) -> (hot_dict, lru_dict)                       │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Package: `feature_stores`

```
┌──────────────────────────────────────────────────────────────────────────┐
│  <<abstract>>  Neo4jAbstractFS                                           │
│  (inherits: torch_geometric.FeatureStore, ABC)                           │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  driver               : Optional[Driver]                                 │
│  uri / user / pwd     : Optional[str]                                    │
│  measurer             : Optional[Measurer]                               │
│  database_name        : Optional[str]                                    │
│  dataset_name         : str              = "neo4j"                       │
│  feature_property     : str              = "features"                    │
│  target_property      : str              = "category"                    │
│  split_property_name  : str              = "split"                       │
│  split_property_type  : str              = "int"                         │
│  nodeid_property      : str              = "nodeId"                      │
│  feature_property_type: str              = "f64[]"                       │
│  profile              : bool             = False                         │
│  profile_accumulator  : Optional[QueryProfileAccumulator]                │
│  node_label           : Optional[str]                                    │
│  _query_both          : str  (@cached_property)                          │
│  _query_x             : str  (@cached_property)                          │
│  _query_y             : str  (@cached_property)                          │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + _multi_get_tensor(attrs) -> Tensor           [batched x+y one round]  │
│  + _get_tensor(attr) -> Tensor                  [single-attr fetch]      │
│  + _get_both_from_db(nids, x_attr)              [fetches x and y]        │
│     -> (fetched_nids, feat_matrix, y_array)                              │
│  + _get_value_from_db(nids, attr) -> Dict[int,obj]                       │
│  + _decode_feature_matrix(records, field) -> ndarray                     │
│  + _get_driver() -> Driver                                               │
│  + close()                                                               │
│  + get_all_tensor_attrs() -> [TensorAttr(x), TensorAttr(y)]              │
│  + _get_tensor_size(attr)                                                │
│  + _put_tensor(...)   [stub]                                             │
│  + _remove_tensor(...)  [stub]                                           │
│  + _get_cached_value(nid, attr)      <<abstract>>                        │
│  + _update_cached_value(nid, v, attr) <<abstract>>                       │
│  + _remove_cached_value(nid, attr)   <<abstract>>                        │
└───────────┬───────────────┬──────────────────┬───────────────────────────┘
            │               │                  │                   │
            ▼               ▼                  ▼                   ▼
┌────────────────┐ ┌──────────────────┐ ┌────────────────────┐
│Neo4jNoCacheFS  │ │ Neo4jCachedFS    │ │Neo4jUDPFeatureStore│
│                │ │                  │ │                    │
├────────────────┤ ├──────────────────┤ ├────────────────────┤
│ (no new attrs) │ │ label_map:       │ │ sampler:           │
│                │ │  Dict[str,int]   │ │  (optional)        │
│                │ │ cache_size_GB:   │ │ max_neighbors: int │
│                │ │  float           │ │ edge_type: str     │
│                │ │ _cache:          │ │                    │
│                │ │  TwoLevelCache ◆─┼─┼────────────────────┤
├────────────────┤ ├──────────────────┤ │ _query_both        │
│ _get_cached    │ │ _prefill_hot_    │ │  (@cached_property)│
│  _value → None │ │  cache(g,k,...)  │ │  UDP Cypher call   │
│ _update_cached │ │ _get_cached_     │ │ _query_x           │
│  _value → pass │ │  value → cache   │ │  (@cached_property)│
│ _remove_cached │ │ _update_cached   │ │ _decode_feature_   │
│  _value → pass │ │  _value → cache  │ │  matrix → np.array │
│                │ │ _remove_cached   │ │ cache methods→noop │
│                │ │  _value → delete │ └────────────────────┘
└────────────────┘ └──────────────────┘
```

### Composition in feature_stores

```
  Neo4jCachedFS
  ┌──────────────────────────────────────┐
  │  _cache ──────────────────────────► Neo4jTwoLevelCache
  └──────────────────────────────────────┘

  Neo4jUDPFeatureStore
  ┌──────────────────────────────────────┐
  │  sampler ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─► Neo4jAggregationSampler (optional)
  └──────────────────────────────────────┘
```

---

## 4. Package: `graph_stores`

```
┌──────────────────────────────────────────────────────────────────────────┐
│  <<abstract>>  Neo4jAbstractGS                                           │
│  (inherits: torch_geometric.GraphStore, ABC)                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  uri / user / pwd     : Optional[str]                                    │
│  measurer             : Optional[Measurer]                               │
│  database_name        : Optional[str]                                    │
│  dataset_name         : str              = "neo4j"                       │
│  split_property_name  : str              = "split"                       │
│  split_property_type  : str              = "int"                         │
│  nodeid_property      : str              = "nodeId"                      │
│  profile_accumulator  : Optional[QueryProfileAccumulator]                │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + _get_driver() -> Driver                    <<abstract>>               │
│  + get_split(limit, offset, split, shuffle)                              │
│     -> torch.Tensor                           [Cypher split lookup]      │
│  + sample_from_nodes(kwargs, query)                                      │
│     -> (unique_nodes, edge_index_local)       [edge-per-row pattern]     │
│  + fetch_ordered_subgraph(query, kwargs)                                 │
│     -> dict | None                            [single-row topology]      │
│  + fetch_aggregated_features(query, kwargs)                              │
│     -> Dict[int, ndarray]                     [UDP aggregation]          │
│  + fetch_aggregated_neighborhood(query, kwargs)  [alias of above]        │
│  + build_topo_etl(record, fallback_seeds)                                │
│     -> (node_tensor, row, col)                [raw record → PyG tensors] │
│  + _put_edge_index()   [stub]                                            │
│  + _remove_edge_index() [stub]                                           │
│  + get_all_edge_attrs() [stub]                                           │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────────┐   ┌───────────────────────────┐
│  Neo4jMultiGS            │   │  Neo4jSingleGS             │
│                          │   │                            │
├─────────────────────────┤   ├───────────────────────────┤
│  _driver: None           │   │  driver: Driver            │
│  (lazy-created from      │   │  feature_property: str     │
│   uri/user/pwd)          │   │  target_property: str      │
├─────────────────────────┤   ├───────────────────────────┤
│  + _get_driver()         │   │  + _get_driver()           │
│    [lazy init + atexit]  │   │    [returns self.driver]   │
│  + close()               │   │                            │
│  + __getstate__()        │   │                            │
│  + __setstate__()        │   │                            │
│    [pickle-safe]         │   │                            │
└─────────────────────────┘   └───────────────────────────┘
```

---

## 5. Package: `models`

```
┌──────────────────────────────────────────────────────────────────────────┐
│  GCN                                                                     │
│  (inherits: torch.nn.Module)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  GCN1       : GCNConv(in_dim   → hidden_dim1)                            │
│  GCN2       : GCNConv(hidden_dim1 → hidden_dim2)                         │
│  classifier : nn.Linear(hidden_dim2 → nbr_classes)                      │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + forward(X: Tensor, edge_index: Tensor) -> Tensor                      │
│    [GCN1 → ReLU → GCN2 → ReLU → classifier]                             │
│  + reset_parameters()   [Xavier init, fixed seed 0]                     │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  GCNPostAggregation                                                      │
│  (inherits: torch.nn.Module)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  lin1        : nn.Linear(in_dim     → hidden_dim1)                       │
│  gcn2        : GCNConv (hidden_dim1 → hidden_dim2)                       │
│  classifier  : nn.Linear(hidden_dim2 → nbr_classes)                     │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + forward(X, edge_index) -> Tensor                                      │
│    [lin1 (pre-aggregated) → ReLU → gcn2 → ReLU → classifier]            │
│  + reset_parameters()   [Xavier init, fixed seed 0]                     │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  MLPPostAggregation                                                      │
│  (inherits: torch.nn.Module)                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  lin1        : nn.Linear(in_dim     → hidden_dim1)                       │
│  lin2        : nn.Linear(hidden_dim1 → hidden_dim2)                      │
│  classifier  : nn.Linear(hidden_dim2 → nbr_classes)                     │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + forward(X, edge_index=None) -> Tensor   [ignores edge_index; pure MLP]│
│  + reset_parameters()   [Xavier + zeros, fixed seed 0]                  │
└──────────────────────────────────────────────────────────────────────────┘

```

---

## 6. Package: `samplers`

### 6a. Module-level utilities (`_utils.py`)

```
  remap_with_seeds(unique_nodes, edge_index_local, seeds)
    -> (all_nodes: Tensor, edge_index_local: Tensor)
    Purpose: inserts isolated seed nodes into sorted node tensor,
             remaps edge indices accordingly.

  build_sampler_output(graph_store, query, seeds, seed_time)
    -> SamplerOutput
    Purpose: runs graph_store.sample_from_nodes, calls remap_with_seeds,
             wraps result in SamplerOutput.
    Used by: OldNeighborSampler
```

### 6b. Class diagrams

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jSampler                                                            │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  graph_store      : Neo4jAbstractGS                (composition)         │
│  num_neighbors    : List[int]                                            │
│  expand_revisited : bool             = False                             │
│  rel_type         : Optional[str]                                        │
│  node_label       : Optional[str]                                        │
│  profile          : bool             = False                             │
│  query            : str              (Cypher, built in __init__)         │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  - _build_fanout_query() -> str                                          │
│    [multi-hop incoming, unrolled CALL blocks, apoc.coll.toSet dedup]     │
│  + sample_from_nodes(ns_input: NodeSamplerInput) -> SamplerOutput        │
│    [fetch_ordered_subgraph → build_topo_etl → SamplerOutput]            │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jNeighborSampler                                                    │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES  (same as Neo4jSampler)                                      │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  - _build_fanout_query() -> str                                          │
│    [ORDER BY i for seed ordering; order-preserving list-reduce dedup     │
│     instead of apoc.coll.toSet — equivalent to pyg-lib semantics]        │
│  + sample_from_nodes(ns_input) -> SamplerOutput                          │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jEdgeModeSampler                                                    │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  graph_store      : Neo4jAbstractGS                (composition)         │
│  num_neighbors    : List[int]                                            │
│  edge_mode        : EdgeMode         = "incoming"                        │
│    EdgeMode = Literal["incoming","outgoing","undirected","induced"]       │
│  expand_revisited : bool             = False                             │
│  rel_type         : Optional[str]                                        │
│  node_label       : Optional[str]                                        │
│  profile          : bool             = False                             │
│  query            : str              (Cypher, built in __init__)         │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  - _expansion_mode_for_fanout() -> InducedExpansion                      │
│  - _edge_fragments(expansion)                                            │
│     -> (edge_pat, nbr_expr, edge_src, edge_dst)                          │
│  - _build_fanout_query() -> str                                          │
│    [supports all 4 EdgeMode values; "induced" replaces walk edges        │
│     with full induced subgraph via separate MATCH]                       │
│  + sample_from_nodes(ns_input) -> SamplerOutput                          │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jJavaNeighborSampler                                                │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  graph_store  : Neo4jAbstractGS                    (composition)         │
│  num_neighbors: List[int]                                                │
│  rel_type     : Optional[str]                                            │
│  node_label   : Optional[str]                                            │
│  node_id_key  : str              = "id"                                  │
│  random_seed  : int              = 42                                    │
│  profile      : bool             = False                                 │
│  query        : str              (calls gnnProcedures.sampling            │
│                                   .neighbor.sample UDP)                  │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + sample_from_nodes(ns_input) -> SamplerOutput                          │
│    [fetch_ordered_subgraph → build_topo_etl → SamplerOutput]            │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  Neo4jAggregationSampler                                                 │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  graph_store   : Neo4jAbstractGS                   (composition)         │
│  node_id_key   : str             = "id"                                  │
│  feature_key   : str             = "embedding_bytes"                     │
│  feature_type  : str             = "byte[]"                              │
│  node_label    : str             = "Paper"                               │
│  edge_type     : str             = ""                                    │
│  max_neighbors : int             = 10                                    │
│  measurer      : Optional[Measurer]                                      │
│  pending_agg   : Dict[int, ndarray]  (populated per sample call)         │
│  _cypher       : str             (calls gnnProcedures.aggregation         │
│                                   .neighbor.mean UDP)                    │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  + sample_from_nodes(index: NodeSamplerInput) -> SamplerOutput           │
│    [returns empty topology; stores pre-agg features in pending_agg]      │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────────────────┐
│  OldNeighborSampler                                                      │
│  (inherits: torch_geometric.sampler.BaseSampler)                         │
├──────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                              │
│  ─────────────────────────────────────────────────────────────────────  │
│  graph_store           : Neo4jAbstractGS            (composition)        │
│  num_neighbors         : List[int]                                       │
│  sample_with_replacement: bool          = False                          │
│  expand_revisited      : bool           = False                          │
│  direction             : str            = "incoming"                     │
│  rel_type              : Optional[str]                                   │
│  node_label            : Optional[str]                                   │
│  profile               : bool           = False                          │
│  query                 : str            (Cypher, built in __init__)      │
├──────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                 │
│  ─────────────────────────────────────────────────────────────────────  │
│  - _build_fanout_query() -> str                                          │
│    [UNWIND edges AS e RETURN e.src_id, e.dst_id — edge-per-row format;  │
│     supports "incoming"/"outgoing"/"both" directions]                    │
│  + sample_from_nodes(ns_input) -> SamplerOutput                          │
│    [uses build_sampler_output from _utils — different path than others]  │
│  + sample_from_edges(...) -> NotImplementedError                         │
└──────────────────────────────────────────────────────────────────────────┘
```

### 6c. GraphSAINT hierarchy

```
┌────────────────────────────────────────────────────────────────────────────┐
│  <<abstract>>  Neo4jGraphSAINTSampler                                      │
│  (inherits: torch.utils.data.DataLoader)                                   │
├────────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTES                                                                │
│  ───────────────────────────────────────────────────────────────────────  │
│  graph_store    : Neo4jAbstractGS         (composition)                    │
│  feature_store  : Neo4jAbstractFS         (composition)                    │
│  batch_size     : int                                                      │
│  num_steps      : int             = 1                                      │
│  sample_coverage: int             = 50                                     │
│  save_dir       : Optional[str]                                            │
│  log            : bool            = True                                   │
│  rel_type       : Optional[str]                                            │
│  node_label     : Optional[str]                                            │
│  measurer       : Optional[Measurer]                                       │
│  profile        : bool            = False                                  │
├────────────────────────────────────────────────────────────────────────────┤
│  METHODS                                                                   │
│  ───────────────────────────────────────────────────────────────────────  │
│  + __len__() -> num_steps                                                  │
│  + __getitem__(idx) -> calls _sample_nodes() then fetch_ordered_subgraph   │
│  + _collate(data_list) -> torch_geometric.data.Data                        │
│    [builds Data with x, y, edge_index, train_mask, node_norm]             │
│  - _build_subgraph_query() -> str  [induced subgraph Cypher]               │
│  - _get_total_nodes() -> int                                               │
│  - _compute_norm() -> (node_norm_dict, default_norm_float)                 │
│  - _run_query_raw(query, **kwargs)  [bypasses measurer; used pre-sampling] │
│  - _setup()   [no-op hook, overridden by subclasses]                       │
│  + _sample_nodes(logged: bool) -> List[int]   <<abstract>>                 │
│  + _filename (@property) -> str  [norm cache filename]                     │
└──────────────────────────────────────────────────────────────────────────┬─┘
                                                                           │
                                                                           ▼
                    ┌──────────────────────────────────────────────────────┐
                    │  Neo4jGraphSAINTRandomWalkSampler                    │
                    │  (inherits: Neo4jGraphSAINTSampler)                  │
                    ├──────────────────────────────────────────────────────┤
                    │  EXTRA ATTRIBUTES                                    │
                    │  ──────────────────────────────────────────────────  │
                    │  walk_length     : int     (required)                │
                    │  _walk_query     : str     (built in _setup)         │
                    │  _walk_query_unprofile : str                         │
                    ├──────────────────────────────────────────────────────┤
                    │  METHODS                                             │
                    │  ──────────────────────────────────────────────────  │
                    │  - _setup()   [builds _walk_query via                │
                    │                _build_walk_query()]                  │
                    │  + _sample_nodes(logged) -> List[int]                │
                    │    [runs random-walk Cypher → visited node IDs]      │
                    │  + _filename (@property) -> str                      │
                    │    ["neo4j_graphsaint_rw_{walk_len}_{coverage}.pt"]  │
                    └──────────────────────────────────────────────────────┘
```

---

## 7. Cross-Package Composition & Data Flow

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                     COMPOSITION RELATIONSHIPS                           │
 │                                                                         │
 │  Neo4jCachedFS ────────────────────────◆ Neo4jTwoLevelCache             │
 │                                          (._cache)                      │
 │                                                                         │
 │  Neo4jUDPFeatureStore  ─ ─ ─ ─ ─ ─ ─ ◇ (optional sampler)             │
 │                                                                         │
 │  Neo4jGraphSAINTSampler ────────────── ◆ Neo4jAbstractGS               │
 │                          └──────────── ◆ Neo4jAbstractFS               │
 │                                          (.graph_store / .feature_store)│
 │                                                                         │
 │  Neo4jSampler              ────────────◆ Neo4jAbstractGS               │
 │  Neo4jNeighborSampler      ────────────◆ Neo4jAbstractGS               │
 │  Neo4jEdgeModeSampler      ────────────◆ Neo4jAbstractGS               │
 │  Neo4jJavaNeighborSampler  ────────────◆ Neo4jAbstractGS               │
 │  Neo4jAggregationSampler   ────────────◆ Neo4jAbstractGS               │
 │  OldNeighborSampler        ────────────◆ Neo4jAbstractGS               │
 │                                          (.graph_store)                 │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                     SAMPLER DATA FLOW PATTERNS                          │
 │                                                                         │
 │  Standard neighbor sampling path:                                       │
 │  ─────────────────────────────────────────────────────────────────────  │
 │  NodeSamplerInput                                                       │
 │       │                                                                 │
 │       ▼                                                                 │
 │  Sampler.sample_from_nodes()                                            │
 │       │                                                                 │
 │       ├──► graph_store.fetch_ordered_subgraph(query, kwargs)            │
 │       │         │                                                       │
 │       │         ▼                                                       │
 │       │    raw dict record { ordered_nodes, edge_pairs }                │
 │       │         │                                                       │
 │       ├──► graph_store.build_topo_etl(record, fallback_seeds)           │
 │       │         │                                                       │
 │       │         ▼                                                       │
 │       │    (node_tensor, row, col)                                      │
 │       │         │                                                       │
 │       └──► SamplerOutput(node=node_tensor, row=row, col=col, ...)       │
 │                                                                         │
 │  Pre-aggregation path (UDP):                                            │
 │  ─────────────────────────────────────────────────────────────────────  │
 │  NodeSamplerInput                                                       │
 │       │                                                                 │
 │       ▼                                                                 │
 │  Neo4jAggregationSampler                                                │
 │  .sample_from_nodes()                                                   │
 │       │                                                                 │
 │       ├──► graph_store.fetch_aggregated_features(cypher, kwargs)        │
 │       │         │                                                       │
 │       │         ▼                                                       │
 │       │    Dict[int, ndarray]                                           │
 │       │         │                                                       │
 │       ├── stores in self.pending_agg / self.pending_sign                │
 │       │                                                                 │
 │       └──► SamplerOutput(node=seeds, row=[], col=[], ...)   (no edges)  │
 │                    │                                                    │
 │                    ▼                                                    │
 │            FeatureStore._multi_get_tensor()                             │
 │                reads from sampler.pending_agg / .pending_sign           │
 │                instead of querying the DB                               │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                  FEATURE STORE CACHE LOOKUP ORDER                       │
 │                                                                         │
 │  Neo4jCachedFS._get_tensor(attr)                                        │
 │       │                                                                 │
 │       ├─ 1. check hot_cache (PageRank top-k nodes, static)              │
 │       │       │  HIT → return immediately                               │
 │       │       │                                                         │
 │       ├─ 2. check LRU cache (OrderedDict, bounded by cache_size_GB)     │
 │       │       │  HIT → return, move to end (most-recently-used)         │
 │       │       │                                                         │
 │       └─ 3. MISS → query Neo4j DB                                       │
 │                  → insert into LRU cache (evict LRU entry if full)      │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Legend

```
  ──────▷    inheritance (open arrowhead → parent)
  ────────◆  composition (filled diamond on owner side)
  ─ ─ ─ ─◇  optional composition / association (open diamond)
  ──────►    association / dependency

  <<abstract>>   abstract class (cannot be instantiated directly)
  @property      Python property decorator
  @cached_property  computed once and cached on instance
```
