"""Registry of all GNN components.

Maps class-name strings (as used in implementation configs) to the actual
Python classes.  ``Main.py`` uses this to instantiate components from config
without any ``if/elif`` chains.

Deprecated classes are imported defensively — if a module is missing the entry
is simply omitted from the registry and an attempt to use it raises a clear
KeyError.
"""

import inspect
from typing import Any, Dict, Type

# ---------------------------------------------------------------------------
# Feature stores
# ---------------------------------------------------------------------------
from neo4j_pyg.feature_stores.Neo4jNoCacheFS import Neo4jNoCacheFS
from neo4j_pyg.feature_stores.Neo4jCachedFS import Neo4jCachedFS
from neo4j_pyg.feature_stores.Neo4jPreAggFeatureStore import Neo4jPreAggFeatureStore
from neo4j_pyg.feature_stores.Neo4jSIGNFeatureStore import Neo4jSIGNFeatureStore

FEATURE_STORES: Dict[str, Type] = {
    "Neo4jNoCacheFS": Neo4jNoCacheFS,
    "Neo4jCachedFS": Neo4jCachedFS,
    "Neo4jPreAggFeatureStore": Neo4jPreAggFeatureStore,
    "Neo4jUDPFeatureStore": Neo4jPreAggFeatureStore,  # backward-compat alias
    "Neo4jSIGNFeatureStore": Neo4jSIGNFeatureStore,
}

try:
    from neo4j_pyg.deprecated.PageRankCacheFeatureStore import PageRankCacheFeatureStore
    FEATURE_STORES["PageRankCacheFeatureStore"] = PageRankCacheFeatureStore
except ImportError:
    pass

try:
    from neo4j_pyg.deprecated.CachedPickleSafeFS import CachedPickleSafeFS
    FEATURE_STORES["CachedPickleSafeFS"] = CachedPickleSafeFS
except ImportError:
    pass

try:
    from neo4j_pyg.deprecated.PickleSafeFeatureStore import PickleSafeFeatureStore
    FEATURE_STORES["PickleSafeFeatureStore"] = PickleSafeFeatureStore
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Graph stores
# ---------------------------------------------------------------------------
from neo4j_pyg.graph_stores.Neo4jSingleGS import Neo4jSingleGS
from neo4j_pyg.graph_stores.Neo4jMultiGS import Neo4jMultiGS

GRAPH_STORES: Dict[str, Type] = {
    "Neo4jSingleGS": Neo4jSingleGS,
    "Neo4jMultiGS": Neo4jMultiGS,
}

try:
    from neo4j_pyg.deprecated.BaseLineGS import BaseLineGS
    GRAPH_STORES["BaseLineGS"] = BaseLineGS
except ImportError:
    pass

try:
    from neo4j_pyg.deprecated.PickleSafeGS import PickleSafeGS
    GRAPH_STORES["PickleSafeGS"] = PickleSafeGS
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
from neo4j_pyg.samplers.Neo4jSampler import Neo4jSampler
from neo4j_pyg.samplers.Neo4jJavaNeighborSampler import Neo4jJavaNeighborSampler
from neo4j_pyg.deprecated.Neo4jAggregationSampler import Neo4jAggregationSampler
from neo4j_pyg.samplers.Neo4jSIGNSampler import Neo4jSIGNSampler
from neo4j_pyg.samplers.Neo4jGraphSAINTSampler import (
    Neo4jGraphSAINTSampler,
    Neo4jGraphSAINTRandomWalkSampler,
)

SAMPLERS: Dict[str, Type] = {
    "Neo4jSampler": Neo4jSampler,
    "Neo4jJavaNeighborSampler": Neo4jJavaNeighborSampler,
    "Neo4jAggregationSampler": Neo4jAggregationSampler,
    "Neo4jSIGNSampler": Neo4jSIGNSampler,
    "Neo4jGraphSAINTSampler": Neo4jGraphSAINTSampler,
    "Neo4jGraphSAINTRandomWalkSampler": Neo4jGraphSAINTRandomWalkSampler,
}

try:
    from neo4j_pyg.deprecated.UniformSampler import UniformSampler
    SAMPLERS["UniformSampler"] = UniformSampler
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
from neo4j_pyg.models.GCN import GCN
from neo4j_pyg.models.GCNPostAggregation import GCNPostAggregation, MLPPostAggregation
from neo4j_pyg.models.SIGNPostAggregation import SIGNPostAggregation

MODELS: Dict[str, Type] = {
    "GCN": GCN,
    "GCNPostAggregation": GCNPostAggregation,
    "MLPPostAggregation": MLPPostAggregation,
    "SIGNPostAggregation": SIGNPostAggregation,
}

# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------
from training.Training import Trainer
from training.GraphSAINTTrainer import GraphSAINTTrainer

TRAINERS: Dict[str, Type] = {
    "Trainer": Trainer,
    "GraphSAINTTrainer": GraphSAINTTrainer,
}

try:
    from training.DistributedTraining import DistributedTrainer
    TRAINERS["DistributedTrainer"] = DistributedTrainer
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def filter_kwargs(cls: Type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the kwargs accepted by ``cls.__init__``'s signature.

    Uses ``inspect.signature`` so that adding/removing constructor parameters
    in any class automatically stays in sync without changes here.
    """
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters) - {"self"}
    if "kwargs" in valid or any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    ):
        # Constructor accepts **kwargs — pass everything through.
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid}
