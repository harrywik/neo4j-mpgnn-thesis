# Backward-compatibility shim — functionality split into adapters.py and hybrid_model.py.

from neo4j_pyg.neo4j_model_interface.adapters import (  # noqa: F401
    PreAggAdapter,
    PreAggGCNConvAdapter,
    _ADAPTER_REGISTRY,
    _replace_first_mp_layer,
)
from neo4j_pyg.neo4j_model_interface.hybrid_model import HybridAggModel  # noqa: F401



