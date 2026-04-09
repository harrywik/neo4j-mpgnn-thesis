from neo4j_pyg.neo4j_model_interface.create_inference_spec import (
    create_inference_spec,
    upload_inference_spec,
)
from neo4j_pyg.neo4j_model_interface.preagg_adapters import (
    HybridAggModel,
    PreAggModelWrapper,
    to_preaggregated_first_layer,
    to_hybrid_last_hop_gcn,
)
