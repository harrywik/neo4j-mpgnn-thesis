"""Cora training run using SIGN-style multi-hop aggregation via the Java UDP.

Architecture
------------
* ``custom.gcn.signAggregate`` Java procedure returns per-hop mean aggregations
  for hops 0..k (hop 0 = seed's own features, hop h = mean of h-hop shell).
* ``Neo4jSIGNSampler`` calls the UDP and stores results in ``pending_sign``.
* ``Neo4jSIGNFeatureStore`` concatenates hop aggregations to form the SIGN
  input tensor: ``x = [x_0 || x_1 || ... || x_k]``.
* ``SIGNPostAggregation`` is a 3-layer MLP that maps the concatenated input
  to class logits; full autograd is intact.

References
----------
Rossi et al. (2020) "SIGN: Scalable Inception Graph Neural Networks"
https://arxiv.org/abs/2004.11198
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GNN_IMPL_DIR = Path(__file__).resolve().parent.parent.parent
if str(GNN_IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(GNN_IMPL_DIR))

from neo4j_pyg.models import SIGNPostAggregation
from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.feature_stores import Neo4jSIGNFeatureStore
from neo4j_pyg.graph_stores import Neo4jMultiGS, Neo4SingleGS
from neo4j_pyg.samplers import Neo4jSIGNSampler
from Neo4jConnection import Neo4jConnection
from benchmarking_tools import Measurer, QueryProfileAccumulator


def main(config: dict):
    uri = os.environ["URI"]
    user = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]

    profile = config.get("profile", False)
    profile_accumulator = QueryProfileAccumulator() if profile else None

    measurer = Measurer(config, profile_accumulator=profile_accumulator)

    dataset_name = "cora"
    database_name = "neo4j"

    driver = Neo4jConnection(uri, user, password).get_driver()

    # Number of hops for SIGN aggregation.
    hops = config.get("hops", 2)
    max_neighbors_per_hop = config.get("max_neighbors_per_hop", 10)

    graph_store = Neo4SingleGS(
        driver=driver,
        measurer=measurer,
        database_name=database_name,
        dataset_name=dataset_name,
        feature_property="embedding_bytes",
        nodeid_property="id",
        split_property_name="split",
        split_property_type="str",
        target_property="subject",
        profile_accumulator=profile_accumulator,
    )

    sampler = Neo4jSIGNSampler(
        graph_store=graph_store,
        node_id_key="id",
        feature_key="embedding_bytes",
        feature_type="byte[]",
        node_label="Paper",
        edge_type="CITES",
        hops=hops,
        max_neighbors_per_hop=max_neighbors_per_hop,
        measurer=measurer,
    )

    feature_store = Neo4jSIGNFeatureStore(
        sampler=sampler,
        hops=hops,
        driver=driver,
        measurer=measurer,
        database_name=database_name,
        dataset_name=dataset_name,
        feature_property="embedding_bytes",
        nodeid_property="id",
        split_property_name="split",
        split_property_type="str",
        target_property="subject",
        feature_property_type="byte[]",
        profile=profile,
        profile_accumulator=profile_accumulator,
        node_label="Paper",
    )

    # Model input dimension = feature_dim * (hops + 1) = 1433 * 3 for hops=2.
    feature_dim = 1433
    model_args = {
        "feature_dim":  feature_dim,
        "hops":         hops,
        "hidden_dim1":  12,
        "hidden_dim2":  12,
        "nbr_classes":  7,
        "init_weights": config.get("init_weights", True),
    }
    model = SIGNPostAggregation(**model_args)

    measurer.write_to_configresult("model", {"name": "SIGNPostAggregation", "args": model_args})
    measurer.write_to_configresult("sampler", {
        "name": "Neo4jSIGNSampler",
        "hops": hops,
        "max_neighbors_per_hop": max_neighbors_per_hop,
    })
    measurer.write_to_configresult("feature_store", "Neo4jSIGNFeatureStore")
    measurer.write_to_configresult("graph_store", "Neo4SingleGS")
    measurer.write_to_configresult("dataset", dataset_name)

    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=False,
        shuffle=config.get("shuffle"),
    )
    measurer.write_to_configresult("nodeloader_args", nodeloader_args)

    trainer = Trainer(
        model=model,
        data=(feature_store, graph_store),
        sampler=sampler,
        min_delta=config.get("min_delta"),
        lr=config.get("lr"),
        patience=config.get("patience"),
        batch_size=config.get("batch_size"),
        nodeloader_args=nodeloader_args,
        measurer=measurer,
        cpu_monitor_interval=1 if config.get("logg_cpu_utilization", True) else None,
        max_training_size=config.get("max_training_size"),
        max_validation_size=config.get("max_validation_size"),
        max_test_size=config.get("max_test_size"),
    )

    trainer.train(max_epochs=config.get("max_epochs"))


if __name__ == "__main__":
    config_path = "src/train_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    main(config)
