"""Cora training run using the Neo4j Java UDP aggregation (Option D hybrid).

Architecture
------------
* ``custom.gcn.aggregateNeighbors`` Java procedure (neo4j-gcn-plugin.jar) runs
  1-hop mean aggregation of neighbour feature vectors entirely inside Neo4j.
* ``Neo4jAggregationSampler`` calls the UDP and caches results in
  ``sampler.pending_agg``.
* ``Neo4jUDPFeatureStore`` reads pre-aggregated ``x`` from that cache; fetches
  labels (``y``) from Neo4j as usual.
* ``GCNPostAggregation`` applies two linear layers + a classifier to the
  already-aggregated features (no GCNConv / no edge_index needed).

Prerequisites
-------------
1. Build and deploy the plugin jar::

       make build-plugin NEO4J_PLUGINS_DIR=/path/to/neo4j/plugins

2. Restart Neo4j.
3. Verify the procedure exists::

       CALL dbms.procedures()
       YIELD name WHERE name STARTS WITH 'custom.gcn' RETURN name

4. Create the uniqueness constraint (once per database)::

       CREATE CONSTRAINT paper_id IF NOT EXISTS
       FOR (p:Paper) REQUIRE p.id IS UNIQUE
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

from neo4j_pyg.models import GCNPostAggregation
from training.Training import Trainer, put_nodeLoader_args_map
from neo4j_pyg.feature_stores import Neo4jUDPFeatureStore
from neo4j_pyg.graph_stores import Neo4jMultiGS, Neo4SingleGS
from neo4j_pyg.samplers import Neo4jAggregationSampler
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

    # -----------------------------------------------------------------------
    # Sampler — calls custom.gcn.aggregateNeighbors and caches results.
    # The sampler must be created first so its pending_agg dict can be
    # passed to the feature store.
    # -----------------------------------------------------------------------
    num_neighbors = config.get("num_neighbors", [10])

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

    sampler = Neo4jAggregationSampler(
        graph_store=graph_store,
        node_id_key="id",
        feature_key="embedding_bytes",
        feature_type="byte[]",
        node_label="Paper",
        edge_type="CITES",
        max_neighbors=num_neighbors[0],
        measurer=measurer,
    )

    # -----------------------------------------------------------------------
    # Feature store — reads x from sampler.pending_agg; fetches y from DB.
    # -----------------------------------------------------------------------
    feature_store = Neo4jUDPFeatureStore(
        sampler=sampler,
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

    # -----------------------------------------------------------------------
    # Model — no GCNConv; operates on pre-aggregated feature vectors.
    # -----------------------------------------------------------------------
    model_args = {
        "in_dim": 1433,
        "hidden_dim1": 12,
        "hidden_dim2": 12,
        "nbr_classes": 7,
        "init_weights": config.get("init_weights", True),
    }
    model = GCNPostAggregation(**model_args)

    measurer.write_to_configresult("model", {"name": "GCNPostAggregation", "args": model_args})
    measurer.write_to_configresult("sampler", {"name": "Neo4jAggregationSampler",
                                               "num_neighbors": num_neighbors,
                                               "feature_key": "embedding_bytes",
                                               "feature_type": "byte[]"})
    measurer.write_to_configresult("feature_store", "Neo4jUDPFeatureStore")
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
