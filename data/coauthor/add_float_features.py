"""Add feature_vector_floats (f64[]) to every Author node.

Loads Coauthor Physics node features (float32) from disk and writes a
decoded f64[] property directly to Neo4j — no byte[] round-trip required.

The script is fully resumable: nodes that already have feature_vector_floats
are skipped automatically.  Run it as many times as needed; it is idempotent.

Usage
-----
    python data/coauthor/add_float_features.py

Or via the Makefile shortcut:
    make add_float_features_coauthor
"""

import os
import time

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.datasets import Coauthor

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE     = "coauthor"
BATCH_SIZE   = 5_000
REPORT_EVERY = 10_000


# ---------------------------------------------------------------------------
# Fast skip: check how many nodes still need the property
# ---------------------------------------------------------------------------

def count_remaining(driver) -> int:
    with driver.session(database=DATABASE) as s:
        return s.run(
            "MATCH (n:Author) WHERE n.feature_vector_floats IS NULL RETURN count(n) AS c"
        ).single()["c"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_float_features(driver):
    print("Checking how many Author nodes still need feature_vector_floats…")
    remaining = count_remaining(driver)
    print(f"  Remaining: {remaining:,}")
    if remaining == 0:
        print("  All nodes already have feature_vector_floats — nothing to do.")
        return 0

    print("Loading Coauthor Physics features from disk…")
    dataset = Coauthor(root="data/coauthor", name="Physics")
    graph = dataset[0]
    features = graph.x.numpy().astype(np.float64)
    num_nodes = features.shape[0]
    print(f"  Loaded {num_nodes:,} × {features.shape[1]} feature matrix.")

    set_q = """
    UNWIND $rows AS row
    MATCH (n:Author {id: row.nid})
    WHERE n.feature_vector_floats IS NULL
    SET n.feature_vector_floats = row.floats
    """

    updated = 0
    t_start = time.monotonic()

    for start in range(0, num_nodes, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_nodes)
        rows = [
            {"nid": int(i), "floats": features[i].tolist()}
            for i in range(start, end)
        ]

        with driver.session(database=DATABASE) as ws:
            ws.run(set_q, rows=rows)

        updated += end - start
        if updated % REPORT_EVERY < BATCH_SIZE:
            elapsed = time.monotonic() - t_start
            rate = updated / elapsed if elapsed > 0 else 0
            eta_s = (remaining - updated) / rate if rate > 0 else 0
            print(
                f"  {updated:>10,} nodes written  "
                f"({rate:,.0f} nodes/s  ETA ~{eta_s/60:.0f} min)"
            )

    elapsed = time.monotonic() - t_start
    print(f"\nDone — {updated:,} nodes written in {elapsed/60:.1f} min.")
    return updated


if __name__ == "__main__":
    uri      = os.environ["URI"]
    user     = os.environ["USERNAME"]
    password = os.environ["PASSWORD"]

    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_acquisition_timeout=300,
        connection_timeout=60,
    )
    try:
        add_float_features(driver)
    finally:
        driver.close()
