"""Add embedding_bytes_floats (f64[]) to every Paper node.

Loads Cora node features (float32, dim=1433) from disk and writes a decoded
f64[] property directly to Neo4j — no byte[] round-trip required.

The script is fully resumable: nodes that already have embedding_bytes_floats
are skipped automatically.  Run it as many times as needed; it is idempotent.

Usage
-----
    python data/cora/add_float_features.py

Or via the Makefile shortcut:
    make add_float_features_cora
"""

import os
import time

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from torch_geometric.datasets import Planetoid

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE     = "neo4j"
BATCH_SIZE   = 500
REPORT_EVERY = 1_000


# ---------------------------------------------------------------------------
# Fast skip: check how many nodes still need the property
# ---------------------------------------------------------------------------

def count_remaining(driver) -> int:
    with driver.session(database=DATABASE) as s:
        return s.run(
            "MATCH (n:Paper) WHERE n.embedding_bytes_floats IS NULL RETURN count(n) AS c"
        ).single()["c"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_float_features(driver):
    print("Checking how many Paper nodes still need embedding_bytes_floats…")
    remaining = count_remaining(driver)
    print(f"  Remaining: {remaining:,}")
    if remaining == 0:
        print("  All nodes already have embedding_bytes_floats — nothing to do.")
        return 0

    print("Loading Cora features from disk…")
    dataset = Planetoid(root="data/Planetoid", name="Cora")
    graph = dataset[0]
    features = graph.x.cpu().numpy().astype(np.float64)
    num_nodes = features.shape[0]
    print(f"  Loaded {num_nodes:,} × {features.shape[1]} feature matrix.")

    set_q = """
    UNWIND $rows AS row
    MATCH (n:Paper {id: row.nid})
    WHERE n.embedding_bytes_floats IS NULL
    SET n.embedding_bytes_floats = row.floats
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
                f"  {updated:>8,} nodes written  "
                f"({rate:,.0f} nodes/s  ETA ~{eta_s:.0f} s)"
            )

    elapsed = time.monotonic() - t_start
    print(f"\nDone — {updated:,} nodes written in {elapsed:.1f} s.")
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
