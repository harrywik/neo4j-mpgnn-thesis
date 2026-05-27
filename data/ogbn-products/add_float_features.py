"""Add feature_vector_floats (f64[]) to every Product node.

Decodes the existing feature_vector (byte[], little-endian float32, dim=100)
property already stored in Neo4j and writes a decoded f64[] sibling property.

The script is fully resumable: nodes that already have feature_vector_floats
are skipped automatically.  Run it as many times as needed; it is idempotent.

Usage
-----
    python data/ogbn-products/add_float_features.py
"""

import os
import struct
import time

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE     = "products"
BATCH_SIZE   = 5_000    # nodes per read+write transaction
REPORT_EVERY = 100_000  # print progress line every N nodes
FEATURE_DIM  = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_float32_bytes(raw: bytes) -> list[float]:
    """Decode little-endian packed float32 bytes to a list of float64."""
    n = len(raw) // 4
    return list(struct.unpack_from(f"<{n}f", raw))


def count_remaining(driver) -> int:
    with driver.session(database=DATABASE) as s:
        return s.run(
            "MATCH (n:Product) WHERE n.feature_vector_floats IS NULL RETURN count(n) AS c"
        ).single()["c"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_float_features(driver):
    print("Checking how many Product nodes still need feature_vector_floats…")
    remaining = count_remaining(driver)
    print(f"  Remaining: {remaining:,}")
    if remaining == 0:
        print("  All nodes already have feature_vector_floats — nothing to do.")
        return 0

    fetch_q = """
    MATCH (n:Product)
    WHERE n.feature_vector_floats IS NULL
    RETURN n.id AS nid, n.feature_vector AS raw
    ORDER BY n.id
    LIMIT $limit
    """
    write_q = """
    UNWIND $rows AS row
    MATCH (n:Product {id: row.nid})
    WHERE n.feature_vector_floats IS NULL
    SET n.feature_vector_floats = row.floats
    """

    updated = 0
    t_start = time.monotonic()

    while True:
        with driver.session(database=DATABASE) as s:
            records = s.run(fetch_q, limit=BATCH_SIZE).data()

        if not records:
            break

        rows = [
            {"nid": r["nid"], "floats": decode_float32_bytes(bytes(r["raw"]))}
            for r in records
            if r["raw"] is not None
        ]

        with driver.session(database=DATABASE) as s:
            s.run(write_q, rows=rows)

        updated += len(rows)

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