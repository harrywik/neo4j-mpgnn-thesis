"""Add feature_vector_floats (f64[]) to every Paper node.

Reads raw float32 features from the .npz file on disk and writes a decoded
f64[] sibling property directly to Neo4j — no byte[] round-trip through the
database required.

The script is fully resumable: nodes that already have feature_vector_floats
are skipped automatically.  Run it as many times as needed; it is idempotent.

Usage
-----
    python data/ogbn-papers100M/add_float_features.py

Or via the Makefile shortcut:
    make add_float_features_papers100M
"""

import ast
import io
import os
import struct
import time
import zipfile
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATABASE       = "papers100M"
BATCH_SIZE     = 50_000           # nodes per write transaction
REPORT_EVERY   = 500_000          # print progress line every N nodes
SCRIPT_DIR     = Path("/var/lib/neo4j/data/ogbn-papers100M")
RAW_DIR        = SCRIPT_DIR / "ogbn_papers100M" / "raw"
NPZ_PATH       = RAW_DIR / "data.npz"
NODE_FEAT_KEY  = "node_feat"


# ---------------------------------------------------------------------------
# Streaming .npy reader (same approach as ingest.py — no full RAM load)
# ---------------------------------------------------------------------------

def _parse_npy_header(f):
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"Not a valid .npy stream (magic={magic!r})")
    major = f.read(1)[0]
    _minor = f.read(1)[0]
    hlen_bytes = f.read(2) if major == 1 else f.read(4)
    hlen = int.from_bytes(hlen_bytes, "little")
    header = f.read(hlen).decode("latin1").strip()
    d = ast.literal_eval(header)
    if d["fortran_order"]:
        raise ValueError("Fortran-order arrays are not supported")
    return tuple(d["shape"]), np.dtype(d["descr"])


def stream_node_feat_chunks(npz_path: Path, chunk_rows: int):
    """Yield (start_row, chunk_ndarray) from node_feat.npy inside the .npz."""
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        with zf.open(NODE_FEAT_KEY + ".npy") as f:
            shape, dtype = _parse_npy_header(f)
            row_nbytes = int(np.prod(shape[1:])) * dtype.itemsize
            total_rows = shape[0]
            start = 0
            while start < total_rows:
                n = min(chunk_rows, total_rows - start)
                raw = f.read(n * row_nbytes)
                if len(raw) != n * row_nbytes:
                    raise RuntimeError(
                        f"Unexpected EOF at row {start}: "
                        f"expected {n * row_nbytes} bytes, got {len(raw)}"
                    )
                chunk = np.frombuffer(raw, dtype=dtype).reshape(n, *shape[1:])
                yield start, chunk
                start += n


# ---------------------------------------------------------------------------
# Fast skip: check how many nodes still need the property
# ---------------------------------------------------------------------------

def count_remaining(driver) -> int:
    with driver.session(database=DATABASE) as s:
        return s.run(
            "MATCH (n:Paper) WHERE n.feature_vector_floats IS NULL RETURN count(n) AS c"
        ).single()["c"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def add_float_features(driver):
    print(f"Checking how many Paper nodes still need feature_vector_floats…")
    remaining = count_remaining(driver)
    print(f"  Remaining: {remaining:,}")
    if remaining == 0:
        print("  All nodes already have feature_vector_floats — nothing to do.")
        return 0

    set_q = """
    UNWIND $rows AS row
    MATCH (n:Paper {id: row.nid})
    WHERE n.feature_vector_floats IS NULL
    SET n.feature_vector_floats = row.floats
    """

    updated = 0
    t_start = time.monotonic()

    for start_row, chunk in stream_node_feat_chunks(NPZ_PATH, BATCH_SIZE):
        # chunk dtype is float32; cast to float64 for f64[] in Neo4j
        rows = [
            {"nid": int(start_row + i), "floats": chunk[i].astype(np.float64).tolist()}
            for i in range(len(chunk))
        ]

        with driver.session(database=DATABASE) as ws:
            ws.run(set_q, rows=rows)

        updated += len(rows)
        if updated % REPORT_EVERY == 0:
            elapsed = time.monotonic() - t_start
            rate = updated / elapsed
            eta_s = (remaining - updated) / rate if rate > 0 else 0
            print(
                f"  {updated:>12,} nodes written  "
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
