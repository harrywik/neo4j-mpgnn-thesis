import argparse
import io
import os
import time
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

DATABASE = "papers100M"
NODE_BATCH_SIZE = 50_000
EDGE_BATCH_SIZE = 200_000
TX_TIMEOUT_SECONDS = 3600
SCRIPT_DIR = Path("/var/lib/neo4j/data/ogbn-papers100M")
RAW_DIR = SCRIPT_DIR / "ogbn_papers100M" / "raw"
SPLIT_DIR = SCRIPT_DIR / "ogbn_papers100M" / "split" / "time"


def load_split_idx() -> dict:
    """Load train/valid/test node indices directly from CSV files.

    Bypasses OGB's NodePropPredDataset.pre_process(), which loads the entire
    graph into RAM and caused an OOM kill on this machine.
    """
    result = {}
    for name in ("train", "valid", "test"):
        csv_path = SPLIT_DIR / f"{name}.csv.gz"
        result[name] = pd.read_csv(str(csv_path), compression="gzip", header=None).values.flatten()
    return result


def _parse_npy_header(f) -> tuple:
    """Read the header of a .npy stream and return (shape, dtype).

    Leaves `f` positioned at the first byte of raw array data.
    Supports .npy format versions 1.0 and 2.0.
    """
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError(f"Not a valid .npy stream (magic={magic!r})")
    major, minor = f.read(1)[0], f.read(1)[0]
    hlen_bytes = f.read(2) if major == 1 else f.read(4)
    hlen = int.from_bytes(hlen_bytes, "little")
    header = f.read(hlen).decode("latin1").strip()
    import ast
    d = ast.literal_eval(header)
    if d["fortran_order"]:
        raise ValueError("Fortran-order arrays are not supported")
    return tuple(d["shape"]), np.dtype(d["descr"])


def stream_npy_chunks(npz_path: Path, key: str, chunk_rows: int):
    """Yield (start_row, chunk) from a .npy entry inside a .npz, chunk by chunk.

    Works with DEFLATE-compressed .npz files. Never loads the full array into
    RAM — at most `chunk_rows` rows are resident at any time.
    """
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        with zf.open(key + ".npy") as f:
            shape, dtype = _parse_npy_header(f)
            row_nbytes = int(np.prod(shape[1:])) * dtype.itemsize
            total_rows = shape[0]
            start = 0
            while start < total_rows:
                n = min(chunk_rows, total_rows - start)
                raw = f.read(n * row_nbytes)
                if len(raw) != n * row_nbytes:
                    raise RuntimeError(
                        f"Unexpected EOF reading '{key}' at row {start}: "
                        f"expected {n * row_nbytes} bytes, got {len(raw)}"
                    )
                chunk = np.frombuffer(raw, dtype=dtype).reshape(n, *shape[1:])
                yield start, chunk
                start += n


def load_small_array(npz_path: Path, key: str) -> np.ndarray:
    """Fully load a small array from a .npz into RAM."""
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        with zf.open(key + ".npy") as f:
            return np.load(io.BytesIO(f.read()))


def load_edge_arrays(npz_path: Path, key: str) -> tuple:
    """Load src and dst node ID arrays from an edge_index .npy stored in a .npz.

    ogbn-papers100M stores edge_index as shape (2, E), not (E, 2), so we
    cannot stream it row-by-row as edge pairs.  Loading the full 25.8 GB
    uncompressed (~7.1 GB on disk) is safe on this 251 GB machine.
    Returns (src_array, dst_array, num_edges).
    """
    print("  Loading edge index into RAM (~25.8 GB uncompressed)...")
    with zipfile.ZipFile(str(npz_path), "r") as zf:
        with zf.open(key + ".npy") as f:
            shape, dtype = _parse_npy_header(f)
            data = np.frombuffer(f.read(), dtype=dtype).reshape(shape)
    if shape[0] == 2 and shape[1] != 2:
        # (2, E): row 0 = all src IDs, row 1 = all dst IDs
        return data[0], data[1], int(shape[1])
    else:
        # (E, 2): column 0 = src, column 1 = dst
        return data[:, 0], data[:, 1], int(shape[0])


def ingest_papers100M(uri, user, password, skip_nodes: bool = False):
    driver = GraphDatabase.driver(
        uri,
        auth=(user, password),
        connection_acquisition_timeout=300,
        connection_timeout=60,
    )

    print("Loading split info from CSV files...")
    split_idx = load_split_idx()

    num_nodes = int(load_small_array(RAW_DIR / "data.npz", "num_nodes_list")[0])
    num_edges = int(load_small_array(RAW_DIR / "data.npz", "num_edges_list")[0])
    print(f"Dataset: {num_nodes:,} nodes, {num_edges:,} edges")

    # int8 split lookup: 0=unknown, 1=train, 2=val, 3=test  (~106 MB)
    print("Building split lookup...")
    _SPLIT_NAMES = {0: "unknown", 1: "train", 2: "val", 3: "test"}
    split = np.zeros(num_nodes, dtype=np.int8)
    split[split_idx["train"]] = 1
    split[split_idx["valid"]] = 2
    split[split_idx["test"]] = 3

    with driver.session(database=DATABASE) as session:
        session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")
        print("Index on Paper.id ensured.")

    # Ingest nodes in chunks
    print(f"Ingesting {num_nodes:,} nodes in batches of {NODE_BATCH_SIZE:,}...")
    with driver.session(database=DATABASE) as session:
        # for start in range(0, num_nodes, NODE_BATCH_SIZE):
        for start in range(num_edges - EDGE_BATCH_SIZE, -1, -EDGE_BATCH_SIZE):
            end = min(start + NODE_BATCH_SIZE, num_nodes)
            batch = [
                {
                    "id": int(i),
                    "subject": int(labels[i]),
                    "feature_vector": features[i].astype(np.float32).tobytes(),
                    "split": _SPLIT_NAMES[int(split[i])],
                }
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS item
            MERGE (p:Paper {id: item.id})
            SET p.subject        = item.subject,
                p.feature_vector = item.feature_vector,
                p.split          = item.split
            """, batch=batch)
            if start % 500_000 == 0:
                print(f"  Nodes {start:,}–{end:,} done")

    # Ingest edges in chunks
    print(f"Ingesting {num_edges:,} edges in batches of {EDGE_BATCH_SIZE:,}...")
    with driver.session(database=DATABASE) as session:
        # for start in range(0, num_edges_actual, EDGE_BATCH_SIZE):
        for start in range(num_edges - EDGE_BATCH_SIZE, -1, -EDGE_BATCH_SIZE):
            end = min(start + EDGE_BATCH_SIZE, num_edges_actual)
            edge_batch = [
                {"src": int(src_arr[i]), "dst": int(dst_arr[i])}
                for i in range(start, end)
            ]
            session.run(
                """
                UNWIND $batch AS edge
                MATCH (p1:Paper {id: edge.src})
                MATCH (p2:Paper {id: edge.dst})
                MERGE (p1)-[:CITES]->(p2)
                """,
                batch=edge_batch,
                timeout=TX_TIMEOUT_SECONDS,
            )
            if start % 50_000_000 == 0:
                print(f"  Edges {start:,}–{end:,} done")

    driver.close()
    print("Ingestion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edges-only",
        action="store_true",
        help="Skip node ingestion and only ingest edges (use when nodes are already loaded).",
    )
    args = parser.parse_args()
    ingest_papers100M(
        os.environ["URI"],
        os.environ["USERNAME"],
        os.environ["PASSWORD"],
        skip_nodes=args.edges_only,
    )
