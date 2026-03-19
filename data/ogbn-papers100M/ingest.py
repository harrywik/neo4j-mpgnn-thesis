import os
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase
from dotenv import load_dotenv
from ogb.nodeproppred import NodePropPredDataset

load_dotenv()

DATABASE = "papers100M"
NODE_BATCH_SIZE = 10_000
EDGE_BATCH_SIZE = 10_000
SCRIPT_DIR = Path(__file__).resolve().parent

def ingest_papers100M(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Load split info via OGB (lightweight — does not load features into RAM)
    print("Loading split info...")
    dataset = NodePropPredDataset(name="ogbn-papers100M", root=str(SCRIPT_DIR))
    split_idx = dataset.get_idx_split()

    # Load raw arrays via mmap — data stays on disk, only accessed slices enter RAM
    raw_dir = SCRIPT_DIR / "ogbn_papers100M" / "raw"
    feat_path = raw_dir / "node-feat.npy"
    label_path = raw_dir / "node-label.npy"
    edge_path = raw_dir / "edge.npy"

    print("Memory-mapping raw numpy files...")
    features = np.load(str(feat_path), mmap_mode="r")   # (111059956, 128) float16
    labels = np.load(str(label_path), mmap_mode="r")    # (111059956, 1)
    edge_index = np.load(str(edge_path), mmap_mode="r") # (2, 1615685872)
    num_nodes = features.shape[0]
    num_edges = edge_index.shape[1]

    # Build split lookup as int8 to minimise RAM (0=unknown, 1=train, 2=val, 3=test)
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
        for start in range(0, num_nodes, NODE_BATCH_SIZE):
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
        for start in range(0, num_edges, EDGE_BATCH_SIZE):
            end = min(start + EDGE_BATCH_SIZE, num_edges)
            edge_batch = [
                {"src": int(edge_index[0, i]), "dst": int(edge_index[1, i])}
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS edge
            MATCH (p1:Paper {id: edge.src})
            MATCH (p2:Paper {id: edge.dst})
            MERGE (p1)-[:CITES]->(p2)
            """, batch=edge_batch)
            if start % 50_000_000 == 0:
                print(f"  Edges {start:,}–{end:,} done")

    driver.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_papers100M(os.environ["URI"], os.environ["USERNAME"], os.environ["PASSWORD"])
