import os
import numpy as np
from neo4j import GraphDatabase
from pathlib import Path
from dotenv import load_dotenv
from ogb.nodeproppred import NodePropPredDataset

load_dotenv()

BATCH_SIZE = 5_000  # tune this — 5k nodes per transaction is safe

def ingest_arxiv(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    print("Loading ogbn-arxiv dataset...")
    dataset = NodePropPredDataset(name="ogbn-arxiv", root="data/arxiv")
    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    features = graph["node_feat"].astype(np.float32)  # (169343, 128)
    labels = labels.flatten().astype(int)             # (169343,)
    num_nodes = features.shape[0]

    # Build split array using same string labels as cora
    split = np.full(num_nodes, "unknown", dtype=object)
    split[split_idx["train"]] = "train"
    split[split_idx["val"]] = "val"
    split[split_idx["test"]] = "test"

    # Ingest nodes in chunks
    print(f"Ingesting {num_nodes} nodes in batches of {BATCH_SIZE}...")
    with driver.session(database="arxiv2") as session:
        for start in range(0, num_nodes, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_nodes)
            batch = [
                {
                    "id": int(i),
                    "subject": int(labels[i]),
                    "embedding": features[i].tobytes(),
                    "split": str(split[i]),
                }
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS item
            MERGE (p:Paper {id: item.id})
            SET p.subject   = item.subject,
                p.embedding = item.embedding,
                p.split     = item.split
            """, batch=batch)
            print(f"  Nodes {start}–{end} done")

    # Ingest edges in chunks
    edge_index = graph["edge_index"]  # (2, num_edges)
    num_edges = edge_index.shape[1]
    print(f"Ingesting {num_edges} edges in batches of {BATCH_SIZE}...")
    with driver.session(database="arxiv") as session:
        for start in range(0, num_edges, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_edges)
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
            print(f"  Edges {start}–{end} done")

    driver.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_arxiv(os.environ["URI"], os.environ["USERNAME"], os.environ["PASSWORD"])