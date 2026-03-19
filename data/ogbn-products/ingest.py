import os
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
import torch
from ogb.nodeproppred import NodePropPredDataset

load_dotenv()

DATABASE = "products"
BATCH_SIZE = 5_000

def ingest_products(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    print("Loading ogbn-products dataset...")
    _orig_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, "weights_only": False})
    try:
        dataset = NodePropPredDataset(name="ogbn-products", root="data/ogbn-products")
    finally:
        torch.load = _orig_torch_load
    graph, labels = dataset[0]
    split_idx = dataset.get_idx_split()

    features = graph["node_feat"].astype(np.float32)  # (2449029, 100)
    labels = labels.flatten().astype(int)             # (2449029,)
    num_nodes = features.shape[0]

    split = np.full(num_nodes, "unknown", dtype=object)
    split[split_idx["train"]] = "train"
    split[split_idx["valid"]] = "val"
    split[split_idx["test"]] = "test"

    with driver.session(database=DATABASE) as session:
        session.run("CREATE INDEX product_id IF NOT EXISTS FOR (p:Product) ON (p.id)")
        print("Index on Product.id ensured.")

    # Ingest nodes in chunks
    print(f"Ingesting {num_nodes} nodes in batches of {BATCH_SIZE}...")
    with driver.session(database=DATABASE) as session:
        for start in range(0, num_nodes, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_nodes)
            batch = [
                {
                    "id": int(i),
                    "category": int(labels[i]),
                    "feature_vector": features[i].tobytes(),
                    "split": str(split[i]),
                }
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS item
            MERGE (p:Product {id: item.id})
            SET p.category       = item.category,
                p.feature_vector = item.feature_vector,
                p.split          = item.split
            """, batch=batch)
            print(f"  Nodes {start}–{end} done")

    # Ingest edges in chunks
    edge_index = graph["edge_index"]  # (2, num_edges)
    num_edges = edge_index.shape[1]
    print(f"Ingesting {num_edges} edges in batches of {BATCH_SIZE}...")
    with driver.session(database=DATABASE) as session:
        for start in range(0, num_edges, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_edges)
            edge_batch = [
                {"src": int(edge_index[0, i]), "dst": int(edge_index[1, i])}
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS edge
            MATCH (p1:Product {id: edge.src})
            MATCH (p2:Product {id: edge.dst})
            MERGE (p1)-[:CO_PURCHASED]->(p2)
            """, batch=edge_batch)
            print(f"  Edges {start}–{end} done")

    driver.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_products(os.environ["URI"], os.environ["USERNAME"], os.environ["PASSWORD"])
