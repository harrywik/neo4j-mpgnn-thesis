import os
import torch
import numpy as np
from neo4j import GraphDatabase
from dotenv import load_dotenv
from torch_geometric.datasets import Coauthor
from tqdm import tqdm

load_dotenv()

DATABASE = "coauthor"
BATCH_SIZE = 5_000
SEED = 42

def ingest_coauthor_physics(uri, user, password):
    # Set seed for reproducible splits
    np.random.seed(SEED)
    
    driver = GraphDatabase.driver(uri, auth=(user, password))

    print("Loading Coauthor Physics dataset...")
    # This will download the dataset if it doesn't exist in the root
    dataset = Coauthor(root="data/coauthor", name="Physics")
    graph = dataset[0]

    features = graph.x.numpy().astype(np.float32)
    labels = graph.y.numpy().astype(int)
    num_nodes = features.shape[0]

    # Generate random splits: 60% train, 20% val, 20% test
    # Coauthor datasets do not have standard splits in PyG
    indices = np.random.permutation(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size : train_size + val_size]
    test_idx = indices[train_size + val_size :]

    split = np.full(num_nodes, "unknown", dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"

    print(f"Target Database: {DATABASE}")
    
    with driver.session(database=DATABASE) as session:
        session.run("CREATE INDEX author_id IF NOT EXISTS FOR (a:Author) ON (a.id)")
        print("Index on Author.id ensured.")

    # Ingest nodes in chunks
    print(f"Ingesting {num_nodes} nodes...")
    with driver.session(database=DATABASE) as session:
        for start in tqdm(range(0, num_nodes, BATCH_SIZE), desc="Nodes"):
            end = min(start + BATCH_SIZE, num_nodes)
            batch = [
                {
                    "id": int(i),
                    "field": int(labels[i]),
                    "feature_vector": features[i].tobytes(),
                    "split": str(split[i]),
                }
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS item
            MERGE (a:Author {id: item.id})
            SET a.field        = item.field,
                a.feature_vector = item.feature_vector,
                a.split          = item.split
            """, batch=batch)

    # Ingest edges in chunks
    edge_index = graph.edge_index.numpy()
    num_edges = edge_index.shape[1]
    print(f"Ingesting {num_edges} edges...")
    with driver.session(database=DATABASE) as session:
        for start in tqdm(range(0, num_edges, BATCH_SIZE), desc="Edges"):
            end = min(start + BATCH_SIZE, num_edges)
            edge_batch = [
                {"src": int(edge_index[0, i]), "dst": int(edge_index[1, i])}
                for i in range(start, end)
            ]
            session.run("""
            UNWIND $batch AS edge
            MATCH (a1:Author {id: edge.src})
            MATCH (a2:Author {id: edge.dst})
            MERGE (a1)-[:COAUTHORED_WITH]->(a2)
            """, batch=edge_batch)

    driver.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    uri = os.getenv("URI", "bolt://localhost:7687")
    user = os.getenv("USERNAME", "neo4j")
    password = os.getenv("PASSWORD", "password")
    
    ingest_coauthor_physics(uri, user, password)
