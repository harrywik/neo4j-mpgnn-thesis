import numpy as np
from neo4j import GraphDatabase
import os
from torch_geometric.datasets import Planetoid

# Create this database first
DATABASE = "citeseer"  # The name of the database in Neo4j

def ingest_citeseer_binary(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Ingest directly from Planetoid to keep IDs aligned with masks
    print("Downloading/Loading CiteSeer dataset...")
    dataset = Planetoid(root="data/Planetoid", name="CiteSeer")
    graph = dataset[0]

    features = graph.x.cpu().numpy().astype(np.float32)
    labels = graph.y.cpu().numpy().astype(int)
    num_nodes = features.shape[0]

    split = np.full(num_nodes, "unknown", dtype=object)
    split[graph.train_mask.cpu().numpy()] = "train"
    split[graph.val_mask.cpu().numpy()] = "val"
    split[graph.test_mask.cpu().numpy()] = "test"

    print(f"Ingesting {num_nodes} nodes into database '{DATABASE}'...")
    with driver.session(database=DATABASE) as session:
        # Create index for performance
        session.run("CREATE INDEX paper_id IF NOT EXISTS FOR (p:Paper) ON (p.id)")
        
        # Batching for nodes
        batch = [
            {
                "id": int(i),
                "sub": int(labels[i]),
                "bin": features[i].tobytes(),
                "split": split[i],
            }
            for i in range(num_nodes)
        ]

        session.run("""
        UNWIND $batch AS item
        MERGE (p:Paper {id: item.id})
        SET p.subject = item.sub,
            p.embedding = item.bin,
            p.split = item.split
        """, batch=batch)

    edge_index = graph.edge_index.cpu().numpy()
    num_edges = edge_index.shape[1]
    print(f"Ingesting {num_edges} relationships into database '{DATABASE}'...")
    
    # Batching for edges
    BATCH_SIZE = 5000
    with driver.session(database=DATABASE) as session:
        for i in range(0, num_edges, BATCH_SIZE):
            end = min(i + BATCH_SIZE, num_edges)
            edge_batch = [
                {"source": int(edge_index[0, j]), "target": int(edge_index[1, j])}
                for j in range(i, end)
            ]
            session.run("""
            UNWIND $batch AS edge
            MATCH (p1:Paper {id: edge.source})
            MATCH (p2:Paper {id: edge.target})
            MERGE (p1)-[:CITES]->(p2)
            """, batch=edge_batch)
            print(f"  Edges {i} to {end} ingested...")

    driver.close()
    print("Ingestion Successful!")

if __name__ == "__main__":
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "thesis-db-0-pw")
    
    ingest_citeseer_binary(uri, user, password)
