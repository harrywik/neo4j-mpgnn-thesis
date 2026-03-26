import numpy as np
from neo4j import GraphDatabase
from pathlib import Path
from torch_geometric.datasets import Planetoid

def ingest_cora_binary(tgz_path, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Ingest directly from Planetoid to keep IDs aligned with masks
    planetoid = Planetoid(root="data/Planetoid", name="Cora")
    graph = planetoid[0]

    features = graph.x.cpu().numpy().astype(np.float32)
    labels = graph.y.cpu().numpy().astype(int)
    num_nodes = features.shape[0]

    split = np.full(num_nodes, "unknown", dtype=object)
    split[graph.train_mask.cpu().numpy()] = "train"
    split[graph.val_mask.cpu().numpy()] = "val"
    split[graph.test_mask.cpu().numpy()] = "test"

    print(f"Ingesting {num_nodes} nodes...")
    with driver.session() as session:
        batch = [
            {
                "id": int(i),
                "sub": int(labels[i]),
                "bin": features[i].tobytes(),
                "feature_vector": features[i].tolist(),
                "split": split[i],
            }
            for i in range(num_nodes)
        ]

        session.run("""
        UNWIND $batch AS item
        MERGE (p:Paper {id: item.id})
        SET p.subject = item.sub,
            p.embedding = item.bin,
            //p.feature_vector = item.feature_vector,
            p.split = item.split
        """, batch=batch)

    edge_index = graph.edge_index.cpu().numpy()
    print(f"Ingesting {edge_index.shape[1]} relationships...")
    with driver.session() as session:
        edge_batch = [
            {"source": int(edge_index[0, i]), "target": int(edge_index[1, i])}
            for i in range(edge_index.shape[1])
        ]
        session.run("""
        UNWIND $batch AS edge
        MATCH (p1:Paper {id: edge.source})
        MATCH (p2:Paper {id: edge.target})
        MERGE (p1)-[:CITES]->(p2)
        """, batch=edge_batch)

    driver.close()
    print("Ingestion Successful!")

if __name__ == "__main__":
    path = str(Path(__file__).parent / "cora.tgz")
    ingest_cora_binary(path, "bolt://localhost:7687", "neo4j", "thesis-db-0-pw")