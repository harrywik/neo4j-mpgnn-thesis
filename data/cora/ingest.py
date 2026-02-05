import tarfile
import pandas as pd
import numpy as np
import io
from neo4j import GraphDatabase
from pathlib import Path

def ingest_cora_binary(tgz_path, uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with tarfile.open(tgz_path, "r:gz") as tar:
        # Find the .content file inside the archive
        content_file_name = [m.name for m in tar.getmembers() if "cora.content" in m.name][0]
        content_bytes = tar.extractfile(content_file_name).read()
        
        # Read into Pandas
        df = pd.read_csv(io.BytesIO(content_bytes), sep='\t', header=None)
        
        paper_ids = df[0].values.astype(int)
        subjects = df.iloc[:, -1].values.astype(str)
        # Features are everything between ID and Subject
        # Cora: 1433 features
        features = df.iloc[:, 1:-1].values.astype(np.float32)

        print(f"Ingesting {len(df)} nodes...")
        with driver.session() as session:
            # Preparing the batch: ID, Subject, and RAW BYTES
            batch = [
                {
                    "id": int(paper_ids[i]), 
                    "sub": subjects[i], 
                    "bin": features[i].tobytes() # <--- RAW FLOAT32 BYTES
                } 
                for i in range(len(df))
            ]
            
            session.run("""
            UNWIND $batch AS item
            MERGE (p:Paper {id: item.id})
            SET p.subject = item.sub,
                p.embedding = item.bin
            """, batch=batch)

        # Load the Cites (Edges) file
        cites_file_name = [m.name for m in tar.getmembers() if "cora.cites" in m.name][0]
        cites_bytes = tar.extractfile(cites_file_name).read()
        edges_df = pd.read_csv(io.BytesIO(cites_bytes), sep='\t', header=None)
        
        print(f"Ingesting {len(edges_df)} relationships...")
        with driver.session() as session:
            edge_batch = [
                {"source": int(row[0]), "target": int(row[1])} 
                for row in edges_df.values
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