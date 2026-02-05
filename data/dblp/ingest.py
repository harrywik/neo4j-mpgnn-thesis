import numpy as np
from torch_geometric.datasets import DBLP
from neo4j import GraphDatabase


def ingest_to_neo4j(uri, user, password, db_name="dblp"):
    dataset = DBLP(root='./data/DBLP')
    hetero_data = dataset[0]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session(database=db_name) as session:
        # Ingest Nodes
        for node_type in hetero_data.node_types:
            print(f"Ingesting node type: {node_type}")
            
            # Extract features and IDs
            # PyG IDs are implicit (0 to N-1)
            num_nodes = hetero_data[node_type].num_nodes
            # We batch the nodes for efficiency
            node_list = []
            for i in range(num_nodes):
                node_data = {"id": i}
                # Add features if they exist
                if 'x' in hetero_data[node_type]:
                    # Remember here we store as .tobytes() for efficency
                    node_data["features"] = hetero_data[node_type].x[i].numpy().astype(np.float32).tobytes()
                if 'y' in hetero_data[node_type]:
                    # Label is needed to classify author's area of research
                    # Convert torch scalar to int for Neo4j
                    node_data["y"] = hetero_data[node_type].y[i].item()
                
                # Add split info if it exists (for Author nodes in DBLP)
                if 'train_mask' in hetero_data[node_type]:
                    if hetero_data[node_type].train_mask[i]: node_data["split"] = "train"
                    elif hetero_data[node_type].val_mask[i]: node_data["split"] = "val"
                    elif hetero_data[node_type].test_mask[i]: node_data["split"] = "test"
                
                node_list.append(node_data)

            # Cypher Unwind for Nodes
            query = f"""
            UNWIND $batch AS row
            CALL (row) {{
                MERGE (n:{node_type.capitalize()} {{id: row.id}})
                SET n += row
            }} IN TRANSACTIONS OF 500 ROWS
            """
            session.run(query, batch=node_list)

        # Ingest Edges
        for edge_type in hetero_data.edge_types:
            # edge_type is a triplet: (src, rel, dst)
            src_type, rel_name, dst_type = edge_type
            # rel_name = 'to'
            rel_name = f"{src_type}_{rel_name}_{dst_type}".upper() # e.g. 'AUTHOR_TO_PAPER'
            src_type = src_type.capitalize()
            dst_type = dst_type.capitalize()

            print(f"Ingesting edge type: {edge_type}")
            
            edge_index = hetero_data[edge_type].edge_index
            edge_list = []
            for i in range(edge_index.shape[1]):
                edge_list.append({
                    "src": edge_index[0, i].item(),
                    "dst": edge_index[1, i].item()
                })

            # Cypher Unwind for Edges
            # Note: We match based on the IDs we set for the nodes
            query = f"""
            UNWIND $batch AS row
            CALL (row) {{
                MATCH (src:{src_type} {{id: row.src}})
                MATCH (dst:{dst_type} {{id: row.dst}})
                MERGE (src)-[:{rel_name.upper()}]->(dst)
            }} IN TRANSACTIONS OF 1000 ROWS
            """
            session.run(query, batch=edge_list)

    driver.close()
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_to_neo4j("bolt://localhost:7687", "neo4j", "thesis-db-0-pw")