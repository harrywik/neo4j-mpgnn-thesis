import pyarrow as pa
import numpy as np
import gc
import os
from graphdatascience import GraphDataScience
from ogb.nodeproppred import NodePropPredDataset
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
DB_NAME = os.getenv("DB_NAME", "ogbnProducts")
NODE_CHUNK = 100_000   
EDGE_CHUNK = 2_000_000 # Increased for better throughput on 128GB RAM
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = (os.getenv("DB_USER"), os.getenv("DB_PW"))

def run_ingestion():
    # 1. Load OGB
    print("Loading OGB Products...")
    dataset = NodePropPredDataset(name='ogbn-products', root='./data/ogbn-products')
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    num_nodes = graph['num_nodes']

    # 2. Build Masks
    print("Generating split masks...")
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    # 3. Client Init
    # Port 8491 must be open for Arrow Flight
    gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH, arrow=True)
    gds.set_database("neo4j") 

    # 4. Generators
    def node_generator():
        feats = graph['node_feat']
        lbls = labels.flatten()
        for i in range(0, num_nodes, NODE_CHUNK):
            end = min(i + NODE_CHUNK, num_nodes)
            yield pa.Table.from_pydict({
                "nodeId": np.arange(i, end),
                "values": [feats[j].tolist() for j in range(i, end)], # GDS expects 'values'
                "label": lbls[i:end].tolist(),
                "train_mask": train_mask[i:end].tolist(),
                "val_mask": val_mask[i:end].tolist(),
                "test_mask": test_mask[i:end].tolist()
            })

    def edge_generator():
        edge_index = graph['edge_index']
        total_edges = edge_index.shape[1]
        for i in range(0, total_edges, EDGE_CHUNK):
            end = min(i + EDGE_CHUNK, total_edges)
            yield pa.Table.from_pydict({
                "source": edge_index[0, i:end],
                "target": edge_index[1, i:end]
            })

    # 5. The API Call
    try:
        print("Concatenating Arrow chunks (Zero-copy)...")
        # We consume the generators and merge into one Table object
        # which provides the .columns and .values attributes GDS needs.
        nodes_table = pa.concat_tables(node_generator())
        edges_table = pa.concat_tables(edge_generator())

        print(f"Streaming {num_nodes} nodes and {graph['edge_index'].shape[1]} edges via Flight...")
        
        # 'construct' now receives a single PyArrow Table for each
        G = gds.alpha.graph.construct(
            graph_name="products_graph",
            nodes={"Product": nodes_table}, # Labeling as Product
            relationships={"SHIPPED_WITH": edges_table}
        )

        # 6. Memory Cleanup
        print("Projection complete. Freeing Python RAM before export...")
        # Unbind the large Arrow tables and original data
        del nodes_table, edges_table
        del graph, labels, train_mask, val_mask, test_mask
        gc.collect() 

        # 7. Export to Neo4j Database
        print(f"Exporting to persistent database '{DB_NAME}'...")
        # Note: Ensure you have enough disk space on your 512GB SSD
        gds.graph.export(G, dbName=DB_NAME)
        
        gds.graph.drop(G)
        print("Ingestion, Projection, and Export Finished Successfully.")

    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    run_ingestion()
