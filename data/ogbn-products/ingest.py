import pyarrow as pa
import numpy as np
import gc
from graphdatascience import GraphDataScience
from ogb.nodeproppred import NodePropPredDataset
import os
from dotenv import load_dotenv

# --- CONFIG ---
load_dotenv()
DB_NAME = os.getenv("DB_NAME")
NODE_CHUNK = 100_000   
EDGE_CHUNK = 1_000_000 
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = (os.getenv("DB_USER"), os.getenv("DB_PW"))

def run_ingestion():
    # 1. Load OGB
    print("Loading OGB Products...")
    dataset = NodePropPredDataset(name='ogbn-products', root='./data')
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    num_nodes = graph['num_nodes']

    # 2. Build Masks
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    # 3. Client Init
    # Set 'arrow=True' to ensure the client attempts Flight connection
    gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH, arrow=True)
    gds.set_database("neo4j") # The 'context' database for projection metadata

    # 4. Generators
    def node_generator():
        feats = graph['node_feat']
        lbls = labels.flatten()
        for i in range(0, num_nodes, NODE_CHUNK):
            end = min(i + NODE_CHUNK, num_nodes)
            yield pa.Table.from_pydict({
                "nodeId": np.arange(i, end),
                "values": [feats[j].tolist() for j in range(i, end)],
                "label": lbls[i:end].tolist(),
                "train_mask": train_mask[i:end].tolist(),
                "val_mask": val_mask[i:end].tolist(),
                "test_mask": test_mask[i:end].tolist()
            })
            if i % (NODE_CHUNK * 5) == 0: gc.collect()

    def edge_generator():
        edge_index = graph['edge_index']
        total_edges = edge_index.shape[1]
        for i in range(0, total_edges, EDGE_CHUNK):
            end = min(i + EDGE_CHUNK, total_edges)
            yield pa.Table.from_pydict({
                "source": edge_index[0, i:end],
                "target": edge_index[1, i:end]
            })
            if i % (EDGE_CHUNK * 2) == 0: gc.collect()

    # 5. The Correct API Call
    try:
        print("Starting Arrow Ingestion via gds.alpha.graph.construct...")
        # 'construct' accepts iterables of Arrow Tables and handles the Flight protocol
        G = gds.alpha.graph.construct(
            graph_name="products_graph",
            nodes=node_generator(),
            relationships=edge_generator()
        )

        # 6. Memory Cleanup
        print("Projection complete. Freeing Python RAM...")
        del graph, labels, train_mask, val_mask, test_mask
        gc.collect()

        # 7. Export to Neo4j Database
        print(f"Exporting to database '{DB_NAME}'...")
        gds.graph.export(G, dbName=DB_NAME)
        
        gds.graph.drop(G)
        print("Finished successfully.")

    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    run_ingestion()
