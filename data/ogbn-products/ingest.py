import pyarrow as pa
import numpy as np
import gc
from tqdm import tqdm
from pathlib import Path
from ogb.nodeproppred import NodePropPredDataset
from graphdatascience import GraphDataScience

# --- SETTINGS ---
DB_NAME = "ogbnProducts"
NODE_CHUNK = 100_000   
EDGE_CHUNK = 500_000 
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "database-password")

def run_ingestion():
    # 1. Load Dataset with mmap to keep initial RAM usage at ~0
    print("Loading OGB Products (Memory-Mapped)...")
    dataset = NodePropPredDataset(name='ogbn-products', root=str(Path(__file__).resolve().parent))
    split_idx = dataset.get_idx_split()
    graph, labels = dataset[0]
    
    num_nodes = graph['num_nodes']
    node_feats = graph['node_feat'] # This is mmap'd
    edge_index = graph['edge_index'] # This is mmap'd
    
    # 2. Prepare Split Masks
    print("Preparing train/val/test masks...")
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[split_idx['train']] = True
    val_mask[split_idx['valid']] = True
    test_mask[split_idx['test']] = True

    # 3. Connect to GDS
    gds = GraphDataScience(NEO4J_URI, auth=NEO4J_AUTH)

    # 4. Define Generators with internal cleanup
    def node_generator():
        lbls = labels.flatten()
        for i in range(0, num_nodes, NODE_CHUNK):
            end = min(i + NODE_CHUNK, num_nodes)
            
            # Creating a dictionary for the Arrow Table
            batch = {
                "nodeId": np.arange(i, end),
                "values": [node_feats[j].tolist() for j in range(i, end)],
                "label": lbls[i:end].tolist(),
                "train_mask": train_mask[i:end].tolist(),
                "val_mask": val_mask[i:end].tolist(),
                "test_mask": test_mask[i:end].tolist()
            }
            
            yield pa.Table.from_pydict(batch)
            
            # Immediate cleanup of the batch dictionary and table
            del batch
            if i % (NODE_CHUNK * 5) == 0:
                gc.collect()

    def edge_generator():
        total_edges = edge_index.shape[1]
        for i in range(0, total_edges, EDGE_CHUNK):
            end = min(i + EDGE_CHUNK, total_edges)
            
            batch = {
                "source": edge_index[0, i:end],
                "target": edge_index[1, i:end]
            }
            
            yield pa.Table.from_pydict(batch)
            del batch
            if i % (EDGE_CHUNK * 2) == 0:
                gc.collect()

    # 5. Project to GDS Memory
    try:
        print(f"Streaming {num_nodes} nodes and {edge_index.shape[1]} edges to GDS...")
        G, _ = gds.alpha.graph.project.arrow(
            graph_name="products_graph",
            nodes=tqdm(node_generator(), total=num_nodes // NODE_CHUNK, desc="Nodes"),
            relationships=tqdm(edge_generator(), total=edge_index.shape[1] // EDGE_CHUNK, desc="Edges")
        )

        # 6. CRITICAL CLEANUP: Clear Python RAM before Neo4j Export starts
        print("Projection finished. Clearing Python memory before Export...")
        del graph, labels, node_feats, edge_index, train_mask, val_mask, test_mask
        gc.collect()

        # 7. Export to Disk
        print(f"Exporting GDS graph to database: {DB_NAME}...")
        gds.graph.export(G, dbName=DB_NAME)
        
        # 8. Final Drop to free Neo4j Heap
        gds.graph.drop(G)
        print("Success! Database is now ready in Neo4j.")

    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    run_ingestion()