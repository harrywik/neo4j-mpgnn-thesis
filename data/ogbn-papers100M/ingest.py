import pyarrow as pa
from graphdatascience import GraphDataScience
import numpy as np
from tqdm import tqdm
import os
from ogb.nodeproppred import NodePropPredDataset
from graphdatascience import GraphDataScience
from pathlib import Path

# Get the directory where the script is located
SCRIPT_DIR = Path(__file__).resolve().parent

print("Initializing OGB dataset...")
dataset = NodePropPredDataset(name='ogbn-papers100M', root=SCRIPT_DIR)

# PATH VALIDATION
base_path = SCRIPT_DIR / "ogbn_papers100M" / "raw"
feat_path = base_path / 'node-feat.npy'
edge_path = base_path / 'edge.npy'

for file in os.listdir(str(base_path)):
    if file.endswith(".gz"):
        file_to_rem = base_path / file
        print(f"Removing compressed file to save space: {str(file)}")
        os.remove(str(file_to_rem))

NODE_CHUNK = 100_000
EDGE_CHUNK = 1_000_000

# Connect to GDS Enterprise
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "database-password"))

feats = np.load(str(feat_path), mmap_mode='r')
num_nodes = feats.shape[0]

edges = np.load(str(edge_path), mmap_mode='r') 
total_edges = edges.shape[1]

def node_generator(chunk_size=NODE_CHUNK):
    for i in range(0, num_nodes, chunk_size):
        end = min(i + chunk_size, num_nodes)
        
        # Use PyArrow for zero-copy efficiency
        # Convert features to 'binary' type for Neo4j ByteArray storage
        node_ids = pa.array(np.arange(i, end), type=pa.int64())
        features = pa.array([feats[j].tobytes() for j in range(i, end)], type=pa.binary())
        
        yield pa.Table.from_arrays([node_ids, features], names=["nodeId", "features"])

def edge_generator(chunk_size=EDGE_CHUNK): # Adapt to RAM size
    for i in range(0, total_edges, chunk_size):
        end = min(i + chunk_size, total_edges)
        # OGB is usually (2, num_edges); slice and convert
        source = pa.array(edges[0, i:end], type=pa.int64())
        target = pa.array(edges[1, i:end], type=pa.int64())
        
        yield pa.Table.from_arrays([source, target], names=["source", "target"])

# Project to GDS Memory via Arrow Flight
G, result = gds.alpha.graph.project.arrow(
    graph_name="papers_graph",
    nodes=tqdm(node_generator(), total=num_nodes // NODE_CHUNK, desc="Streaming Nodes"),
    relationships=tqdm(edge_generator(), total=total_edges // EDGE_CHUNK, desc="Streaming Edges")
)

# Export to Disk
# IMPORTANT: Exporting 1.6B edges can take significant time. 
# Ensure your SSD has at least 300GB of free space before running.
gds.graph.export(G, dbName="papers100M")