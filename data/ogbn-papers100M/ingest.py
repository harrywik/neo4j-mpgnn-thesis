import pyarrow as pa
from graphdatascience import GraphDataScience
import numpy as np
from tqdm import tqdm
import os

# Connect to GDS Enterprise
gds = GraphDataScience("bolt://localhost:7687", auth=("neo4j", "database-password"))

raw_dir = './ogbn_papers100M/raw'
feats = np.load(os.path.join(raw_dir, 'node-feat.npy'), mmap_mode='r')
num_nodes = feats.shape[0]

edge_path = os.path.join(raw_dir, "edge.npy")
edges = np.load(edge_path, mmap_mode='r') 
total_edges = edges.shape[1]

def node_generator(chunk_size=100_000):
    for i in range(0, num_nodes, chunk_size):
        end = min(i + chunk_size, num_nodes)
        
        # Use PyArrow for zero-copy efficiency
        # Convert features to 'binary' type for Neo4j ByteArray storage
        node_ids = pa.array(np.arange(i, end), type=pa.int64())
        features = pa.array([feats[j].tobytes() for j in range(i, end)], type=pa.binary())
        
        yield pa.Table.from_arrays([node_ids, features], names=["nodeId", "features"])

def edge_generator(chunk_size=1_000_000): # Adapt to RAM size
    for i in range(0, total_edges, chunk_size):
        end = min(i + chunk_size, total_edges)
        # OGB is usually (2, num_edges); slice and convert
        source = pa.array(edges[0, i:end], type=pa.int64())
        target = pa.array(edges[1, i:end], type=pa.int64())
        
        yield pa.Table.from_arrays([source, target], names=["source", "target"])

# Project to GDS Memory via Arrow Flight
G, result = gds.alpha.graph.project.arrow(
    graph_name="papers_graph",
    nodes=tqdm(node_generator(), total=num_nodes // 500_000, desc="Streaming Nodes"),
    relationships=tqdm(edge_generator(), total=total_edges // 2_000_000, desc="Streaming Edges")
)

# Export to Disk
# IMPORTANT: Exporting 1.6B edges can take significant time. 
# Ensure your SSD has at least 300GB of free space before running.
gds.graph.export(G, dbName="papers100M")