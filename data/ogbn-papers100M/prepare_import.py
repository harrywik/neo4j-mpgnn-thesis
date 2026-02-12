import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path

# This will trigger the download (~56 GB) and extraction.
dataset = NodePropPredDataset(name='ogbn-papers100M', root=str(Path(__file__).resolve().parent / 'dataset'))

# To access the raw graph data (once download completes)
graph = dataset[0] 

labels = dataset.labels.flatten()  # Node labels
labels_int = np.nan_to_num(labels, nan=-1).astype(np.int32) # Originally float type with NaN-values present
split_dict = dataset.get_idx_split()

num_nodes = graph['num_nodes']
features = graph['node_feat']  # This is usually a memory-mapped numpy array

# Map Splits to Integers (0: train, 1: val, 2: test, -1: unlabeled)
# Note: In Papers100M, only a subset of nodes have labels/splits.
split_types = np.full(num_nodes, -1, dtype=np.int8)
split_types[split_dict['train']] = 0
split_types[split_dict['valid']] = 1
split_types[split_dict['test']] = 2

# Define the Schema for Neo4j Native Vector
# Column names must match the neo4j-admin header expectations
schema = pa.schema([
    ('paperId:ID(Paper)', pa.int64()),
    ('features:VECTOR<FLOAT32>(128)', pa.list_(pa.float32(), 128)),
    ('label:INT', pa.int32()),
    ('split:INT', pa.int8())
])

# Process in Chunks (e.g., 1 million nodes at a time)
CHUNK_SIZE = 1_000_000
with pq.ParquetWriter('papers_nodes.parquet', schema) as writer:
    for start in range(0, num_nodes, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_nodes)
        
        # Prepare the chunk
        chunk_df = pd.DataFrame({
            'paperId:ID(Paper)': np.arange(start, end),
            'features:VECTOR<FLOAT32>(128)': [features[i] for i in range(start, end)],
            'label:INT': labels_int[start:end],
            'split:INT': split_types[start:end]
        })
        
        table = pa.Table.from_pandas(chunk_df, schema=schema)
        writer.write_table(table)
        print(f"Progress: {end}/{num_nodes} nodes converted.")

edge_index = graph['edge_index'] # Shape (2, 1.6B)
num_edges = edge_index.shape[1]

# Define the Schema of the relations
# Column names must match the header CSV
schema = pa.schema([
    (':START_ID(Paper)', pa.int64()),
    (':END_ID(Paper)', pa.int64())
])

# Write in Chunks
CHUNK_SIZE = 5_000_000 # Larger chunks for edges are okay as they are simple pairs
with pq.ParquetWriter('citations_data.parquet', schema) as writer:
    for start in range(0, num_edges, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_edges)
        
        # Transpose the OGB format [2, N] to [N, 2]
        chunk_df = pd.DataFrame({
            ':START_ID(Paper)': edge_index[0, start:end],
            ':END_ID(Paper)': edge_index[1, start:end]
        })
        
        table = pa.Table.from_pandas(chunk_df, schema=schema)
        writer.write_table(table)
        
        if end % 100_000_000 == 0:
            print(f"Progress: {end:,} / {num_edges:,} edges converted.")
