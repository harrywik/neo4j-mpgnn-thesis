import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent / 'dataset'
RAW_DATA_DIR = BASE_DIR / 'ogbn_papers100M' / 'raw'

# --- INITIALIZE DATA ACCESS ---
print("Initializing Memory-Mapped Data Access...")
dataset_path = RAW_DATA_DIR / 'data.npz'
label_path = RAW_DATA_DIR / 'node-label.npz'
split_dir = BASE_DIR / 'ogbn_papers100M' / 'split' / 'time'

# Use mmap_mode='r' to keep the 56GB on disk
data = np.load(dataset_path, mmap_mode='r')

node_feat = data['node_feat']
edge_index = data['edge_index']
num_nodes = node_feat.shape[0] 
print(f"Detected {num_nodes:,} nodes and {edge_index.shape[1]:,} edges.")

# --- LOAD LABELS AND SPLITS ---
# These are small enough to fit in RAM comfortably (~1-2 GB total)
print("Loading labels and splits into RAM...")
labels = np.load(label_path, mmap_mode='r')['node_label']
labels_int = np.nan_to_num(labels, nan=-1).astype(np.int32).flatten()

split_types = np.full(num_nodes, -1, dtype=np.int8)
for i, s in enumerate(['train', 'valid', 'test']):
    s_path = split_dir / f'{s}.csv.gz'
    if s_path.exists():
        idx = pd.read_csv(s_path, compression='gzip', header=None).values.flatten()
        split_types[idx] = i

# --- 3. NODE CONVERSION ---
node_schema = pa.schema([
    ('paperId:ID(Paper)', pa.int64()),
    ('features:VECTOR<FLOAT32>(128)', pa.list_(pa.float32(), 128)),
    ('label:INT', pa.int32()),
    ('split:INT', pa.int8())
])

# CHUNK_SIZE optimization: 100k reduces the RES spike compared to 500k
CHUNK_SIZE = 100_000 
print(f"Converting nodes...")

with pq.ParquetWriter('papers_nodes.parquet', node_schema) as writer:
    for start in range(0, num_nodes, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, num_nodes)
        
        # We slice first, then convert to list to minimize the time Python 
        # objects spend sitting in the heap
        feat_slice = node_feat[start:end].astype(np.float32)
        
        chunk_df = pd.DataFrame({
            'paperId:ID(Paper)': np.arange(start, end, dtype=np.int64),
            'features:VECTOR<FLOAT32>(128)': feat_slice.tolist(),
            'label:INT': labels_int[start:end],
            'split:INT': split_types[start:end]
        })
        
        writer.write_table(pa.Table.from_pandas(chunk_df, schema=node_schema))
        if end % 5_000_000 == 0 or end == num_nodes:
            print(f"Nodes: {end:,}/{num_nodes:,}")

# --- EDGE CONVERSION ---
num_edges = edge_index.shape[1]
edge_schema = pa.schema([
    (':START_ID(Paper)', pa.int64()),
    (':END_ID(Paper)', pa.int64())
])

print(f"Converting edges...")
EDGE_CHUNK = 10_000_000
with pq.ParquetWriter('citations_data.parquet', edge_schema) as writer:
    for start in range(0, num_edges, EDGE_CHUNK):
        end = min(start + EDGE_CHUNK, num_edges)
        
        # Accessing edge_index[0] and [1] is efficient in mmap
        chunk_df = pd.DataFrame({
            ':START_ID(Paper)': edge_index[0, start:end].astype(np.int64),
            ':END_ID(Paper)': edge_index[1, start:end].astype(np.int64)
        })
        
        writer.write_table(pa.Table.from_pandas(chunk_df, schema=edge_schema))
        if end % 100_000_000 == 0 or end == num_edges:
            print(f"Edges: {end:,}/{num_edges:,}")

print("Success. All Parquet files generated.")
