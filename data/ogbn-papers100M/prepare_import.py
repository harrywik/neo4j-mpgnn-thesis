import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# --- CONFIGURATION ---
DATASET_NAME = 'ogbn-papers100M'
DOWNLOAD_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/papers100M-bin.zip"
BASE_DIR = Path(__file__).resolve().parent / 'dataset'
RAW_DATA_DIR = BASE_DIR / 'ogbn_papers100M' / 'raw'
EXTRACT_DIR = BASE_DIR / 'ogbn_papers100M'

# --- SETUP DIRECTORIES ---
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# --- DOWNLOAD RAW DATA (If missing) ---
zip_path = BASE_DIR / "papers100M-bin.zip"
if not (RAW_DATA_DIR / 'data.npz').exists():
    if not zip_path.exists():
        print(f"Downloading {DATASET_NAME} (56GB)... This will take a while.")
        # Using a custom reporthook for progress feedback
        def progress(count, block_size, total_size):
            if count % 1000 == 0:
                print(f"Downloaded: {count * block_size / 1e9:.2f} GB / {total_size / 1e9:.2f} GB", end='\r')
        urllib.request.urlretrieve(DOWNLOAD_URL, zip_path, reporthook=progress)
        print("\nDownload Complete.")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(BASE_DIR)
    print("Extraction Complete.")
    # Optional: os.remove(zip_path) # Uncomment to save 56GB after extraction
else:
    print("Raw data already exists. Skipping download/extraction.")

# --- MEMORY-MAPPED CONVERSION ---
print("Initializing Memory-Mapped Data Access...")
dataset_path = RAW_DATA_DIR / 'data.npz'
label_path = RAW_DATA_DIR / 'node-label.npz'
split_dir = BASE_DIR / 'ogbn_papers100M' / 'split' / 'time'

with np.load(dataset_path, mmap_mode='r') as data:
    node_feat = data['node_feat']
    edge_index = data['edge_index']
    num_nodes = data['num_nodes'].item()

    # Node Conversion
    node_schema = pa.schema([
        ('paperId:ID(Paper)', pa.int64()),
        ('features:VECTOR<FLOAT32>(128)', pa.list_(pa.float32(), 128)),
        ('label:INT', pa.int32()),
        ('split:INT', pa.int8())
    ])

    # Load Splits
    split_types = np.full(num_nodes, -1, dtype=np.int8)
    for i, s in enumerate(['train', 'valid', 'test']):
        idx = pd.read_csv(split_dir / f'{s}.csv.gz', compression='gzip', header=None).values.flatten()
        split_types[idx] = i

    labels = np.load(label_path, mmap_mode='r')['node_label']
    labels_int = np.nan_to_num(labels, nan=-1).astype(np.int32).flatten()

    print(f"Converting {num_nodes:,} nodes...")
    CHUNK_SIZE = 500_000
    with pq.ParquetWriter('papers_nodes.parquet', node_schema) as writer:
        for start in range(0, num_nodes, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_nodes)
            chunk_df = pd.DataFrame({
                'paperId:ID(Paper)': np.arange(start, end, dtype=np.int64),
                'features:VECTOR<FLOAT32>(128)': list(node_feat[start:end]),
                'label:INT': labels_int[start:end],
                'split:INT': split_types[start:end]
            })
            writer.write_table(pa.Table.from_pandas(chunk_df, schema=node_schema))
            if end % 5_000_000 == 0: print(f"Nodes: {end:,}/{num_nodes:,}")

    # Edge Conversion
    num_edges = edge_index.shape[1]
    edge_schema = pa.schema([(':START_ID(Paper)', pa.int64()), (':END_ID(Paper)', pa.int64())])
    
    print(f"Converting {num_edges:,} edges...")
    with pq.ParquetWriter('citations_data.parquet', edge_schema) as writer:
        for start in range(0, num_edges, 10_000_000):
            end = min(start + 10_000_000, num_edges)
            chunk_df = pd.DataFrame({
                ':START_ID(Paper)': edge_index[0, start:end].astype(np.int64),
                ':END_ID(Paper)': edge_index[1, start:end].astype(np.int64)
            })
            writer.write_table(pa.Table.from_pandas(chunk_df, schema=edge_schema))
            if end % 100_000_000 == 0: print(f"Edges: {end:,}/{num_edges:,}")

print("Ingestion files ready for Neo4j.")
