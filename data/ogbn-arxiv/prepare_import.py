import pandas as pd
from ogb.nodeproppred import NodePropPredDataset
from pathlib import Path

# Load the dataset
dataset = NodePropPredDataset(name="ogbn-arxiv")
graph, labels = dataset[0]

# Prepare Node Data
# 'graph' contains node_feat, node_year, edge_index, etc.
num_nodes = graph['num_nodes']
node_ids = range(num_nodes)
years = graph['node_year'].flatten()
categories = labels.flatten()

def get_split(year):
    if year <= 2017:
        # 'train'
        return 0
    elif year == 2018:
        # 'valid'
        return 1
    else:
        # 'test'
        return 2

# Prepare features: converting 128-dim array to a string 
features = [";".join(map(str, f)) for f in graph['node_feat']]

nodes_df = pd.DataFrame({
    'nodeId:ID': node_ids,           # Unique ID for Neo4j
    'split:INT': [get_split(y) for y in years],
    'category:INT': categories,
    'features:FLOAT[]': features     # Neo4j supports array types
})


# Prepare Relationship Data
edge_index = graph['edge_index']
edges_df = pd.DataFrame({
    ':START_ID': edge_index[0],     # Citing paper
    ':END_ID': edge_index[1]        # Cited paper
})

# Export to CSV
nodes_files = str(Path(__file__).resolve().parent / 'arxiv_nodes.csv')
edges_files = str(Path(__file__).resolve().parent / 'arxiv_edges.csv')

nodes_df.to_csv(nodes_files, index=False)
edges_df.to_csv(edges_files, index=False)

print(f"Exported {num_nodes} nodes and {len(edges_df)} edges.")
