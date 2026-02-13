from typing import Dict
from Neo4jConnection import Neo4jConnection
from feature_stores.v000 import Neo4jFeatureStore
from graph_stores.v000 import Neo4jGraphStore
from samplers.v000 import Neo4jSampler
from torch_geometric.loader import NodeLoader
from models.simple_gcn import GCN
import torch
import numpy as np
import cProfile
import pstats
import argparse
from pathlib import Path

def evaluate(model, graph_store, feature_store, sampler, split: str = "val") -> None:
    
    model.eval()
    with torch.no_grad():
        N: int = 2**8
        i: int = 0

        counts = []
        partial_accuracies = []

        while True:
            node_ids = graph_store.get_split(N, offset=i, split=split, shuffle=False)
            
            if node_ids.numel() == 0:
                break

            i += node_ids.numel()

            val_loader = NodeLoader(
                data=(feature_store, graph_store), 
                node_sampler=sampler,
                input_nodes=node_ids,
                batch_size=N,
                shuffle=False
            )
            for data in val_loader:
                break

            out: torch.Tensor = model(data.x, data.edge_index)
            seed_mask = torch.isin(data.n_id, data.input_id)
            targets = data.y[seed_mask]
            preds = out[seed_mask].argmax(dim=1)

            counts.append(i)
            partial_accuracies.append((targets == preds).sum().item() / targets.numel())

        cnts = np.array(counts, dtype=np.float32)
        cnts /= cnts.sum()
        print(split.capitalize(), "accuracy:", cnts  @ np.array(partial_accuracies))
        
def main():
    # Demo local user with unsecure passwd
    driver = Neo4jConnection("bolt://localhost:7687", "neo4j", "thesis-db-0-pw").get_driver()
    feature_store = Neo4jFeatureStore(driver)
    graph_store = Neo4jGraphStore(driver) # Sampler handles all topology
    sampler = Neo4jSampler(graph_store, num_neighbors=[10, 5])
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 16, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(10):
        train_indices = graph_store.get_split(256, split="train", shuffle=True)

        train_loader = NodeLoader(
            data=(feature_store, graph_store), 
            node_sampler=sampler,
            input_nodes=train_indices,
            batch_size=32,
            shuffle=False
        )

        for bi, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            seed_mask = torch.isin(batch.n_id, batch.input_id)
            loss = criterion(out[seed_mask], batch.y[seed_mask])

            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch} batch: {bi} | Loss: {loss:5f}")


    evaluate(model, graph_store, feature_store, sampler, "train")
    evaluate(model, graph_store, feature_store, sampler, "val")
    evaluate(model, graph_store, feature_store, sampler, "test")


if __name__ == "__main__":
    main()

