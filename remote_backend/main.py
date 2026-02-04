from Neo4jConnection import Neo4jConnection
from Neo4jFeatureStore import Neo4jFeatureStore
from Neo4jGraphStore import Neo4jGraphStore
from Neo4jSampler import Neo4jSampler
from torch_geometric.loader import NodeLoader
from Model import GCN
import torch

if __name__ == "__main__":
    # Demo local user with unsecure passwd
    driver = Neo4jConnection("bolt://localhost:7687", "neo4j", "thesis-db-0-pw").get_driver()

    feature_store = Neo4jFeatureStore(driver)
    graph_store = Neo4jGraphStore(driver) # Sampler handles all topology
    sampler = Neo4jSampler(driver, num_neighbors=[10, 5])
    graph_store.train_val_test_split_db([0.6, 0.2, 0.2])
    model = GCN(1433, 32, 16, 7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for pe in range(10):
        train_indices = graph_store.get_random_split(1000)

        train_loader = NodeLoader(
            data=(feature_store, graph_store), 
            node_sampler=sampler,
            input_nodes=train_indices,
            batch_size=32,
            shuffle=False
        )

        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()
            print(batch)
            out = model(batch.x, batch.edge_index)
            seed_mask = torch.isin(batch.n_id, batch.input_id)
            loss = criterion(out[seed_mask], batch.y[seed_mask])

            loss.backward()
            optimizer.step()
            print(f"Loss: {loss:5f}")
