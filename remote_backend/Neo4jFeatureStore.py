import torch
from torch_geometric.typing import FeatureTensorType
from torch_geometric.data.feature_store import FeatureStore, TensorAttr
from neo4j import Driver

class Neo4jFeatureStore(FeatureStore):
    def __init__(self, driver: Driver) -> None:
        super().__init__()
        self.driver = driver

    def _get_tensor(self, attr: TensorAttr) -> FeatureTensorType:
        node_ids: list = attr.index.tolist()

        query = """
        MATCH (n)
        WHERE n.id IN $node_ids
        RETURN n.id AS id, n.features AS feats
        ORDER BY n.id ASC
        """

        with self.driver.session() as session:
            res = session.run(query, node_ids=node_ids)
            order_mapper = {record["id"]: record["feats"] for record in res}

        features = [order_mapper[node_id] for node_id in node_ids]
        return torch.tensor(features, dtype=torch.float32)



