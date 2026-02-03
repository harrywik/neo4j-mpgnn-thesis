from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch


class Neo4jSampler(BaseSampler):
    def __init__(self, driver, num_neighbors: list):
        self.driver = driver
        self.num_neighbors = num_neighbors # e.g., [10, 5] for 2 hops

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seeds_list = seeds.tolist()
        seed_time = getattr(ns_input, "time", None)
        # For a 2-hop sampler, num_neighbors would be [n, m]
        total_hops = len(self.num_neighbors)

        # We use APOC to expand the paths and return the edges
        # Assumption:
        # .id is a property that is unique on every node
        query = """
        MATCH (n) WHERE n.id IN $seed_ids
        CALL apoc.path.expandConfig(n, {
            relationshipFilter: "<>",
            minLevel: 1,
            maxLevel: $hops,
            uniqueness: "RELATIONSHIP_PATH"
        }) YIELD path
        WITH nodes(path) AS ns
        UNWIND range(0, size(ns)-2) AS i
        RETURN ns[i].id AS src, ns[i+1].id AS dst
        """

        with self.driver.session() as session:
            result = session.run(query, seed_ids=seeds_list, hops=total_hops)
            
            # Extract edges and format for PyG
            edges = [[r["src"], r["dst"]] for r in result]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
        # Get unique nodes involved in this sampled subgraph
        nodes = torch.unique(edge_index)

        return SamplerOutput(
            node=nodes,
            row=edge_index[0],
            col=edge_index[1],
            edge=None,
            batch=None, 
            metadata=(seeds, seed_time)
        )
    
    def sample_from_edges(self, index, neg_sampling = None):
        pass