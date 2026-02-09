import torch
from typing import Dict, List 
from torch_geometric.sampler import BaseSampler, HeteroSamplerOutput
from torch_geometric.typing import NodeType, EdgeType

class Neo4jHeteroSampler(BaseSampler):
    def __init__(self, graph_store, feature_store, num_neighbors: List[int]):
        super().__init__()
        self.graph_store = graph_store
        self.feature_store = feature_store
        self.num_neighbors = num_neighbors # e.g., [10, 5] for 2 hops

    def sample_from_nodes(self, index: torch.Tensor, **kwargs) -> HeteroSamplerOutput:
        seed_ids = index.tolist()
        
        # We use apoc.path.subgraphAll to find all nodes/edges within total_hops
        query = """
        MATCH (start) WHERE start.id IN $seed_ids
        CALL apoc.path.subgraphAll(start, {
            maxLevel: $hops,
            relationshipFilter: $rel_filter
        }) YIELD nodes, relationships
        RETURN nodes, relationships
        """
        
        # Construct rel_filter from the schema (e.g., "WRITES>|PUBLISHED_IN>")
        # For DBLP, we want to specify which relations to traverse
        rel_filter = "" 

        with self.graph_store.driver.session(database=self.graph_store.database) as session:
            result = session.run(query, seed_ids=seed_ids, hops=len(self.num_neighbors), rel_filter=rel_filter)
            record = result.single()
            
        # Process Neo4j Result into Dicts
        nodes_dict: Dict[NodeType, torch.Tensor] = {}
        edges_dict: Dict[EdgeType, torch.Tensor] = {}
        
        # Map to track global_id -> local_index per node type
        id_mappings: Dict[NodeType, Dict[int, int]] = {}

        # Populate nodes_dict and mappings
        for node in record["nodes"]:
            # Handle multiple labels by picking the one in our schema
            node_type = [l for l in node.labels if l in self.graph_store.meta[0]][0]
            global_id = node["id"]
            
            if node_type not in nodes_dict:
                nodes_dict[node_type] = []
                id_mappings[node_type] = {}
            
            id_mappings[node_type][global_id] = len(nodes_dict[node_type])
            nodes_dict[node_type].append(global_id)

        # Populate edges_dict (re-indexing to local)
        for rel in record["relationships"]:
            src_node = rel.start_node
            dst_node = rel.end_node
            
            src_type = [l for l in src_node.labels if l in self.graph_store.meta[0]][0]
            dst_type = [l for l in dst_node.labels if l in self.graph_store.meta[0]][0]
            rel_type = rel.type
            triplet = (src_type, rel_type, dst_type)

            if triplet not in edges_dict:
                edges_dict[triplet] = [[], []]
            
            edges_dict[triplet][0].append(id_mappings[src_type][src_node["id"]])
            edges_dict[triplet][1].append(id_mappings[dst_type][dst_node["id"]])

        # Final conversion to Tensors
        final_nodes = {k: torch.tensor(v, dtype=torch.long) for k, v in nodes_dict.items()}
        final_edges = {k: torch.tensor(v, dtype=torch.long) for k, v in edges_dict.items()}

        return HeteroSamplerOutput(
            node=final_nodes,
            row=final_edges, # PyG expects 'row' and 'col' for CSR/COO
            col=None, # In simple COO, we put the 2xN tensor in 'row' or separate
            edge=None # Edge IDs if you have them
        )
