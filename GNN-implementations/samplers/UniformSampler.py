from typing import List
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore

class UniformSampler(BaseSampler):
    """Sampling method for generating sub-graphs around seed nodes of a homogeneous graph."""
    
    _instance_counter = 0
    
    def __init__(self, graph_store: GraphStore, num_neighbors: List[int], ):
        self.instance_id = UniformSampler._instance_counter
        UniformSampler._instance_counter += 1
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors  # e.g. [10, 5] for 2 hops
        self.query = self._build_fanout_query(len(num_neighbors))

    def _build_fanout_query(self, hops: int) -> str:
        # Homogeneous graph. Use directed or undirected:
        rel_pattern = "--"  # change to "-[:REL]->" if you want directed

        q = []
        q.append("""
        MATCH (s) WHERE s.id IN $seed_ids
        WITH s, [s] AS frontier, [s] AS visited, [] AS edges
        """)

        for i in range(hops):
            q.append(f"""
            CALL (s, frontier, visited, edges) {{
              UNWIND frontier AS src
              MATCH (src){rel_pattern}(nbr)
              WHERE NOT nbr IN visited
              WITH src, collect(DISTINCT nbr) AS cand, visited, edges
              WITH src, apoc.coll.randomItems(cand, $num_neighbors[{i}], false) AS picked, visited, edges

              // aggregate first, then concatenate (avoids implicit grouping error)
              WITH visited, edges,
                   collect(picked) AS dsts_list,
                   collect([d IN picked | {{src_id: src.id, dst_id: d.id}}]) AS es_list

              WITH visited, edges,
                   apoc.coll.toSet(apoc.coll.flatten(dsts_list)) AS next_frontier,
                   apoc.coll.flatten(es_list) AS new_edges

              RETURN
                next_frontier AS next_frontier,
                visited + next_frontier AS next_visited,
                edges + new_edges AS next_edges
            }}
            WITH s,
                 next_frontier AS frontier,
                 next_visited AS visited,
                 next_edges AS edges
            """)

        q.append("""
        UNWIND edges AS e
        RETURN DISTINCT e.src_id AS src, e.dst_id AS dst
        """)
        return "\n".join(q)

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seeds_list = seeds.tolist()
        seed_time = getattr(ns_input, "time", None)

        if hasattr(self.graph_store, "sample_from_nodes"):
            unique_nodes, edge_index_local = self.graph_store.sample_from_nodes(
                kwargs={"seed_ids": seeds_list, "num_neighbors": self.num_neighbors},
                query=self.query,
            )
            return SamplerOutput(
                node=unique_nodes,
                row=edge_index_local[0],
                col=edge_index_local[1],
                edge=None,
                batch=None,
                metadata=(seeds, seed_time),
            )
        raise NotImplementedError("GraphStore lacks 'sample_from_nodes'.")

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Not implemented")

