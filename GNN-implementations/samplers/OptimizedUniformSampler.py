from typing import List
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore

class OptimizedUniformSampler(BaseSampler):
    """Sampling method for generating sub-graphs around seed nodes of a homogeneous graph."""
    
    _instance_counter = 0
    
    def __init__(self, graph_store: GraphStore, num_neighbors: List[int], with_replacement:bool = False):
        self.instance_id = UniformSampler._instance_counter
        UniformSampler._instance_counter += 1
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors  # e.g. [10, 5] for 2 hops
        self.nodeid_property = graph_store.nodeid_property
        self.query = self._build_fanout_query(len(num_neighbors))
        self.with_replacement = "true" if with_replacement else "false"

    def _build_fanout_query(self, hops: int) -> str:
        # Homogeneous graph. Use directed or undirected:
        rel_pattern = "--"  # change to "-[:REL]->" if you want directed
        q = []
           q.append(f"""
           MATCH (s) WHERE s.{self.nodeid_property} IN $seed_ids
           WITH s,
               [s.{self.nodeid_property}] AS frontier_ids,
               [s.{self.nodeid_property}] AS visited_ids,
               [] AS edges
           """)

        for i in range(hops):
            q.append(f"""
              CALL (s, frontier_ids, visited_ids, edges) {{
                UNWIND frontier_ids AS src_id
                MATCH (src {{{self.nodeid_property}: src_id}}){rel_pattern}(nbr)
                WHERE NOT nbr.{self.nodeid_property} IN visited_ids
                WITH src_id,
                    collect(DISTINCT nbr.{self.nodeid_property}) AS cand_ids,
                    visited_ids,
                    edges
                WITH src_id,
                    apoc.coll.randomItems(cand_ids, $num_neighbors[{i}], {self.with_replacement}) AS picked_ids,
                    visited_ids,
                    edges

              // aggregate first, then concatenate (avoids implicit grouping error)
                WITH visited_ids, edges,
                    collect(picked_ids) AS dsts_list,
                    collect([d IN picked_ids | {src_id: src_id, dst_id: d}]) AS es_list

                WITH visited_ids, edges,
                    apoc.coll.toSet(apoc.coll.flatten(dsts_list)) AS next_frontier_ids,
                    apoc.coll.flatten(es_list) AS new_edges

                            RETURN
                                next_frontier_ids AS next_frontier_ids,
                                visited_ids + next_frontier_ids AS next_visited_ids,
                                edges + new_edges AS next_edges
            }}
            WITH s,
                                 next_frontier_ids AS frontier_ids,
                                 next_visited_ids AS visited_ids,
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

