from typing import List
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore


class NewUniformSampler(BaseSampler):
    """Uniform homogeneous neo4j sampler (GraphSAGE-style neighbor sampling).

    Difference vs. PyG NeighborSampler:
    - Can revisit nodes during sampling (if revisit_nodes=True), and expand them again (if expand_revisited=True).
    """

    _instance_counter = 0

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        sample_with_replacement: bool = False,
        revisit_nodes: bool = True, # this should be true for it to work like the pyg-lib sampler
        expand_revisited: bool = False, # only matters is revisit_nodes is set to true
    ):
        """Create a sampler.

        Args:
            graph_store: Neo4j-backed GraphStore instance.
            num_neighbors: Fanout per hop, e.g. [10, 5, 5].
            revisit_nodes: If True, allow already visited nodes to be sampled again.
            sample_with_replacement: If True, sample neighbors with replacement.
            expand_revisited: If True, allow revisited nodes to be expanded again.
        """
        self.instance_id = NewUniformSampler._instance_counter
        NewUniformSampler._instance_counter += 1
        self.graph_store = graph_store
        self.query = self._build_fanout_query(
            num_neighbors,
            revisit_nodes,
            sample_with_replacement,
            expand_revisited,
            graph_store.nodeid_property,
        )

    def _build_fanout_query(
        self,
        num_neighbors: List[int],
        revisit_nodes: bool,
        sample_with_replacement: bool,
        expand_revisited: bool,
        nodeid_property,
    ) -> str:
        # Homogeneous graph. Use directed or undirected:
        revisit_nodes = "true" if revisit_nodes else "false"
        sample_with_replacement = 'true' if sample_with_replacement else 'false'
        expand_revisited = "true" if expand_revisited else "false"
        q = []
        q.append(f"""
        MATCH (s) WHERE s.{nodeid_property} IN $seed_ids
        WITH s, [s] AS frontier, [s] AS visited, [] AS edges
        """)

        for i in num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{
              UNWIND frontier AS src
              MATCH (src)--(neighbor)
              WHERE ({revisit_nodes}) OR NOT neighbor IN visited
              WITH src, collect(DISTINCT neighbor) AS cand, visited, edges
              WITH src, apoc.coll.randomItems(cand, {i}, {sample_with_replacement}) AS picked, visited, edges

              // aggregate first, then concatenate (avoids implicit grouping error)
              WITH visited, edges,
                   CASE
                     WHEN {expand_revisited} THEN collect(picked)
                     ELSE collect([n IN picked WHERE NOT n IN visited])
                   END AS dsts_list,
                   collect([d IN picked | {{src_id: src.{nodeid_property}, dst_id: d.{nodeid_property}}}]) AS es_list

              WITH visited, edges,
                   apoc.coll.toSet(apoc.coll.flatten(dsts_list)) AS next_frontier,
                   apoc.coll.flatten(es_list) AS new_edges

              RETURN
                  next_frontier,
                  visited + next_frontier AS next_visited,
                  edges + new_edges AS next_edges
            }}
            WITH next_frontier AS frontier,
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
                kwargs={"seed_ids": seeds_list},
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


# class NewUniformSampler(BaseSampler):
#     """Sampling method that tracks visited nodes to avoid revisits across hops."""

#     _instance_counter = 0

#     def __init__(self, graph_store: GraphStore, num_neighbors: List[int], revisit_nodes: bool = False, sample_with_replacement:bool = False):
#         self.instance_id = NewUniformSampler._instance_counter
#         NewUniformSampler._instance_counter += 1
#         self.graph_store = graph_store
#         self.num_neighbors = num_neighbors
#         self.nodeid_property = graph_store.nodeid_property
#         self.query = self._build_fanout_query(revisit_nodes, sample_with_replacement)

#     def _build_fanout_query(self, revisit_nodes: bool, sample_with_replacement:bool) -> str:
#         # Homogeneous graph. Use directed or undirected:
#         rel_pattern = "--"  # change to "-[:REL]->" if you want directed
#         num_neighbors_literal = "[" + ", ".join(str(n) for n in self.num_neighbors) + "]"
#         revisit_nodes = "true" if revisit_nodes else "false"
#         sample_with_replacement = 'true' if sample_with_replacement else 'false'

#         return f"""
#         WITH {num_neighbors_literal} AS num_neighbors
#         MATCH (s) WHERE s.{self.nodeid_property} IN $seed_ids
#         WITH collect(s) AS startNodes,
#              num_neighbors,
#              [] AS edges,
#              [] AS nextFrontier,
#              [n IN collect(s) | n.{self.nodeid_property}] AS visitedIds
#         UNWIND range(0, size(num_neighbors) - 1) AS hop
#         WITH hop,
#             num_neighbors[hop] AS k,
#             CASE hop
#                 WHEN 0 THEN startNodes
#                 ELSE nextFrontier
#             END AS frontier,
#             num_neighbors,
#             edges,
#             nextFrontier,
#             visitedIds
#         CALL (frontier, k, visitedIds) {{
#             WITH frontier, k, visitedIds
#             UNWIND frontier AS ni
#             MATCH (ni){rel_pattern}(nj)
#             WHERE (NOT {revisit_nodes}) OR NOT nj.{self.nodeid_property} IN visitedIds
#             WITH ni, collect(DISTINCT nj) AS cand, k
#             WITH ni, apoc.coll.randomItems(cand, k, {sample_with_replacement}) AS sampled_nodes
#             WITH ni, sampled_nodes,
#                  [x IN sampled_nodes | x.{self.nodeid_property}] AS sampledIds,
#                  [x IN sampled_nodes | {{src: ni.{self.nodeid_property}, dst: x.{self.nodeid_property}}}] AS sampledEdges
#             RETURN
#                 apoc.coll.flatten(collect(sampledEdges)) AS newEdges,
#                 apoc.coll.flatten(collect(sampled_nodes)) AS newNodes,
#                 apoc.coll.toSet(apoc.coll.flatten(collect(sampledIds))) AS newNodeIds
#         }}
#         WITH
#             num_neighbors,
#             hop,
#             coalesce(edges, []) + newEdges AS edges,
#             newNodes AS nextFrontier,
#             apoc.coll.toSet(visitedIds + coalesce(newNodeIds, [])) AS visitedIds

#         UNWIND edges AS e
#         RETURN DISTINCT e.src AS src, e.dst AS dst
#         """

#     def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
#         seeds = ns_input.node.to(torch.int64)
#         seeds_list = seeds.tolist()
#         seed_time = getattr(ns_input, "time", None)

#         if hasattr(self.graph_store, "sample_from_nodes"):
#             unique_nodes, edge_index_local = self.graph_store.sample_from_nodes(
#                 kwargs={"seed_ids": seeds_list},
#                 query=self.query,
#             )
#             return SamplerOutput(
#                 node=unique_nodes,
#                 row=edge_index_local[0],
#                 col=edge_index_local[1],
#                 edge=None,
#                 batch=None,
#                 metadata=(seeds, seed_time),
#             )
#         raise NotImplementedError("GraphStore lacks 'sample_from_nodes'.")

#     def sample_from_edges(self, index, neg_sampling=None):
#         raise NotImplementedError("Not implemented")
