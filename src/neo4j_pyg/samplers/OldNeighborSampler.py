from typing import List
from torch_geometric.sampler import BaseSampler, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore
from neo4j_pyg.samplers._utils import build_sampler_output


class OldNeighborSampler(BaseSampler):
    """Old uniform homogeneous neo4j sampler (GraphSAGE-style neighbor sampling)."""

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        sample_with_replacement: bool = False,
        expand_revisited: bool = False,
        direction: str = "incoming",  # 'outgoing', 'incoming', 'both'
        rel_type: str = None,
        node_label: str = None,
    ):
        if direction not in ("incoming", "outgoing", "both"):
            raise ValueError(
                f"direction must be 'incoming', 'outgoing', or 'both', got {direction!r}"
            )
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors
        self.sample_with_replacement = sample_with_replacement
        self.expand_revisited = expand_revisited
        self.nodeid_property = graph_store.nodeid_property
        self.direction = direction
        self.rel_type = rel_type
        self.directed = (self.direction != "both")
        self.node_label = node_label

        self.query = self._build_fanout_query()

    def _build_fanout_query(self) -> str:
        replace_s = "true" if self.sample_with_replacement else "false"
        expand_revisited_s = "true" if self.expand_revisited else "false"

        rel = "" if self.rel_type is None else f":{self.rel_type}"

        if self.direction == "incoming":
            edge_pat = f"<-[r{rel}]-"
            # list-comprehension iterates with variable `rel`
            nbr_list_expr = "startNode(rel)"
            edge_src_in_list = f"startNode(rel).{self.nodeid_property}"
            edge_dst_in_list = f"endNode(rel).{self.nodeid_property}"
        elif self.direction == "outgoing":
            edge_pat = f"-[r{rel}]->"
            nbr_list_expr = "endNode(rel)"
            edge_src_in_list = f"startNode(rel).{self.nodeid_property}"
            edge_dst_in_list = f"endNode(rel).{self.nodeid_property}"
        else:
            edge_pat = f"-[r{rel}]-"
            nbr_list_expr = f"CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END"
            edge_src_in_list = f"src.{self.nodeid_property}"
            edge_dst_in_list = f"CASE WHEN startNode(rel) = src THEN endNode(rel).{self.nodeid_property} ELSE startNode(rel).{self.nodeid_property} END"

        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        # For undirected ("both") MATCH on bidirected graphs (A→B and B→A stored as
        # separate relationships), each unique neighbor appears twice in the result.
        # Group by neighbor first and keep one canonical relationship so that
        # cand_rels contains exactly one entry per unique neighbor — matching
        # the CSC-column semantics used by pyg-lib.
        if self.direction == "both":
            cand_rels_clause = (
                f"WITH src, visited, edges, neighbor, head(collect(r)) AS canonical_r\n"
                f"              WITH src, visited, edges, collect(canonical_r) AS cand_rels"
            )
        else:
            cand_rels_clause = "WITH src, visited, edges, collect(r) AS cand_rels"

        q = []

        q.append(f"""
        // 1. initialise the frontier, visited and edges
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} IN $seed_ids
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{
              // 2. unwind all the frontier nodes
              UNWIND frontier AS src
              MATCH (src){edge_pat}(neighbor{nbr_label})
              {cand_rels_clause}

              // 3. sample the neighbors
              WITH src, visited, edges,
                   CASE
                     WHEN {k} < 0 OR ({replace_s} = false AND {k} >= size(cand_rels))
                     THEN cand_rels
                     ELSE apoc.coll.randomItems(cand_rels, {k}, {replace_s})
                   END AS picked_rels

              // 4.build the neighbor list and edge list
              WITH src, visited, edges,
                   [rel IN picked_rels | {nbr_list_expr}] AS picked_nbrs,
                   [rel IN picked_rels | {{
                     src_id: {edge_src_in_list},
                     dst_id: {edge_dst_in_list}
                   }}] AS new_edges

              // 5. aggregate the neighbor list and edge list
              WITH visited, edges,
                   apoc.coll.flatten(collect(picked_nbrs)) AS picked_nbrs,
                   apoc.coll.flatten(collect(new_edges)) AS new_edges

              // 6. filter the revisited nodes
              WITH visited, edges, new_edges,
                   CASE
                     WHEN {expand_revisited_s}
                       THEN picked_nbrs
                     ELSE [n IN picked_nbrs WHERE NOT n IN visited]
                   END AS next_frontier_raw

              // 7. order the frontier nodes
              WITH visited, edges, new_edges,
                   apoc.coll.toSet(next_frontier_raw) AS next_frontier

              // 8. return the next frontier, visited and edges
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
        RETURN e.src_id AS src, e.dst_id AS dst
        """)

        return "\n".join(q)

    def sample_from_nodes(self, ns_input: NodeSamplerInput):
        seeds = ns_input.node.to(torch.int64)
        seed_time = getattr(ns_input, "time", None)
        if not hasattr(self.graph_store, "sample_from_nodes"):
            raise NotImplementedError("GraphStore lacks 'sample_from_nodes'.")
        return build_sampler_output(self.graph_store, self.query, seeds, seed_time)

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Not implemented")