from typing import List

import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput


class PyGEquivalentSampler(BaseSampler):
    """Uniform homogeneous Neo4j sampler intended to match PyG/pyg-lib neighbor_sample.

    This implementation aims to be *distributionally equivalent* to the default
    pyg-lib sampler used by torch_geometric.loader.NeighborLoader when:
      - replace=False
      - disjoint=False
      - subgraph_type="directional" (PyG uses CSC sampling)

    Notes for undirected graphs:
      If your graph is represented as a bidirected edge list ((a,b) and (b,a)),
      PyG's CSC sampling corresponds to sampling *incoming* neighbors. In Neo4j,
      using an undirected pattern `-[r]-` would double the candidate edges and can
      waste samples. Therefore, direction='both' is treated as 'incoming'.

    Args:
      graph_store: GraphStore instance to execute sampling queries against.
      num_neighbors: Fanout sizes per hop (e.g. [10, 5]).
      sample_with_replacement: Sample with replacement.
      expand_revisited: If True, allows revisited nodes to enter next frontier.
                       For PyG equivalence, keep False.
      direction: 'incoming', 'outgoing', or 'both'. For PyG equivalence on
                 bidirected graphs, use 'incoming' (and 'both' maps to it).
      rel_type: Optional relationship type (e.g. 'REL').
      node_label: Optional node label constraint.
    """

    _instance_counter = 0

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        sample_with_replacement: bool = False,
        expand_revisited: bool = False,
        direction: str = 'incoming',
        rel_type: str = None,
        node_label: str = None,
    ):
        self.instance_id = PyGEquivalentSampler._instance_counter
        PyGEquivalentSampler._instance_counter += 1

        self.graph_store = graph_store
        self.num_neighbors = num_neighbors
        self.sample_with_replacement = sample_with_replacement
        self.expand_revisited = expand_revisited
        self.nodeid_property = graph_store.nodeid_property
        self.direction = direction
        self.rel_type = rel_type
        self.node_label = node_label

        self.query = self._build_fanout_query()

    def _build_fanout_query(self) -> str:
        replace_s = "true" if self.sample_with_replacement else "false"
        expand_revisited_s = "true" if self.expand_revisited else "false"

        # IMPORTANT for your case:
        # Your "undirected" graph is stored as bidirected edges (a->b and b->a).
        # PyG neighbor_sample (CSC) effectively samples "incoming neighbors".
        # If you sample "both" in Neo4j you change the candidate multiset and distort counts.
        eff_direction = self.direction
        if eff_direction == "both":
            eff_direction = "incoming"

        rel = "" if self.rel_type is None else f":{self.rel_type}"
        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        if eff_direction == "incoming":
            # (neighbor)-[r]->(src) ; emit neighbor -> src
            edge_pat = f"<-[r{rel}]-"
            nbr_expr = "startNode(rel)"
            edge_src_expr = "startNode(rel)"
            edge_dst_expr = "endNode(rel)"
        elif eff_direction == "outgoing":
            # (src)-[r]->(neighbor) ; emit src -> neighbor
            edge_pat = f"-[r{rel}]->"
            nbr_expr = "endNode(rel)"
            edge_src_expr = "startNode(rel)"
            edge_dst_expr = "endNode(rel)"
        else:
            # kept for completeness; should not happen since both->incoming above
            edge_pat = f"-[r{rel}]-"
            nbr_expr = "CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END"
            edge_src_expr = "src"
            edge_dst_expr = nbr_expr

        q = []

        # Preserve seed order:
        q.append(f"""
        UNWIND range(0, size($seed_ids)-1) AS i
        WITH i, $seed_ids[i] AS seed_id
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} = seed_id
        WITH i, s
        ORDER BY i
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{
            // FIRST importing WITH must be simple references ONLY:
            WITH frontier, visited, edges

            // Stable frontier order:
            UNWIND range(0, size(frontier)-1) AS i
            WITH i, frontier[i] AS src, visited, edges

            MATCH (src){edge_pat}(neighbor{nbr_label})
            WITH i, src, visited, edges, collect(r) AS cand_rels

            // pyg-lib "take all" rule:
            WITH i, src, visited, edges,
                CASE
                    WHEN {k} < 0 OR ({replace_s} = false AND {k} >= size(cand_rels))
                    THEN cand_rels
                    ELSE apoc.coll.randomItems(cand_rels, {k}, {replace_s})
                END AS picked_rels

            // Picked neighbors + edges (direction-consistent):
            WITH i, visited, edges,
                [rel IN picked_rels | {nbr_expr}] AS picked_nbrs,
                [rel IN picked_rels | {{
                    src_id: ({edge_src_expr}).{self.nodeid_property},
                    dst_id: ({edge_dst_expr}).{self.nodeid_property}
                }}] AS new_edges
            ORDER BY i

            WITH visited, edges,
                apoc.coll.flatten(collect(picked_nbrs)) AS picked_nbrs,
                apoc.coll.flatten(collect(new_edges)) AS new_edges

            // next frontier raw (filter revisits unless expand_revisited):
            WITH visited, edges, new_edges,
                CASE
                    WHEN {expand_revisited_s}
                    THEN picked_nbrs
                    ELSE [n IN picked_nbrs WHERE NOT n IN visited]
                END AS next_frontier_raw

            // Order-preserving unique (mapper-like):
            WITH visited, edges, new_edges,
                reduce(acc = [], n IN next_frontier_raw |
                    CASE WHEN n IN acc THEN acc ELSE acc + [n] END
                ) AS next_frontier

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
