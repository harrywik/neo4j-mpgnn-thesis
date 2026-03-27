from typing import List

import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput


class Neo4jSampler(BaseSampler):
    """Neo4j neighbor sampler that is structurally equivalent to pyg-lib.
    This sampler is a multi-hop incoming-edge sampler that samples the neighbors of the seed nodes.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        expand_revisited: bool = False,
        edge_direction: str = "incoming",  # 'outgoing', 'incoming', 'both'
        rel_type: str = "CITES",
        node_label: str = "Paper",
        profile: bool = False,
    ):
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors
        self.expand_revisited = expand_revisited
        self.edge_direction = edge_direction
        self.nodeid_property = graph_store.nodeid_property
        self.rel_type = rel_type
        self.node_label = node_label
        self.profile = profile

        self.query = self._build_fanout_query()

    def _build_fanout_query(self) -> str:
        """Build a Cypher query that performs multi-hop incoming-edge sampling.

        The query returns a single row with:

        * ``ordered_nodes`` — list of ``nodeid_property`` values in encounter
          order (seeds first, then hop-1 new nodes, …).
        * ``edge_pairs`` — list of ``[src_id, dst_id]`` (global IDs) for
          every sampled edge.

        Sampling strategy
        -----------------
        * ``replace=False``, ``disjoint=False`` — matches pyg-lib defaults.
        * **Take-all rule** — when ``k < 0`` or ``k >= |neighbourhood|``,
          all edges are taken (same as pyg-lib C++ ``_sample`` Case 1).
        * **Order-preserving frontier deduplication** — uses a Cypher
          ``reduce`` to append newly-seen nodes in first-encounter order,
          mirroring pyg-lib's ``Mapper`` insertion semantics.
        * **All edges recorded** — edges to already-visited nodes are included
          in ``edge_pairs``, consistent with pyg-lib's ``add()`` function.
        """
        expand_revisited_s = "true" if self.expand_revisited else "false"

        rel = "" if self.rel_type is None else f":{self.rel_type}"
        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        # Incoming edge pattern: (src)<-[r]-(neighbor)
        # startNode(r) = neighbor, endNode(r) = src
        edge_pat = None
        nbr_expr = None
        edge_src_expr = None
        edge_dst_expr = None
        if self.edge_direction == "outgoing":
            edge_pat = f"-[rel{rel}]->"
            nbr_expr = "endNode(rel)"
            edge_src_expr = f"startNode(rel).{self.nodeid_property}"
            edge_dst_expr = f"endNode(rel).{self.nodeid_property}"
        elif self.edge_direction == "both":
            edge_pat = f"-[rel{rel}]-"
            nbr_expr = f"CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END"
            edge_src_expr = f"src.{self.nodeid_property}"
            edge_dst_expr = f"CASE WHEN startNode(rel) = src THEN endNode(rel).{self.nodeid_property} ELSE startNode(rel).{self.nodeid_property} END"
        else:
            edge_pat = f"<-[rel{rel}]-"
            nbr_expr = "startNode(rel)"
            edge_src_expr = f"startNode(rel).{self.nodeid_property}"
            edge_dst_expr = f"endNode(rel).{self.nodeid_property}"

        q = []

        profile_prefix = "PROFILE\n        " if self.profile else ""

        # Initialise with seeds in supplied order.
        q.append(f"""
        // 1. initialise the frontier, visited and edges
        {profile_prefix}UNWIND range(0, size($seed_ids)-1) AS i
        WITH i, $seed_ids[i] AS seed_id
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} = seed_id
        WITH i, s
        //ORDER BY i
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{

            // 2. process frontier nodes in stable index order.
            UNWIND range(0, size(frontier)-1) AS i
            WITH i, frontier[i] AS src, visited, edges

            // 3. match the neighbors
            MATCH (src){edge_pat}(neighbor{nbr_label})
            WITH i, src, visited, edges, collect(rel) AS cand_rels

            // 4. pyg-lib "take all" rule (Case 1 in _sample).
            WITH i, src, visited, edges,
                CASE
                    WHEN {k} < 0 OR (false = false AND {k} >= size(cand_rels))
                    THEN cand_rels
                    ELSE apoc.coll.randomItems(cand_rels, {k}, false)
                END AS picked_rels

            // 5. build the neighbour list and edge list for this src.
            WITH i, visited, edges,
                [rel IN picked_rels | {nbr_expr}] AS picked_nbrs,
                [rel IN picked_rels | [{edge_src_expr}, {edge_dst_expr}]] AS new_edges
            ORDER BY i

            // 6. aggregate across all src nodes — back to a single row.
            WITH visited, edges,
                apoc.coll.flatten(collect(picked_nbrs)) AS picked_nbrs,
                apoc.coll.flatten(collect(new_edges)) AS new_edges

            // 7. filter revisited + deduplicate next frontier
            WITH visited, edges, new_edges,
                apoc.coll.toSet(
                    CASE
                        WHEN {expand_revisited_s}
                        THEN picked_nbrs
                        ELSE [n IN picked_nbrs WHERE NOT n IN visited]
                    END
                ) AS next_frontier

            // 9. return the next frontier, visited and edges
            RETURN
                next_frontier,
                visited + next_frontier AS next_visited,
                edges + new_edges AS next_edges
            }}
            WITH next_frontier AS frontier,
                next_visited AS visited,
                next_edges AS edges
            """)

        q.append(f"""
        // 11. return the ordered nodes and edge pairs
        RETURN
            [n IN visited | n.{self.nodeid_property}] AS ordered_nodes,
            edges AS edge_pairs
        """)

        return "\n".join(q)

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seed_time = getattr(ns_input, "time", None)

        # DB call — timing lives in the graph store.
        record = self.graph_store.fetch_ordered_subgraph(
            self.query, {"seed_ids": seeds.tolist()}
        )

        # ETL (tensor building) delegated to graph store so all ETL
        # instrumentation lives in one place.
        node, row, col = self.graph_store.build_topo_etl(record, seeds)

        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=None,
            batch=None,
            metadata=(seeds, seed_time),
        )

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Not implemented")
