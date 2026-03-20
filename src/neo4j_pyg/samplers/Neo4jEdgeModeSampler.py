"""Neo4j neighbor sampler with configurable edge semantics.

Same structure as :class:`~neo4j_pyg.samplers.Neo4jSampler.Neo4jSampler`, but
``edge_mode`` selects how neighbors are expanded and which edges are returned:

* ``incoming`` — multi-hop over incoming edges (same as ``Neo4jSampler``).
* ``outgoing`` — multi-hop over outgoing edges.
* ``undirected`` — neighbors are any node linked by an edge in either direction;
  recorded edges go from the frontier node to the neighbor (center → neighbor).
* ``induced`` — multi-hop neighbor sampling uses the same incoming expansion as
  ``Neo4jSampler`` to build ``visited``; **edge_pairs** are then replaced by
  **every relationship** among those nodes (full induced subgraph, both
  orientations per relationship).
"""

from __future__ import annotations

import time
from typing import List, Literal

import torch
from torch_geometric.data.graph_store import GraphStore
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput

EdgeMode = Literal["incoming", "outgoing", "undirected", "induced"]
InducedExpansion = Literal["incoming", "outgoing", "undirected"]


class Neo4jEdgeModeSampler(BaseSampler):
    """Neo4j neighbor sampler with configurable edge direction / induced edges."""

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        edge_mode: EdgeMode = "incoming",
        expand_revisited: bool = False,
        rel_type: str | None = None,
        node_label: str | None = None,
        profile: bool = False,
    ):
        self.graph_store = graph_store
        self.num_neighbors = num_neighbors
        self.edge_mode = edge_mode
        self.expand_revisited = expand_revisited
        self.nodeid_property = graph_store.nodeid_property
        self.rel_type = rel_type
        self.node_label = node_label
        self.profile = profile

        self.query = self._build_fanout_query()

    def _expansion_mode_for_fanout(self) -> InducedExpansion:
        """Which topology to use for multi-hop frontier expansion.

        For ``induced``, the walk matches ``Neo4jSampler`` (incoming); the
        returned edges are still the full induced subgraph among visited nodes.
        """
        if self.edge_mode == "induced":
            return "incoming"
        if self.edge_mode == "incoming":
            return "incoming"
        if self.edge_mode == "outgoing":
            return "outgoing"
        return "undirected"

    def _edge_fragments(self, expansion: InducedExpansion) -> tuple[str, str, str, str]:
        """Return (edge_pat, nbr_list_comprehension, edge_src_expr, edge_dst_expr).

        ``nbr_list_comprehension`` is the expression inside
        ``[rel IN picked_rels | ...]`` for the neighbor node(s).
        ``edge_src_expr`` / ``edge_dst_expr`` are expressions for ``[src, dst]``
        inside ``[rel IN picked_rels | [...]]`` (may reference ``src``).
        """
        rel = "" if self.rel_type is None else f":{self.rel_type}"
        pid = self.nodeid_property

        if expansion == "incoming":
            # (src)<-[r]-(neighbor): message along incoming edges to src
            edge_pat = f"<-[r{rel}]-"
            nbr_expr = f"startNode(rel)"
            edge_src = f"startNode(rel).{pid}"
            edge_dst = f"endNode(rel).{pid}"
            return edge_pat, nbr_expr, edge_src, edge_dst

        if expansion == "outgoing":
            edge_pat = f"-[r{rel}]->"
            nbr_expr = f"endNode(rel)"
            edge_src = f"startNode(rel).{pid}"
            edge_dst = f"endNode(rel).{pid}"
            return edge_pat, nbr_expr, edge_src, edge_dst

        # undirected: (src)-[r]-(other); neighbor is the endpoint that is not src
        edge_pat = f"-[r{rel}]-"
        nbr_expr = (
            f"CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END"
        )
        edge_src = f"src.{pid}"
        edge_dst = (
            f"CASE WHEN startNode(rel) = src THEN endNode(rel).{pid} "
            f"ELSE startNode(rel).{pid} END"
        )
        return edge_pat, nbr_expr, edge_src, edge_dst

    def _build_fanout_query(self) -> str:
        expand_revisited_s = "true" if self.expand_revisited else "false"

        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        expansion = self._expansion_mode_for_fanout()
        edge_pat, nbr_expr, edge_src_expr, edge_dst_expr = self._edge_fragments(
            expansion
        )

        # Undirected: exclude self-loops; relationship variable in patterns is always `r`.
        undirected_extra = ""
        if expansion == "undirected":
            undirected_extra = "\n            WHERE neighbor <> src"

        q: list[str] = []

        profile_prefix = "PROFILE\n        " if self.profile else ""

        q.append(f"""
        // 1. initialise the frontier, visited and edges
        {profile_prefix}UNWIND range(0, size($seed_ids)-1) AS i
        WITH i, $seed_ids[i] AS seed_id
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} = seed_id
        WITH i, s
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{

            // 2. process frontier nodes in stable index order.
            UNWIND range(0, size(frontier)-1) AS i
            WITH i, frontier[i] AS src, visited, edges

            // 3. match the neighbors
            MATCH (src){edge_pat}(neighbor{nbr_label}){undirected_extra}
            WITH i, src, visited, edges, collect(r) AS cand_rels

            // 4. pyg-lib "take all" rule (Case 1 in _sample).
            WITH i, src, visited, edges,
                CASE
                    WHEN {k} < 0 OR (false = false AND {k} >= size(cand_rels))
                    THEN cand_rels
                    ELSE apoc.coll.randomItems(cand_rels, {k}, false)
                END AS picked_rels

            // 5. build the neighbour list and edge list for this src.
            // (keep `src` — undirected mode references it in edge expressions)
            WITH i, src, visited, edges,
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

        if self.edge_mode == "induced":
            rel = "" if self.rel_type is None else f":{self.rel_type}"
            pid = self.nodeid_property
            q.append(f"""
        // induced: replace walk edges with all edges between visited nodes
        WITH visited
        MATCH (a)-[r{rel}]-(b)
        WHERE a IN visited AND b IN visited AND a.{pid} < b.{pid}
        WITH visited, collect([a.{pid}, b.{pid}]) AS half
        WITH visited,
            apoc.coll.flatten(
                [pair IN half | [pair, [pair[1], pair[0]]]]
            ) AS edge_pairs_flat
        WITH visited,
            [pair IN apoc.coll.toSet(edge_pairs_flat) WHERE pair IS NOT NULL] AS edge_pairs
        RETURN
            [n IN visited | n.{pid}] AS ordered_nodes,
            edge_pairs
        """)
        else:
            q.append(f"""
        // return the ordered nodes and edge pairs (walk edges)
        RETURN
            [n IN visited | n.{self.nodeid_property}] AS ordered_nodes,
            edges AS edge_pairs
        """)

        return "\n".join(q)

    def sample_from_nodes(self, ns_input: NodeSamplerInput) -> SamplerOutput:
        seeds = ns_input.node.to(torch.int64)
        seed_time = getattr(ns_input, "time", None)
        measurer = getattr(self.graph_store, "measurer", None)

        record = self.graph_store.fetch_ordered_subgraph(
            self.query, {"seed_ids": seeds.tolist()}
        )

        t_etl_start = time.monotonic()

        if record is None or not record["ordered_nodes"]:
            if measurer is not None:
                measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)
            return SamplerOutput(
                node=seeds,
                row=torch.zeros(0, dtype=torch.long),
                col=torch.zeros(0, dtype=torch.long),
                edge=None,
                batch=None,
                metadata=(seeds, seed_time),
            )

        ordered_global_ids = torch.tensor(record["ordered_nodes"], dtype=torch.long)
        global_to_local = {
            int(gid): i for i, gid in enumerate(ordered_global_ids.tolist())
        }

        edge_pairs = record["edge_pairs"]
        if edge_pairs:
            row = torch.tensor(
                [global_to_local[e[0]] for e in edge_pairs], dtype=torch.long
            )
            col = torch.tensor(
                [global_to_local[e[1]] for e in edge_pairs], dtype=torch.long
            )
        else:
            row = col = torch.zeros(0, dtype=torch.long)

        if measurer is not None:
            measurer.log_event("topo_etl_ms", (time.monotonic() - t_etl_start) * 1000)

        return SamplerOutput(
            node=ordered_global_ids,
            row=row,
            col=col,
            edge=None,
            batch=None,
            metadata=(seeds, seed_time),
        )

    def sample_from_edges(self, index, neg_sampling=None):
        raise NotImplementedError("Not implemented")
