from typing import List
from torch_geometric.sampler import BaseSampler, SamplerOutput, NodeSamplerInput
import torch
from torch_geometric.data.graph_store import GraphStore


class NeighborSampler(BaseSampler):
    """Uniform homogeneous neo4j sampler (GraphSAGE-style neighbor sampling).

    Args:
    - graph_store: GraphStore instance to execute sampling queries against.
    - num_neighbors: List of fanout sizes per GNN layer (e.g. [10, 5] for 2 layers).
    - sample_with_replacement: Whether to sample neighbors with replacement (default False).
    - expand_revisited: Whether to allow already visited nodes to be included in the frontier for further expansion (default False).
    - direction: Sampling direction: 'outgoing' (src)->(nbr), 'incoming' (src)<-(nbr), or 'both' (undirected) (default 'both').
    - rel_type: Optional relationship type to restrict sampling to (default None for all types).
    - node_label: Optional node label to restrict seeds and neighbors to (default None for all labels).
    """

    _instance_counter = 0

    def __init__(
        self,
        graph_store: GraphStore,
        num_neighbors: List[int],
        sample_with_replacement: bool = False,
        expand_revisited: bool = False,
        direction: str = 'both', #'outgoing', 'incoming', 'both' (default)
        rel_type: str = None,  # optional: restrict to a relationship type
        node_label: str = None,  # optional: restrict seed/neighbor label
    ):
        self.instance_id = NeighborSampler._instance_counter
        NeighborSampler._instance_counter += 1
        self.graph_store = graph_store
        self.num_neighbors=num_neighbors
        self.sample_with_replacement=sample_with_replacement
        self.expand_revisited=expand_revisited
        self.nodeid_property=graph_store.nodeid_property
        self.direction=direction
        self.rel_type=rel_type
        self.directed=(self.direction != 'both')
        self.node_label=node_label
        self.query = self._build_fanout_query()

    def _build_fanout_query(self) -> str:
        replace_s = "true" if self.sample_with_replacement else "false"
        expand_revisited_s = "true" if self.expand_revisited else "false"

        # Relationship pattern:
        # - direction='out'  => (src)-[r]->(nbr)
        # - direction='in'   => (src)<-[r]-(nbr)
        # - direction='both' => (src)-[r]-(nbr)
        rel = "" if self.rel_type is None else f":{self.rel_type}"
        
        endpoint_selector = None
        edge_pat = None
        if self.direction == 'incoming':
            edge_pat = f"<-[r{rel}]-"
            endpoint_selector = "startNode"
        elif self.direction == 'outgoing':
            edge_pat = f"-[r{rel}]->"
            endpoint_selector = "endNode"
        else:
            edge_pat = f"-[r{rel}]-"
            endpoint_selector = "endNode"  # arbitrary for undirected

        # Optional label constraints:
        seed_label = "" if self.node_label is None else f":{self.node_label}"
        nbr_label = "" if self.node_label is None else f":{self.node_label}"

        q = []

        # IMPORTANT: batch seeds into a single row (global frontier/visited)
        q.append(f"""
        MATCH (s{seed_label})
        WHERE s.{self.nodeid_property} IN $seed_ids
        WITH collect(s) AS frontier, collect(s) AS visited, [] AS edges
        """)

        for k in self.num_neighbors:
            q.append(f"""
            CALL (frontier, visited, edges) {{
              // expand all nodes in the current frontier
              UNWIND frontier AS src
              MATCH (src){edge_pat}(neighbor{nbr_label})

              // IMPORTANT: sample EDGES (r) not DISTINCT neighbors
              WITH src, collect(r) AS cand_rels, visited, edges
              WITH src,
                   apoc.coll.randomItems(cand_rels, {k}, {replace_s}) AS picked_rels,
                   visited, edges

              // derive picked neighbors + edge list
              WITH visited, edges,
                   // neighbors corresponding to picked edges
                   collect([rel IN picked_rels | 
                     CASE WHEN {str(self.directed).lower()} THEN {endpoint_selector}(rel) 
                          ELSE CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END
                     END
                   ]) AS picked_nbrs_list,
                   collect([rel IN picked_rels | {{
                     src_id: src.{self.nodeid_property},
                     dst_id: (
                       CASE WHEN {str(self.directed).lower()} THEN {endpoint_selector}(rel)
                            ELSE CASE WHEN startNode(rel) = src THEN endNode(rel) ELSE startNode(rel) END
                       END
                     ).{self.nodeid_property}
                   }}]) AS es_list

              WITH visited, edges,
                   apoc.coll.flatten(picked_nbrs_list) AS picked_nbrs,
                   apoc.coll.flatten(es_list) AS new_edges

              // next frontier:
              // - if expand_revisited: allow already visited nodes into frontier
              // - else: only truly new nodes become frontier
              WITH visited, edges, new_edges,
                   CASE
                     WHEN {expand_revisited_s}
                       THEN picked_nbrs
                     ELSE [n IN picked_nbrs WHERE NOT n IN visited]
                   END AS next_frontier_raw

              WITH visited, edges, new_edges,
                   apoc.coll.toSet(next_frontier_raw) AS next_frontier

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
        UNWIND edges AS e
        // no DISTINCT: closer to PyG semantics (edge multiplicity preserved)
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