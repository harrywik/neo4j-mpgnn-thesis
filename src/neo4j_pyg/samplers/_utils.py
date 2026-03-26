import torch
from torch_geometric.sampler import SamplerOutput


def remap_with_seeds(
    unique_nodes: torch.Tensor,
    edge_index_local: torch.Tensor,
    seeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure every seed node appears in ``unique_nodes`` and that
    ``edge_index_local`` remains consistent after any insertion.

    ``BaseLineGS.sample_from_nodes`` returns nodes sorted by global ID via
    ``torch.unique``.  When an isolated seed (no sampled edges) is not yet
    present in that sorted tensor, inserting it shifts the indices of all
    nodes with a larger global ID.  This function detects that case and
    rebuilds ``edge_index_local`` against the updated node ordering.

    Args:
        unique_nodes: 1-D tensor of global node IDs returned by the graph
            store, in sorted order.
        edge_index_local: ``[2, E]`` tensor of local (index-into-unique_nodes)
            source/destination pairs.
        seeds: 1-D tensor of seed node global IDs for the current batch.

    Returns:
        Tuple of ``(all_nodes, edge_index_local)`` where ``all_nodes`` is the
        sorted union of ``unique_nodes`` and ``seeds``, and
        ``edge_index_local`` is remapped to be consistent with ``all_nodes``.
    """
    all_nodes = torch.unique(torch.cat([unique_nodes, seeds]))
    if all_nodes.shape[0] != unique_nodes.shape[0] and edge_index_local.numel() > 0:
        old_to_new = torch.zeros(all_nodes.max().item() + 1, dtype=torch.long)
        old_to_new[all_nodes] = torch.arange(all_nodes.shape[0])
        global_edge = unique_nodes[edge_index_local]
        edge_index_local = old_to_new[global_edge]
    return all_nodes, edge_index_local


def build_sampler_output(
    graph_store,
    query: str,
    seeds: torch.Tensor,
    seed_time,
) -> SamplerOutput:
    """Run ``query`` against ``graph_store``, remap with seeds, and return a
    :class:`~torch_geometric.sampler.SamplerOutput`.

    Used by :class:`NeighborSampler` and :class:`PyGEquivalentSampler` to
    avoid duplicating the fetch + remap + output-construction pattern.
    """
    unique_nodes, edge_index_local = graph_store.sample_from_nodes(
        kwargs={"seed_ids": seeds.tolist()},
        query=query,
    )
    unique_nodes, edge_index_local = remap_with_seeds(unique_nodes, edge_index_local, seeds)
    return SamplerOutput(
        node=unique_nodes,
        row=edge_index_local[0],
        col=edge_index_local[1],
        edge=None,
        batch=None,
        metadata=(seeds, seed_time),
    )
