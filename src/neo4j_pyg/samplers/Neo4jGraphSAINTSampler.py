import os.path as osp
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

_SAMPLERS_DIR = Path(__file__).resolve().parent.parent
if str(_SAMPLERS_DIR) not in sys.path:
    sys.path.insert(0, str(_SAMPLERS_DIR))

from neo4j_pyg.graph_stores.Neo4jAbstractGS import Neo4jAbstractGS
from neo4j_pyg.feature_stores.Neo4jAbstractFS import Neo4jAbstractFS
from benchmarking_tools import Measurer


class Neo4jGraphSAINTSampler(torch.utils.data.DataLoader):
    """Base class for Neo4j-backed GraphSAINT samplers.

    Mirrors the structure of PyG's ``GraphSAINTSampler``: this class provides
    all common infrastructure (subgraph assembly, norm pre-computation,
    ``_collate``) and leaves the concrete node-sampling strategy to child
    classes via the abstract :meth:`_sample_nodes` method.

    Child classes must:

    * Set any variant-specific attributes **before** calling
      ``super().__init__``.
    * Override :meth:`_setup` to build their sampling query (called inside
      ``__init__`` after all common attributes are set).
    * Implement :meth:`_sample_nodes` to return a list of global node IDs for
      one subgraph sample.

    Two DB round-trips are made per training batch:

    1. :meth:`_sample_nodes` — child-specific query that returns node IDs.
    2. The base-class subgraph query — fetches induced edges + features +
       labels + splits for those node IDs.

    Normalization coefficients (``node_norm``) are pre-computed exactly as in
    PyG: ``node_norm[v] = num_samples / visit_count[v] / N``.

    Args:
        graph_store: Neo4j graph store providing DB connection,
            ``nodeid_property``, ``split_property_name``, and
            ``fetch_ordered_subgraph``.
        feature_store: Neo4j feature store providing ``feature_property``,
            ``target_property``, and ``feature_property_type`` metadata.
        batch_size: Passed to :meth:`_sample_nodes` to control subgraph size.
        num_steps: Subgraphs produced per epoch (i.e. ``__len__``).
        sample_coverage: Target average visit count per node for norm
            pre-computation.  ``0`` disables norm computation.
        save_dir: Directory for caching computed norms to disk.
        log: Show a ``tqdm`` progress bar during norm pre-computation.
        rel_type: Neo4j relationship type filter (``None`` = all types).
        node_label: Neo4j node label filter (``None`` = all labels).
        measurer: Benchmarking measurer for per-batch timing events.
        profile: Prefix the sampling queries with ``PROFILE`` so that Neo4j
            returns execution-plan statistics.  The ``graph_store`` must have a
            :class:`~benchmarking_tools.QueryProfileAccumulator` attached (via
            its ``profile_accumulator`` attribute) for the data to be
            collected and saved.  Has no effect during norm pre-computation
            (the ``logged=False`` path) to avoid unnecessary overhead.
    """

    def __init__(
        self,
        graph_store: Neo4jAbstractGS,
        feature_store: Neo4jAbstractFS,
        batch_size: int,
        num_steps: int = 1,
        sample_coverage: int = 50,
        save_dir: Optional[str] = None,
        log: bool = True,
        rel_type: Optional[str] = None,
        node_label: Optional[str] = None,
        measurer: Optional[Measurer] = None,
        profile: bool = False,
    ) -> None:
        self.graph_store = graph_store
        self.feature_store = feature_store
        self._batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log
        self.rel_type = rel_type
        self.node_label = node_label
        self.measurer = measurer
        self.profile = profile
        self.nodeid_property = graph_store.nodeid_property

        # Build the induced-subgraph query shared by all variants.
        self._subgraph_query = self._build_subgraph_query()

        # Hook: child classes build their sampling query here, after all
        # common attributes above are in place.
        self._setup()

        # DataLoader with batch_size=1: one subgraph record per collate call.
        super().__init__(
            dataset=self,
            batch_size=1,
            collate_fn=self._collate,
        )

        self.N: int = self._get_total_nodes()

        self.node_norm: Optional[dict] = None
        self._default_norm: float = 1.0

        if self.sample_coverage > 0:
            path = osp.join(save_dir or "", self._filename)
            if save_dir is not None and osp.exists(path):
                self.node_norm, self._default_norm = torch.load(path)
            else:
                self.node_norm, self._default_norm = self._compute_norm()
                if save_dir is not None:
                    torch.save((self.node_norm, self._default_norm), path)

    # ------------------------------------------------------------------
    # Child-class interface
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Post-attribute-init hook for child classes.

        Called inside ``__init__`` after all common attributes (``graph_store``,
        ``nodeid_property``, ``rel_type``, etc.) are set, but before the
        DataLoader is initialised and before norm pre-computation starts.

        Override this in child classes to build variant-specific sampling
        queries.  The base implementation is a no-op.
        """

    def _sample_nodes(self, logged: bool = True) -> list[int]:
        """Return a list of global node IDs forming one subgraph sample.

        Must be overridden by child classes.

        Args:
            logged: When ``True`` the DB call is routed through
                ``graph_store.fetch_ordered_subgraph`` so that timing events
                are logged to the graph store's measurer (training path).
                When ``False`` the call bypasses the measurer via
                ``_run_query_raw`` (norm pre-sampling path).
        """
        raise NotImplementedError

    @property
    def _filename(self) -> str:
        """Cache filename for serialised norms.  Override in child classes."""
        return f"neo4j_graphsaint_{self.sample_coverage}.pt"

    # ------------------------------------------------------------------
    # DataLoader protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, idx: int) -> Optional[object]:
        """Sample one subgraph from Neo4j.

        Calls :meth:`_sample_nodes` (child-specific) to obtain a set of node
        IDs, then runs the shared subgraph query to fetch induced edges,
        features, labels, and splits.

        The ``idx`` argument is ignored; every call performs a fresh sample,
        matching PyG's ``GraphSAINTSampler.__getitem__`` behaviour.
        """
        t_sample = time.monotonic()
        node_ids = self._sample_nodes(logged=True)
        if self.measurer is not None:
            self.measurer.log_event(
                "sample_nodes_ms", (time.monotonic() - t_sample) * 1000
            )

        if not node_ids:
            return None

        t_subgraph = time.monotonic()
        subgraph_record = self.graph_store.sample_from_nodes(
            self._subgraph_query, {"node_ids": node_ids}
        )
        if self.measurer is not None:
            self.measurer.log_event(
                "subgraph_fetch_ms", (time.monotonic() - t_subgraph) * 1000
            )

        return subgraph_record

    def _collate(self, data_list: list) -> Data:
        """Assemble a ``torch_geometric.data.Data`` batch from one Neo4j record.

        Builds:

        * ``x``          — node feature matrix (float32).
        * ``y``          — node label vector (int64).
        * ``edge_index`` — local-index edge tensor ``[2, E]``.
        * ``train_mask`` — bool mask over nodes in the training split.
        * ``node_norm``  — per-node importance weights (only when
                           ``sample_coverage > 0``).
        * ``num_nodes``  — number of nodes in the subgraph.
        """
        assert len(data_list) == 1
        record = data_list[0]

        if record is None or not record["node_ids"]:
            return Data(
                x=torch.zeros((0, 0)),
                y=torch.zeros(0, dtype=torch.long),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                train_mask=torch.zeros(0, dtype=torch.bool),
                num_nodes=0,
            )

        node_ids: list = list(record["node_ids"])
        features = record["features"]
        labels = record["labels"]
        splits = record["splits"]
        edge_pairs = record["edge_pairs"]

        # --- Feature tensor (x) -----------------------------------------
        fpt = self.feature_store.feature_property_type
        if fpt == "byte[]":
            x = torch.from_numpy(
                np.stack(
                    [np.frombuffer(bytes(f), dtype=np.float32) for f in features]
                )
            )
        else:
            x = torch.from_numpy(np.array(features, dtype=np.float32))

        # --- Label tensor (y) -------------------------------------------
        label_map: dict = getattr(self.feature_store, "_labels", {})
        processed: list = []
        for raw in labels:
            if isinstance(raw, str):
                if raw not in label_map:
                    label_map[raw] = len(label_map)
                processed.append(label_map[raw])
            else:
                processed.append(int(raw))
        y = torch.tensor(processed, dtype=torch.long)

        # --- Edge index (local indices) ---------------------------------
        global_to_local = {nid: i for i, nid in enumerate(node_ids)}
        if edge_pairs:
            row = torch.tensor(
                [global_to_local[e[0]] for e in edge_pairs], dtype=torch.long
            )
            col = torch.tensor(
                [global_to_local[e[1]] for e in edge_pairs], dtype=torch.long
            )
            edge_index = torch.stack([row, col], dim=0)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        # --- Train mask -------------------------------------------------
        if self.graph_store.split_property_type == "str":
            train_mask = torch.tensor(
                [s == "train" for s in splits], dtype=torch.bool
            )
        else:
            train_mask = torch.tensor(
                [int(s) == 0 for s in splits], dtype=torch.bool
            )

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            num_nodes=len(node_ids),
        )

        # --- Node norm --------------------------------------------------
        if self.sample_coverage > 0 and self.node_norm is not None:
            norms = [
                self.node_norm.get(nid, self._default_norm) for nid in node_ids
            ]
            data.node_norm = torch.tensor(norms, dtype=torch.float)

        return data

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_subgraph_query(self) -> str:
        """Build the induced-subgraph Cypher query shared by all variants.

        Given ``$node_ids`` (a list of global node IDs), returns the induced
        subgraph plus node properties in one round-trip:

        * ``node_ids``   — ordered list of global node IDs in the subgraph.
        * ``features``   — list of feature vectors (one per node).
        * ``labels``     — list of label values (one per node).
        * ``splits``     — list of split values (one per node).
        * ``edge_pairs`` — list of ``[src_id, dst_id]`` for induced edges.
        """
        node_lbl = f":{self.node_label}" if self.node_label else ""
        rel_flt = f":{self.rel_type}" if self.rel_type else ""
        nid = self.nodeid_property
        feat = self.feature_store.feature_property
        lbl = self.feature_store.target_property
        split = self.graph_store.split_property_name

        profile_prefix = "PROFILE\n        " if self.profile else ""
        return f"""
        {profile_prefix}MATCH (n{node_lbl}) WHERE n.{nid} IN $node_ids
        WITH collect(n) AS nodes
        CALL (nodes) {{
            UNWIND nodes AS a
            OPTIONAL MATCH (a)-[r{rel_flt}]->(b) WHERE b IN nodes
            RETURN collect(
                CASE WHEN b IS NOT NULL THEN [a.{nid}, b.{nid}] END
            ) AS edge_pairs
        }}
        RETURN
            [n IN nodes | n.{nid}]   AS node_ids,
            [n IN nodes | n.{feat}]  AS features,
            [n IN nodes | n.{lbl}]   AS labels,
            [n IN nodes | n.{split}] AS splits,
            edge_pairs
        """

    def _get_total_nodes(self) -> int:
        """Query the total number of nodes in the graph (filtered by node_label)."""
        node_lbl = f":{self.node_label}" if self.node_label else ""
        query = f"MATCH (n{node_lbl}) RETURN count(n) AS total"
        with self.graph_store._get_driver().session(
            database=self.graph_store.database_name
        ) as session:
            record = session.run(query).single()
        return int(record["total"]) if record else 0

    def _compute_norm(self) -> tuple[dict, float]:
        """Pre-sample subgraphs to estimate per-node normalization weights.

        Delegates sampling to :meth:`_sample_nodes` with ``logged=False`` so
        that norm pre-sampling does not pollute training metrics.

        Mirrors PyG's formula:

            node_norm[v] = num_samples / visit_count[v] / N

        Runs until ``total_sampled_nodes >= N * sample_coverage``.

        Returns:
            Tuple of ``(node_norm dict, default_norm float)``.  The default
            norm is applied in ``_collate`` to nodes that were never visited
            during pre-sampling (count treated as 0.1, matching PyG).
        """
        node_count: defaultdict = defaultdict(float)
        total_sampled_nodes = 0
        num_samples = 0
        target = self.N * self.sample_coverage

        pbar = None
        if self.log:
            pbar = tqdm(
                total=int(target),
                desc="Computing Neo4j GraphSAINT normalization",
            )

        while total_sampled_nodes < target:
            node_ids = self._sample_nodes(logged=False)
            if not node_ids:
                continue
            for nid in node_ids:
                node_count[nid] += 1
            total_sampled_nodes += len(node_ids)
            num_samples += 1
            if pbar is not None:
                pbar.update(len(node_ids))

        if pbar is not None:
            pbar.close()

        # Nodes with zero visits get count=0.1 (same as PyG).
        default_norm = float(num_samples) / 0.1 / self.N if self.N > 0 else 1.0
        node_norm: dict = {
            nid: float(num_samples) / max(count, 0.1) / self.N
            for nid, count in node_count.items()
        }
        return node_norm, default_norm

    def _run_query_raw(self, query: str, **kwargs) -> Optional[dict]:
        """Execute a Cypher query directly, bypassing measurer logging.

        Used by :meth:`_sample_nodes` implementations when ``logged=False``
        to avoid polluting training metrics during norm pre-sampling.
        """
        with self.graph_store._get_driver().session(
            database=self.graph_store.database_name
        ) as session:
            records = list(session.run(query, **kwargs))
        return records[0] if records else None


class Neo4jGraphSAINTRandomWalkSampler(Neo4jGraphSAINTSampler):
    """GraphSAINT Random Walk Sampler backed by Neo4j.

    Concrete child of :class:`Neo4jGraphSAINTSampler` that implements
    :meth:`_sample_nodes` via random walks in Neo4j.

    Each call to :meth:`_sample_nodes` runs a single Cypher query that:

    1. Picks ``batch_size`` random root nodes.
    2. Advances each walker ``walk_length`` steps through randomly chosen
       outgoing edges (staying put when a node has no outgoing edges).
    3. Returns the deduplicated union of all visited node IDs.

    The walk query is built once at construction time by :meth:`_setup`,
    following the same unrolled-``CALL``-block pattern used in
    ``Neo4jNeighborSampler._build_fanout_query``.

    Args:
        graph_store: See :class:`Neo4jGraphSAINTSampler`.
        feature_store: See :class:`Neo4jGraphSAINTSampler`.
        batch_size: Number of random-walk root nodes per subgraph.
        walk_length: Number of steps per random walk.
        num_steps: See :class:`Neo4jGraphSAINTSampler`.
        sample_coverage: See :class:`Neo4jGraphSAINTSampler`.
        save_dir: See :class:`Neo4jGraphSAINTSampler`.
        log: See :class:`Neo4jGraphSAINTSampler`.
        rel_type: See :class:`Neo4jGraphSAINTSampler`.
        node_label: See :class:`Neo4jGraphSAINTSampler`.
        measurer: See :class:`Neo4jGraphSAINTSampler`.
        profile: See :class:`Neo4jGraphSAINTSampler`.
    """

    def __init__(
        self,
        graph_store: Neo4jAbstractGS,
        feature_store: Neo4jAbstractFS,
        batch_size: int,
        walk_length: int,
        num_steps: int = 1,
        sample_coverage: int = 50,
        save_dir: Optional[str] = None,
        log: bool = True,
        rel_type: Optional[str] = None,
        node_label: Optional[str] = None,
        measurer: Optional[Measurer] = None,
        profile: bool = False,
    ) -> None:
        # walk_length must be set before super().__init__ because _setup()
        # (called inside super().__init__) needs it to build _walk_query.
        self.walk_length = walk_length
        super().__init__(
            graph_store=graph_store,
            feature_store=feature_store,
            batch_size=batch_size,
            num_steps=num_steps,
            sample_coverage=sample_coverage,
            save_dir=save_dir,
            log=log,
            rel_type=rel_type,
            node_label=node_label,
            measurer=measurer,
            profile=profile,
        )

    # ------------------------------------------------------------------
    # Child-class overrides
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        """Build the walk query after common attributes are available.

        Two variants are stored:

        * ``_walk_query`` — used during training (``logged=True``).  Prefixed
          with ``PROFILE`` when ``self.profile`` is ``True``.
        * ``_walk_query_unprofile`` — always plain Cypher, used during norm
          pre-computation (``logged=False``) to avoid profiling overhead on
          thousands of throwaway samples.
        """
        base_query = self._build_walk_query()
        self._walk_query_unprofile = base_query
        self._walk_query = ("PROFILE\n        " + base_query) if self.profile else base_query

    @property
    def _filename(self) -> str:
        return f"neo4j_graphsaint_rw_{self.walk_length}_{self.sample_coverage}.pt"

    def _sample_nodes(self, logged: bool = True) -> list[int]:
        """Run the random walk query and return the visited node IDs.

        Args:
            logged: Routes the DB call through
                ``graph_store.fetch_ordered_subgraph`` (metrics logged) when
                ``True``, or through ``_run_query_raw`` (no metrics) when
                ``False``.
        """
        if logged:
            record = self.graph_store.sample_from_nodes(
                self._walk_query, {"batch_size": self._batch_size}
            )
        else:
            record = self._run_query_raw(
                self._walk_query_unprofile, batch_size=self._batch_size
            )
        if record is None or not record["node_ids"]:
            return []
        return list(record["node_ids"])

    def _build_walk_query(self) -> str:
        """Build the random walk Cypher query for this sampler.

        Picks ``$batch_size`` random root nodes via ``apoc.coll.randomItems``
        (O(N) shuffle, no sort), then unrolls ``walk_length`` ``CALL`` blocks
        (one per step).  Each block advances every walker to a randomly chosen
        incoming neighbor, staying put when the current node has no such edges.
        Visited node IDs are accumulated as a raw list throughout and
        deduplicated only once at the end via ``apoc.coll.toSet``.

        Parameters injected at runtime: ``$batch_size``.
        """
        node_lbl = f":{self.node_label}" if self.node_label else ""
        rel_flt = f":{self.rel_type}" if self.rel_type else ""
        nid = self.nodeid_property

        q: list[str] = []

        # 1. Pick random roots via apoc.coll.randomItems (O(N), no sort) and
        #    initialise the running visited list.
        q.append(f"""
        MATCH (root{node_lbl})
        WITH collect(root) AS all_nodes
        WITH apoc.coll.randomItems(all_nodes, $batch_size, false) AS roots
        WITH roots AS walkers,
             [n IN roots | n.{nid}] AS visited_ids
        """)

        # 2. Unroll walk_length steps, one CALL block each.
        #    Walker order is not preserved (no index tracking) since only the
        #    visited ID set matters at the end.  Deduplication is deferred to
        #    the final RETURN to avoid repeated apoc.coll.toSet calls.
        for step in range(self.walk_length):
            q.append(f"""
        // Walk step {step + 1} of {self.walk_length}
        CALL (walkers) {{
            UNWIND walkers AS walker
            OPTIONAL MATCH (walker)<-[r{rel_flt}]-(nbr{node_lbl})
            WITH walker, collect(nbr) AS candidates
            WITH CASE WHEN size(candidates) > 0
                      THEN candidates[toInteger(rand() * size(candidates))]
                      ELSE walker
                 END AS next_pos
            RETURN collect(next_pos) AS new_walkers
        }}
        WITH new_walkers AS walkers,
             visited_ids + [n IN new_walkers | n.{nid}] AS visited_ids
        """)

        # 3. Deduplicate once and return.
        q.append("""
        RETURN apoc.coll.toSet(visited_ids) AS node_ids
        """)

        return "\n".join(q)
