"""GPU-cached Neo4j feature store.

Combines the PaGraph-style GPU cache (:class:`Neo4jGPUCache`) with the
standard Neo4j feature-store pipeline.  The hot path in
:meth:`_multi_get_tensor` is fully vectorised: a single tensor-index
operation retrieves all cached node features from GPU memory, and only
uncached nodes trigger a Neo4j round-trip.

Compared to ``Neo4jCachedFS``, this implementation avoids:
1. Per-node Python dict lookups on the hot path.
2. CPU → GPU transfer for cached nodes (features are already on the GPU).
3. GDS PageRank dependency for cache ranking (uses out-degree instead).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from neo4j import Driver
from torch_geometric.data.feature_store import TensorAttr
from torch_geometric.typing import FeatureTensorType

from benchmarking_tools import Measurer
from neo4j_pyg.feature_caches.Neo4jGPUCache import Neo4jGPUCache, prefill_gpu_cache_from_neo4j
from neo4j_pyg.feature_stores.Neo4jFS import Neo4jFS


class Neo4jGPUCachedFS(Neo4jFS):
    """Feature store with a GPU-resident static cache ranked by out-degree.

    Parameters
    ----------
    cache_size_GB:
        Cache budget in GB.  Ignored when ``auto_size=True``.
    auto_size:
        If ``True``, cache capacity is derived from free GPU memory
        (minus ``reserved_gb``) rather than ``cache_size_GB``.
    device:
        PyTorch device for the GPU cache (e.g. ``"cuda:0"``).
        Falls back to ``"cpu"`` when CUDA is unavailable.
    reserved_gb:
        GPU memory to keep free for model parameters and activations
        (only relevant when ``auto_size=True``).  Default: 1.0 GB.
    """

    def __init__(
        self,
        driver: Optional[Driver] = None,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        pwd: Optional[str] = None,
        measurer: Optional[Measurer] = None,
        database_name: Optional[str] = None,
        dataset_name: str = "neo4j",
        feature_property: str = "features",
        target_property: str = "category",
        split_property_name: str = "split",
        split_property_type: str = "int",
        nodeid_property: str = "nodeId",
        feature_property_type: str = "f64[]",
        label_map: Optional[Dict[str, int]] = None,
        cache_size_GB: float = 0.5,
        auto_size: bool = False,
        device: str = "cuda",
        reserved_gb: float = 1.0,
    ) -> None:
        cache_db = database_name if database_name else dataset_name

        # Build a driver for the prefill query if only credentials were given.
        if driver is None and uri is not None:
            from neo4j import GraphDatabase
            _drv = GraphDatabase.driver(uri, auth=(user, pwd))
        else:
            _drv = driver

        gpu_cache = prefill_gpu_cache_from_neo4j(
            driver=_drv,
            database_name=cache_db,
            nodeid_property=nodeid_property,
            feature_property=feature_property,
            target_property=target_property,
            feature_property_type=feature_property_type,
            label_map=label_map,
            device=device,
            auto_size=auto_size,
            cache_size_GB=cache_size_GB,
            reserved_gb=reserved_gb,
        )
        super().__init__(
            driver=driver,
            uri=uri,
            user=user,
            pwd=pwd,
            measurer=measurer,
            database_name=database_name,
            dataset_name=dataset_name,
            feature_property=feature_property,
            target_property=target_property,
            split_property_name=split_property_name,
            split_property_type=split_property_type,
            nodeid_property=nodeid_property,
            feature_property_type=feature_property_type,
            cache=gpu_cache,
        )
        self._gpu_cache: Neo4jGPUCache = gpu_cache

        n_cached = len(self._gpu_cache._nid_to_cache_idx)
        feat_dim = self._gpu_cache._feat_dim or 0
        device_used = str(self._gpu_cache.device)
        print(
            f"GPUFeatureCache: {n_cached} nodes cached "
            f"(feat_dim={feat_dim}, device={device_used})"
        )

    # ------------------------------------------------------------------
    # Vectorised hot path
    # ------------------------------------------------------------------

    def _multi_get_tensor(
        self, attrs: List[TensorAttr]
    ) -> List[Optional[FeatureTensorType]]:
        """Fetch features + labels with vectorised GPU cache lookup.

        For each batch:
        1. ``fetch_batch`` splits node IDs into cached (GPU) and uncached.
        2. Uncached nodes are fetched from Neo4j in one round-trip.
        3. Both parts are assembled into output tensors on ``self._gpu_cache.device``.
        """
        x_attr = next((a for a in attrs if a.attr_name == "x"), None)
        y_attr = next((a for a in attrs if a.attr_name == "y"), None)

        # Fall back to parent implementation if attrs don't match standard pair.
        if x_attr is None or y_attr is None:
            return [self._get_tensor(attr) for attr in attrs]

        node_ids: List[int] = x_attr.index.tolist()
        n = len(node_ids)
        dev = self._gpu_cache.device

        # --- vectorised cache split ---
        cached_x, cached_y, cached_pos, uncached_nids, uncached_pos = (
            self._gpu_cache.fetch_batch(node_ids)
        )

        n_cached = len(cached_pos)
        n_uncached = len(uncached_nids)

        if self.measurer is not None:
            self.measurer.log_event("cache_hit", n_cached)
            self.measurer.log_event("cache_miss", n_uncached)

        # Determine feat_dim.
        feat_dim = self._gpu_cache._feat_dim or self._feature_dim
        if feat_dim is None and cached_x is not None:
            feat_dim = cached_x.shape[1]
            self._feature_dim = feat_dim

        # Allocate output tensors on the cache device.
        if feat_dim is not None:
            feat_out = torch.empty(n, feat_dim, dtype=torch.float32, device=dev)
            y_out = torch.empty(n, dtype=torch.long, device=dev)
        else:
            feat_out = None
            y_out = None

        # Place cached entries.
        if n_cached > 0 and feat_out is not None:
            cp = torch.tensor(cached_pos, dtype=torch.long, device=dev)
            feat_out[cp] = cached_x
            y_out[cp] = cached_y

        # Fetch uncached entries from Neo4j.
        if n_uncached > 0:
            if self.measurer is not None:
                self.measurer.log_event("remote_feature_fetch", 1)

            fetched_nids, feat_matrix, y_array = self._get_both_from_db(
                uncached_nids, x_attr
            )

            if feat_dim is None and len(feat_matrix) > 0:
                feat_dim = feat_matrix.shape[1]
                self._feature_dim = feat_dim
                feat_out = torch.empty(n, feat_dim, dtype=torch.float32, device=dev)
                y_out = torch.empty(n, dtype=torch.long, device=dev)
                if n_cached > 0:
                    cp = torch.tensor(cached_pos, dtype=torch.long, device=dev)
                    feat_out[cp] = cached_x
                    y_out[cp] = cached_y

            if feat_out is not None and len(fetched_nids) > 0:
                # Build position map for DB-returned nodes (order may differ).
                nid_to_uncached_pos = {
                    nid: uncached_pos[i] for i, nid in enumerate(uncached_nids)
                }
                out_positions = [nid_to_uncached_pos[nid] for nid in fetched_nids]
                op = torch.tensor(out_positions, dtype=torch.long, device=dev)

                feat_out[op] = torch.from_numpy(feat_matrix).to(dev)
                y_out[op] = torch.from_numpy(y_array).to(dev)

        self.t_feat_etl_start = time.monotonic()

        if self.measurer is not None:
            self.measurer.log_event("end_etl", 1)
            self.measurer.set_phase("sampling")
            self.measurer.log_event(
                "feat_x_etl_ms",
                (time.monotonic() - self.t_feat_etl_start) * 1000,
            )

        if feat_out is None:
            # Nothing was fetched at all.
            feat_out = torch.empty(0, 0, dtype=torch.float32)
            y_out = torch.empty(0, dtype=torch.long)

        result: List[Optional[FeatureTensorType]] = []
        for attr in attrs:
            if attr.attr_name == "x":
                result.append(feat_out)
            elif attr.attr_name == "y":
                result.append(y_out)
            else:
                raise ValueError(f"Unsupported attribute: {attr.attr_name}")
        return result
