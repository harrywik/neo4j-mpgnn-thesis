"""Main.py — unified entry point for all GNN training experiments.

Usage
-----
    python -m training.Main --dataset cora --implementation baseline_neo4j
    python -m training.Main --dataset arxiv --implementation multsampler
    python -m training.Main --dataset cora --implementation neo4j_udp
    python -m training.Main --dataset cora --implementation saint_neo4j

The --dataset argument selects a JSON from src/configs/datasets/.
The --implementation argument selects a JSON from src/configs/implementations/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure src/ is on the path when run directly.
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CONFIGS_DIR = SRC_DIR / "configs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_configs(dataset: str, implementation: str) -> tuple[dict, dict]:
    dataset_cfg = _load_json(CONFIGS_DIR / "datasets" / f"{dataset}.json")
    impl_cfg = _load_json(CONFIGS_DIR / "implementations" / f"{implementation}.json")
    return dataset_cfg, impl_cfg


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------

def _build_common_kwargs(dataset_cfg: dict, uri: str, user: str, password: str) -> dict:
    """Args shared by most feature/graph store constructors."""
    return {
        "database_name":       dataset_cfg["database_name"],
        "dataset_name":        dataset_cfg["name"],
        "feature_property":    dataset_cfg["feature_property"],
        "feature_property_type": dataset_cfg["feature_property_type"],
        "target_property":     dataset_cfg["target_property"],
        "nodeid_property":     dataset_cfg["nodeid_property"],
        "split_property_name": dataset_cfg["split_property_name"],
        "split_property_type": dataset_cfg["split_property_type"],
        "node_label":          dataset_cfg["node_label"],
        "uri":  uri,
        "user": user,
        "pwd":  password,
    }


def _make_graph_store(impl_cfg: dict, common_kwargs: dict, driver, measurer, profile_accumulator):
    from training.registry import GRAPH_STORES, filter_kwargs
    gs_cfg = impl_cfg["graph_store"]
    cls = GRAPH_STORES[gs_cfg["class_name"]]
    kwargs = dict(common_kwargs)
    kwargs.update(gs_cfg.get("extra_kwargs", {}))
    kwargs["measurer"] = measurer
    kwargs["profile_accumulator"] = profile_accumulator
    if gs_cfg["connection_mode"] == "driver":
        kwargs["driver"] = driver
    return cls(**filter_kwargs(cls, kwargs))


def _make_sampler(impl_cfg: dict, graph_store, dataset_cfg: dict, measurer):
    from training.registry import SAMPLERS, filter_kwargs
    s_cfg = impl_cfg["sampler"]
    cls = SAMPLERS[s_cfg["class_name"]]
    kwargs = dict(s_cfg.get("extra_kwargs", {}))
    # Common sampler args
    if "num_neighbors" in s_cfg:
        kwargs["num_neighbors"] = s_cfg["num_neighbors"]
    if "hops" in s_cfg:
        kwargs["hops"] = s_cfg["hops"]
    if "max_neighbors" in s_cfg:
        kwargs["max_neighbors"] = s_cfg["max_neighbors"]
    if "max_neighbors_per_hop" in s_cfg:
        kwargs["max_neighbors_per_hop"] = s_cfg["max_neighbors_per_hop"]
    kwargs["node_label"] = dataset_cfg["node_label"]
    kwargs["edge_type"] = dataset_cfg.get("edge_type", "")
    kwargs["node_id_key"] = dataset_cfg["nodeid_property"]
    kwargs["feature_key"] = dataset_cfg["feature_property"]
    kwargs["feature_type"] = dataset_cfg["feature_property_type"]
    kwargs["measurer"] = measurer
    return cls(**filter_kwargs(cls, {"graph_store": graph_store, **kwargs}))


def _make_feature_store(impl_cfg: dict, common_kwargs: dict, driver, measurer,
                        profile_accumulator, sampler):
    from training.registry import FEATURE_STORES, filter_kwargs
    fs_cfg = impl_cfg["feature_store"]
    cls = FEATURE_STORES[fs_cfg["class_name"]]
    kwargs = dict(common_kwargs)
    kwargs.update(fs_cfg.get("extra_kwargs", {}))
    kwargs["measurer"] = measurer
    kwargs["profile_accumulator"] = profile_accumulator
    if fs_cfg["connection_mode"] == "driver":
        kwargs["driver"] = driver
    if impl_cfg.get("needs_sampler_in_fs"):
        kwargs["sampler"] = sampler
    return cls(**filter_kwargs(cls, kwargs))


def _make_model(impl_cfg: dict, dataset_cfg: dict):
    from training.registry import MODELS, filter_kwargs
    m_cfg = impl_cfg["model"]
    cls = MODELS[m_cfg["class_name"]]
    kwargs = {
        "in_dim":      dataset_cfg["feature_dim"],
        "feature_dim": dataset_cfg["feature_dim"],
        "hidden_dim1": dataset_cfg["default_hidden_dim1"],
        "hidden_dim2": dataset_cfg["default_hidden_dim2"],
        "nbr_classes": dataset_cfg["nbr_classes"],
    }
    kwargs.update(m_cfg.get("extra_kwargs", {}))
    if "hops" in m_cfg.get("extra_kwargs", {}):
        kwargs["hops"] = m_cfg["extra_kwargs"]["hops"]
    return cls(**filter_kwargs(cls, kwargs))


# ---------------------------------------------------------------------------
# Trainer factories
# ---------------------------------------------------------------------------

def _run_standard(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model):
    from training.Training import Trainer, put_nodeLoader_args_map
    from training.registry import filter_kwargs

    nla = impl_cfg["nodeloader_args"]
    nodeloader_args = put_nodeLoader_args_map(**nla)

    trainer = Trainer(
        model=model,
        data=(feature_store, graph_store),
        sampler=sampler,
        lr=dataset_cfg["lr"],
        patience=dataset_cfg["patience"],
        min_delta=dataset_cfg["min_delta"],
        batch_size=dataset_cfg["batch_size"],
        nodeloader_args=nodeloader_args,
        measurer=measurer,
        cpu_monitor_interval=1 if dataset_cfg.get("logg_cpu_utilization", True) else None,
        max_training_size=dataset_cfg.get("max_training_size"),
        max_validation_size=dataset_cfg.get("max_validation_size"),
        max_test_size=dataset_cfg.get("max_test_size"),
    )
    trainer.train(max_epochs=dataset_cfg["max_epochs"])


def _run_in_memory(dataset_cfg, impl_cfg, measurer, model):
    """Baseline PyG / saint_pyg path — loads Planetoid from disk."""
    import torch
    from torch_geometric.datasets import Planetoid
    from training.Training import Trainer, put_nodeLoader_args_map

    planetoid_root = dataset_cfg.get("planetoid_root", "data/Planetoid")
    planetoid_name = dataset_cfg.get("planetoid_name", dataset_cfg["name"].capitalize())
    dataset = Planetoid(root=planetoid_root, name=planetoid_name)
    graph = dataset[0]
    train_indices = torch.where(graph.train_mask.reshape(-1))[0].tolist()

    num_neighbors = impl_cfg.get("num_neighbors", [10, 5])

    nla = impl_cfg.get("nodeloader_args", {})
    from training.Training import put_nodeLoader_args_map
    nodeloader_args = put_nodeLoader_args_map(
        pickle_safe=nla.get("pickle_safe", False),
        shuffle=dataset_cfg.get("shuffle", True),
    )

    trainer = Trainer(
        model=model,
        data=graph,
        measurer=measurer,
        train_indices=train_indices,
        patience=dataset_cfg["patience"],
        min_delta=dataset_cfg["min_delta"],
        lr=dataset_cfg["lr"],
        batch_size=dataset_cfg["batch_size"],
        nodeloader_args=nodeloader_args,
        num_neighbors=num_neighbors,
    )
    trainer.train(max_epochs=dataset_cfg["max_epochs"])


def _run_saint_neo4j(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, model):
    from training.GraphSAINTTrainer import GraphSAINTTrainer
    from neo4j_pyg.samplers.Neo4jGraphSAINTSampler import Neo4jGraphSAINTRandomWalkSampler
    from neo4j_pyg.samplers.Neo4jNeighborSampler import Neo4jNeighborSampler

    saint = impl_cfg["saint_extra"]
    train_loader = Neo4jGraphSAINTRandomWalkSampler(
        graph_store=graph_store,
        feature_store=feature_store,
        batch_size=saint["graphsaint_batch_size"],
        walk_length=saint["walk_length"],
        num_steps=saint["num_steps"],
        sample_coverage=saint["sample_coverage"],
        node_label=dataset_cfg["node_label"],
        measurer=measurer,
        profile=dataset_cfg.get("profile", False),
    )

    eval_cfg = impl_cfg.get("eval_sampler", {})
    eval_num_neighbors = eval_cfg.get("num_neighbors", [10, 5])
    eval_sampler = Neo4jNeighborSampler(
        graph_store,
        num_neighbors=eval_num_neighbors,
        node_label=dataset_cfg["node_label"],
    )

    trainer = GraphSAINTTrainer(
        model=model,
        data=(feature_store, graph_store),
        train_loader=train_loader,
        eval_sampler=eval_sampler,
        patience=dataset_cfg["patience"],
        min_delta=dataset_cfg["min_delta"],
        lr=dataset_cfg["lr"],
        batch_size=saint["graphsaint_batch_size"],
        measurer=measurer,
        cpu_monitor_interval=1 if dataset_cfg.get("logg_cpu_utilization", True) else None,
        max_validation_size=dataset_cfg.get("max_validation_size"),
        max_test_size=dataset_cfg.get("max_test_size"),
    )
    trainer.train(max_epochs=dataset_cfg["max_epochs"])


def _run_saint_pyg(dataset_cfg, impl_cfg, measurer, model):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from training.GraphSAINTTrainer import GraphSAINTTrainer

    planetoid_root = dataset_cfg.get("planetoid_root", "data/Planetoid")
    planetoid_name = dataset_cfg.get("planetoid_name", dataset_cfg["name"].capitalize())
    dataset = Planetoid(root=planetoid_root, name=planetoid_name, transform=NormalizeFeatures())
    graph = dataset[0]

    saint = impl_cfg["saint_extra"]
    trainer = GraphSAINTTrainer(
        model=model,
        data=graph,
        patience=dataset_cfg["patience"],
        min_delta=dataset_cfg["min_delta"],
        lr=dataset_cfg["lr"],
        batch_size=saint["graphsaint_batch_size"],
        walk_length=saint["walk_length"],
        num_steps=saint["num_steps"],
        sample_coverage=saint["sample_coverage"],
        measurer=measurer,
        cpu_monitor_interval=1 if dataset_cfg.get("logg_cpu_utilization", True) else None,
        max_validation_size=dataset_cfg.get("max_validation_size"),
        max_test_size=dataset_cfg.get("max_test_size"),
    )
    trainer.train(max_epochs=dataset_cfg["max_epochs"])


def _run_distributed(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model):
    import torch.distributed as dist
    from training.DistributedTraining import DistributedTrainer

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch
        import torch.distributed as dist
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        dist.barrier()

    nla = impl_cfg["nodeloader_args"]
    from training.Training import put_nodeLoader_args_map
    nodeloader_args = put_nodeLoader_args_map(**nla)

    trainer = DistributedTrainer(
        model=model,
        feature_store=feature_store,
        graph_store=graph_store,
        sampler=sampler,
        batch_size=dataset_cfg["batch_size"],
        measurer=measurer,
    )
    trainer.train(max_epochs=dataset_cfg["max_epochs"])


# ---------------------------------------------------------------------------
# Measurer logging helpers
# ---------------------------------------------------------------------------

def _log_config(measurer, impl_cfg, dataset_cfg, sampler, model):
    measurer.write_to_configresult("dataset", dataset_cfg["name"])
    measurer.write_to_configresult("implementation", impl_cfg.get("_name", ""))
    if impl_cfg.get("feature_store"):
        measurer.write_to_configresult("feature_store", impl_cfg["feature_store"]["class_name"])
    if impl_cfg.get("graph_store"):
        measurer.write_to_configresult("graph_store", impl_cfg["graph_store"]["class_name"])
    if impl_cfg.get("sampler"):
        s_cfg = impl_cfg["sampler"]
        measurer.write_to_configresult("sampler", {
            "name": s_cfg["class_name"],
            **{k: v for k, v in s_cfg.items() if k != "class_name"},
        })
    measurer.write_to_configresult("model", {
        "class": impl_cfg["model"]["class_name"],
        **impl_cfg["model"].get("extra_kwargs", {}),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a GNN training experiment.")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (e.g. cora, arxiv, products, papers100M)")
    parser.add_argument("--implementation", required=True,
                        help="Implementation name (e.g. baseline_neo4j, neo4j_udp)")
    args = parser.parse_args()

    dataset_cfg, impl_cfg = load_configs(args.dataset, args.implementation)
    impl_cfg["_name"] = args.implementation

    uri      = os.environ.get("URI", "")
    user     = os.environ.get("USERNAME", "")
    password = os.environ.get("PASSWORD", "")

    in_memory = impl_cfg.get("in_memory", False)
    trainer_name = impl_cfg["trainer"]

    # -----------------------------------------------------------------------
    # Measurer
    # -----------------------------------------------------------------------
    from benchmarking_tools import Measurer, QueryProfileAccumulator
    profile = dataset_cfg.get("profile", False)
    profile_accumulator = QueryProfileAccumulator() if profile else None
    measurer = Measurer(dataset_cfg, profile_accumulator=profile_accumulator)

    # -----------------------------------------------------------------------
    # Model (always needed)
    # -----------------------------------------------------------------------
    model = _make_model(impl_cfg, dataset_cfg)

    # -----------------------------------------------------------------------
    # In-memory paths (baseline_pyg, saint_pyg)
    # -----------------------------------------------------------------------
    if in_memory:
        if trainer_name == "GraphSAINTTrainer":
            _run_saint_pyg(dataset_cfg, impl_cfg, measurer, model)
        else:
            _run_in_memory(dataset_cfg, impl_cfg, measurer, model)
        return

    # -----------------------------------------------------------------------
    # Neo4j-backed paths
    # -----------------------------------------------------------------------
    from Neo4jConnection import Neo4jConnection
    driver = Neo4jConnection(uri, user, password).get_driver()
    common_kwargs = _build_common_kwargs(dataset_cfg, uri, user, password)

    graph_store = _make_graph_store(impl_cfg, common_kwargs, driver, measurer, profile_accumulator)
    sampler = None
    if impl_cfg.get("sampler"):
        sampler = _make_sampler(impl_cfg, graph_store, dataset_cfg, measurer)

    feature_store = None
    if impl_cfg.get("feature_store"):
        feature_store = _make_feature_store(
            impl_cfg, common_kwargs, driver, measurer, profile_accumulator, sampler
        )

    _log_config(measurer, impl_cfg, dataset_cfg, sampler, model)

    if trainer_name == "GraphSAINTTrainer":
        _run_saint_neo4j(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, model)
    elif trainer_name == "DistributedTrainer":
        _run_distributed(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model)
    else:
        _run_standard(dataset_cfg, impl_cfg, measurer, feature_store, graph_store, sampler, model)


if __name__ == "__main__":
    main()
