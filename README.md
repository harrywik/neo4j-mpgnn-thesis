# Large-graph GNN Training and Inference with Graph Database Backends

This repository contains all the code and experiments for our (Victor's and Harry's) master's thesis *"Cost-efficient graph database-backed Graph neural network pipelines"*, and the in-progress paper by us (and Borun Shi, Alfred Clemedtson, and Xuan Son-Vu) *"Towards Training and Inference of Graph Neural Networks in Graph Databases"*. The project benchmarks multiple sampling, caching, and inference strategies for GNNs, built on [PyTorch Geometric](https://pyg.org/) with [Neo4j](https://neo4j.com/) as the graph database backend.

This repository shows how to benchmark training pipelines, run inference experiments, and compare the behavior of different samplers.

<details>
<summary><b>📋 Table of Contents</b></summary>

- [Large-graph GNN Training and Inference with Graph Database Backends](#large-graph-gnn-training-and-inference-with-graph-database-backends)
  - [Repository Structure](#repository-structure)
  - [Getting Started](#getting-started)
  - [Run Experiments](#run-experiments)
    - [Ingest Datasets](#ingest-datasets)
    - [Set Configs](#set-configs)
    - [Benchmark Training](#benchmark-training)
    - [Run Sampler Comparison](#run-sampler-comparison)
    - [Run Inference Experiments](#run-inference-experiments)
    - [Combine Results into Plots](#combine-results-into-plots)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

</details>

---

<details>
<summary><b>📁 Repository Structure</b></summary>

```text
src/
  neo4j_pyg/              PyG-compatible Neo4j backend (feature stores, graph stores, samplers, models, caches)
  training/               Training entry point, distributed training, early stopping
  comparison_experiments/  Experiment scripts (sampler comparison, inference, dataset comparison)
  benchmarking_tools/     Measurement, profiling, and plotting utilities
  configs/                JSON configs for training and inference experiments
data/                     Dataset ingestion scripts (Cora, CiteSeer, Coauthor, OGB)
neo4j-gcn-plugin/         Neo4j Java plugin for in-database GNN inference (Maven)
experiment_results/       Combined experiment outputs and plots
results/                  Raw experiment run outputs
gnn_models/               Serialized model specs and weights
related_works/            Related work papers
thesis_presentations/     Presentation slides
tests/                    Unit tests
run_benchmark.py          Top-level benchmark launcher
run_experiment.sh         Top-level experiment launcher
```

</details>

---

## Getting Started

<details>
<summary><b>📦 Requirements</b></summary>

- Python $\geq$ 3.12
- PyTorch 2.7
- PyTorch Geometric 2.7
- Neo4j (for Neo4j-backed implementations)
- Java 17+ & Maven (for the Neo4j GCN plugin)
- Redis (optional, for `redis_cache_neo4j`)
- Linux (x86\_64) for full GPU + PyG extension support

See [`pyproject.toml`](pyproject.toml) for the full, version-pinned dependency list.

</details>

```bash
uv sync   # creates a venv and installs all dependencies
```

Copy [`example.env`](example.env) to `.env` and fill in your Neo4j credentials and paths. You need a running Neo4j instance with the GDS and APOC libraries installed.

The Neo4j GCN plugin is built separately:

```bash
make build-plugin
```

Restart Neo4j after building the plugin for changes to take effect.

---

## Run Experiments

### Ingest Datasets

Before running experiments, load your dataset into Neo4j:

```bash
make ingest_cora        # or: ingest_arxiv, ingest_products, ingest_papers100M
make add_float_features_cora   # required for Cypher-based inference
```

> **Note on features:** Byte-array features (`embedding_bytes`) are faster for training; float features (`embedding_bytes_floats`) are needed for `in_db_cypher` inference. Training is ~2× slower when both feature types are present on the node.

### Set Configs

All experiment parameters are JSON files under `src/configs/`. Each Makefile target combines a dataset and an implementation config (e.g. `make baseline_neo4j DATASET=cora` loads `cora.json` + `baseline_neo4j.json`):

```text
src/configs/training/
  datasets/               cora.json  arxiv.json  citeseer.json  coauthor.json  products.json  papers100M.json
  implementations/        baseline_pyg.json  baseline_neo4j.json  multsampler.json  …  saint_neo4j.json
```


<details>
<summary><b>📄 Config → Paper implementation</b></summary>

**Training** — a run combines one *sampling strategy* + one *dataset* config.

| Sampling strategy (`implementations/`) | Dataset (`datasets/`) |
|---|---|
| `baseline_neo4j.json` — Cypher sampling | `cora.json` — Cora |
| `neo4j_java_sampler.json` — Java UDP sampling | `arxiv.json` — ogbn-arxiv |
| `preagg_neo4j.json` — Pre-aggregation | `products.json` — ogbn-products |
| | `papers100M.json` — ogbn-papers100M |
| | `coauthor.json` — Coauthor Physics |

**Inference** — a run combines a *model* config (architecture + which strategies to test) with a *dataset* config (data + fanout, batch size, node counts, and which training run produced the weights).

| Model (`models/`) | Dataset + inference params (`datasets/`) |
|---|---|
| `gcn.json` — GCN, hidden 64/64 | `cora.json` — Cora, fanout [10,5] |
| `gcn_arxiv.json` — GCN (Arxiv) | `arxiv.json` — ogbn-arxiv |
| `gcn_products.json` — GCN (Products) | `products.json` — ogbn-products |
| `gcn_papers100M.json` — GCN (Papers100M) | `papers100M.json` — ogbn-papers100M |
| `gcn_coauthor.json` — GCN (Coauthor) | `coauthor.json` — Coauthor Physics |

The four inference methods (defined in the model config):

| Method | What it does |
|---|---|
| `full_graph` | Loads the entire graph into RAM via PyG; forward pass in Python (no DB involved) |
| `neighborhood_sampling` | Fetches k-hop subgraphs from Neo4j via Cypher; forward pass in Python |
| `in_db_cypher` | Inference runs entirely inside Neo4j using Cypher queries |
| `in_db_java` | Inference runs inside Neo4j via the Java UDP plugin |

</details>

### Benchmark Training

Run a single training run:

```bash
make baseline_neo4j DATASET=cora
```

### Run Sampler Comparison

Compares Neo4j GraphSAINT vs. PyG GraphSAINT random-walk samplers under identical settings:

```bash
make sampler_comparison DATASET=cora
```

### Run Inference Experiments

```bash
make inference_experiment INFERENCE_DATASET=cora
```

### Combine Results into Plots

```bash
make combine_results DIRS_CMP="path/to/run_1 path/to/run_2"
```

All experiment outputs are written to:

| Experiment | Output location |
|---|---|
| Training runs | `experiment_results/results/run_N_YYYY-MM-DD/` |
| Combined plots | `experiment_results/results/combined/` |
| Sampler comparison | `experiment_results/sampler_comparison/` |
| Inference experiments | `results/inference_comparison/` |

---

## Citation
coming soon...
<!-- ```bibtex
@mastersthesis{pekkariwik2026,
  title  = {Neo4j-Backed Message-Passing Graph Neural Networks},
  author = {Victor Pekkari and Harry Wik},
  school = {[University]},
  year   = {2026}
}
``` -->

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Acknowledgements

This project builds on [PyTorch Geometric](https://pyg.org/) and the [Neo4j Graph Data Science](https://neo4j.com/product/graph-data-science/) library.
