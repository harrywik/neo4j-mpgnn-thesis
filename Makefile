VENV ?= .venv
PY := $(VENV)/bin/python
PYTHONPATH := src
GNNSRC := src/gnn_implementations
DATASET ?= cora
IMPLS := baseline_multsampler baseline_neo4j baseline_pyg \
	cache_multsampler cache_neo4j distributed saint_pyg saint_neo4j multsampler neo4j_udp neo4j_java_sampler neo4j_udp_hybrid_last_hop
NBR_RUNS ?= 3
EXPERIMENTS := sampler_comparison compare_implementations compare_datasets
DATASET_TARGETS := cache_multsampler_cora cache_multsampler_arxiv \
	cache_neo4j_cora cache_neo4j_arxiv

# Neo4j plugins directory — override on the command line:
#   make build-plugin NEO4J_PLUGINS_DIR=/var/lib/neo4j/plugins
NEO4J_PLUGINS_DIR ?= $(shell [ -f .env ] && awk 'BEGIN{FS="="} /^NEO4J_PLUGINS_DIR=/{val=substr($$0, index($$0, "=")+1); gsub(/^"|"$$/, "", val); print val; exit}' .env)

.PHONY: run help $(IMPLS) $(DATASET_TARGETS) baseline_db $(EXPERIMENTS) ingest_cora ingest_arxiv ingest_products ingest_papers100M summarise combine build-plugin neo4j_udp_sign

help:
	@echo "Usage: make <implementation> [DATASET=cora]"
	@echo "       make run SCRIPT=path/to/script.py"
	@echo ""
	@echo "Available datasets: cora arxiv products papers100M"
	@echo "Available implementations: baseline_neo4j baseline_pyg cache_multsampler cache_neo4j"
	@echo "                           multsampler neo4j_udp neo4j_udp_hybrid_last_hop neo4j_java_sampler neo4j_udp_sign saint_neo4j saint_pyg distributed"
	@echo ""
	@echo "Compare multiple implementations:"
	@echo "  make compare_implementations IMPLS_CMP=\"baseline_neo4j multsampler\" [DATASET=cora] [NBR_RUNS=3]"
	@echo "Compare one implementation across datasets:"
	@echo "  make compare_datasets IMPL_CMP=baseline_pyg DATASETS_CMP=\"cora arxiv\" [NBR_RUNS=3]"

run:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "SCRIPT is required, e.g. make run SCRIPT=src/gnn_implementations/baseline_pyg/cora.py"; \
		exit 1; \
	fi
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT)

$(IMPLS):
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset $(DATASET) --implementation $@

neo4j_udp_sign:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset $(DATASET) --implementation neo4j_udp_sign

baseline_db:
	@$(MAKE) baseline_neo4j DATASET=$(DATASET)

cache_multsampler_cora:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset cora --implementation cache_multsampler

cache_multsampler_arxiv:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset arxiv --implementation cache_multsampler

cache_neo4j_cora:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset cora --implementation cache_neo4j

cache_neo4j_arxiv:
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m training.Main --dataset arxiv --implementation cache_neo4j

sampler_comparison:
	@DATASET=$(DATASET) PYTHONPATH=$(PYTHONPATH) $(PY) src/comparison_experiments/sampler_comparison/run_experiment.py

compare_implementations:
	@if [ -z "$(IMPLS_CMP)" ]; then \
		echo "IMPLS_CMP is required, e.g. make compare_implementations IMPLS_CMP=\"baseline_neo4j multsampler baseline_pyg\" DATASET=cora NBR_RUNS=3"; \
		exit 1; \
	fi
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m comparison_experiments.compare_implementations \
		--dataset $(DATASET) \
		--implementations $(IMPLS_CMP) \
		--nbr_runs $(NBR_RUNS)

compare_datasets:
	@if [ -z "$(IMPL_CMP)" ]; then \
		echo "IMPL_CMP is required, e.g. IMPL_CMP=baseline_pyg"; \
		exit 1; \
	fi
	@if [ -z "$(DATASETS_CMP)" ]; then \
		echo "DATASETS_CMP is required, e.g. DATASETS_CMP=\"cora arxiv products\""; \
		exit 1; \
	fi
	@PYTHONPATH=$(PYTHONPATH) $(PY) -m comparison_experiments.compare_datasets \
		--implementation $(IMPL_CMP) \
		--datasets $(DATASETS_CMP) \
		--nbr_runs $(NBR_RUNS)

ingest_cora:
	@PYTHONPATH=$(PYTHONPATH) $(PY) data/cora/new_ingest.py

ingest_arxiv:
	@PYTHONPATH=$(PYTHONPATH) $(PY) data/arxiv/ingest.py

ingest_products:
	@PYTHONPATH=$(PYTHONPATH) $(PY) data/ogbn-products/ingest.py

ingest_papers100M:
	@PYTHONPATH=$(PYTHONPATH) $(PY) data/ogbn-papers100M/ingest.py

summarise:
	@PYTHONPATH=$(PYTHONPATH) $(PY) src/benchmarking_tools/summarise.py

combine:
	@if [ -z "$(FILES)" ]; then \
		echo "FILES is required, e.g. make combine FILES=\"results/run_6 results/run_7\""; \
		exit 1; \
	fi
	@PYTHONPATH=$(PYTHONPATH) $(PY) src/benchmarking_tools/combine_plots.py $(FILES)

# ---------------------------------------------------------------------------
# Java plugin build
# ---------------------------------------------------------------------------
# Builds neo4j-gcn-plugin.jar and optionally copies it to the Neo4j plugins
# directory so the procedure is immediately available after a server restart.
#
# Usage:
#   make build-plugin                           # build only
#   make build-plugin NEO4J_PLUGINS_DIR=/var/lib/neo4j/plugins   # build + deploy
#   # or set NEO4J_PLUGINS_DIR in .env and just run: make build-plugin
#   make build-plugin NEO4J_PLUGINS_DIR="/Users/victorpekkari/Library/Application Support/neo4j-desktop/Application/Data/dbmss/dbms-9cbe0496-69f2-405d-8ef4-8d5204cd0058/plugins"
#   make build-plugin NEO4J_VERSION=5.22.0      # override Neo4j version
#
NEO4J_VERSION ?= 2025.12.1

build-plugin:
	@echo "Building neo4j-gcn-plugin (neo4j=$(NEO4J_VERSION))…"
	@cd neo4j-gcn-plugin && mvn clean package -q -Dneo4j.version=$(NEO4J_VERSION)
	@echo "Built: neo4j-gcn-plugin/target/neo4j-gcn-plugin-1.0.0.jar"
	@if [ -n "$(NEO4J_PLUGINS_DIR)" ]; then \
		cp neo4j-gcn-plugin/target/neo4j-gcn-plugin-1.0.0.jar "$(NEO4J_PLUGINS_DIR)/"; \
		echo "Deployed to $(NEO4J_PLUGINS_DIR)"; \
		echo "Restart Neo4j and run: SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gnnProcedures.' RETURN name ORDER BY name"; \
	else \
		echo "Built plugin jar. Set NEO4J_PLUGINS_DIR in .env or pass it on the command line to auto-deploy."; \
	fi
