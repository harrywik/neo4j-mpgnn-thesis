VENV ?= .venv
PY := $(VENV)/bin/python
PYTHONPATH := src
GNNSRC := src/gnn_implementations
DATASET ?= cora
IMPLS := baseline_multsampler baseline_neo4j baseline_pyg \
	cache_multsampler cache_neo4j distributed
EXPERIMENTS := sampler_comparison
DATASET_TARGETS := cache_multsampler_cora cache_multsampler_arxiv \
	cache_neo4j_cora cache_neo4j_arxiv

.PHONY: run help $(IMPLS) $(DATASET_TARGETS) baseline_db $(EXPERIMENTS)

help:
	@echo "Usage: make <implementation_folder> [DATASET=cora]"
	@echo "       make run SCRIPT=path/to/script.py"

run:
	@if [ -z "$(SCRIPT)" ]; then \
		echo "SCRIPT is required, e.g. make run SCRIPT=src/gnn_implementations/baseline_pyg/cora.py"; \
		exit 1; \
	fi
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(SCRIPT)

$(IMPLS):
	@if [ -d "$(GNNSRC)/$@" ]; then \
		PYTHONPATH=$(PYTHONPATH) $(PY) $(GNNSRC)/$@/$(DATASET).py; \
	else \
		echo "Unknown target '$@'. Use 'make help' for usage."; \
		exit 1; \
	fi

baseline_db:
	@$(MAKE) baseline_neo4j DATASET=$(DATASET)

cache_multsampler_cora:
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(GNNSRC)/cache_multsampler/cora.py

cache_multsampler_arxiv:
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(GNNSRC)/cache_multsampler/arxiv.py

cache_neo4j_cora:
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(GNNSRC)/cache_neo4j/cora.py

cache_neo4j_arxiv:
	@PYTHONPATH=$(PYTHONPATH) $(PY) $(GNNSRC)/cache_neo4j/arxiv.py

sampler_comparison:
	@PYTHONPATH=$(PYTHONPATH) $(PY) src/comparison_experiments/sampler_comparison/run_experiment.py
