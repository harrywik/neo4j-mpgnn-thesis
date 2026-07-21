# RAM-Constrained Experiment Plan

## Overview

We're benchmarking ogbn-papers100M (111M nodes, 1.6B edges) across different RAM configurations to understand the memory-performance tradeoff between PyG in-memory and Neo4j-backed GNN training/inference.

**Key insight**: PyG in-memory requires loading the entire graph (~26 GB for edge_index alone + features), while Neo4j can stream from disk. This creates a crossover point where Neo4j becomes viable or even superior as RAM decreases.

## RAM Tiers & GCP Machines

| Tier | RAM | GCP Machine Type | Neo4j Config | Expected Behavior |
|------|-----|------------------|--------------|-------------------|
| **128 GB** | 128 GB | `n2-highmem-16` | pagecache=85g, heap=20g, swap=100g | Both backends work. PyG uses swap during in-memory phases. |
| **96 GB** | 96 GB | `n2-highmem-12` | pagecache=58g, heap=16g, swap=150g | PyG tight, heavy swap usage. Neo4j has less cache. |
| **72 GB** | 72 GB | `n2-highmem-9` | pagecache=40g, heap=14g, swap=200g | PyG likely OOMs without swap. Neo4j slower due to cache misses. |
| **48 GB** | 48 GB | `n2-highmem-6` | pagecache=24g, heap=10g, swap=250g | PyG almost certainly OOMs. Neo4j under pressure. |
| **32 GB** | 32 GB | `n2-highmem-4` | pagecache=12g, heap=6g, swap=250g | Both backends severely constrained. Neo4j page cache can't help much. |

**Common setup for all tiers**:
- 500 GB local SSD at `/dev/sdb` (mounted at `/mnt/ssd`)
- Neo4j data directory on SSD
- Swap file on SSD
- Full dataset ingestion (111M nodes, 1.6B edges)

## Usage

### 1. Create GCP VM

```bash
gcloud compute instances create papers100m-bench-128 \
    --zone=us-central1-a \
    --machine-type=n2-highmem-16 \
    --image-family=debian-13 \
    --image-project=debian-cloud \
    --boot-disk-size=10GB \
    --local-ssd=interface=nvme
```

Replace `n2-highmem-16` with the appropriate machine type for your tier.

### 2. SSH and Clone

```bash
gcloud compute ssh papers100m-bench-128
git clone <repo-url> ~/neo4j-mpgnn-thesis
cd ~/neo4j-mpgnn-thesis
```

**Note:** The boot disk is small (10 GB). The script will automatically move the repo to the SSD during Phase 0. After running the script once, the repo will be at `/mnt/ssd/neo4j-mpgnn-thesis`.

### 3. Run Experiment

```bash
sudo ./run_experiment.sh --ram-tier 128
```

**Important:** The script moves the repo to the SSD on first run. If you need to re-run after the move, use:
```bash
cd /mnt/ssd/neo4j-mpgnn-thesis
sudo ./run_experiment.sh --ram-tier 128 --skip-to 1
```

The script:
- Formats and mounts the SSD
- Creates swap on SSD
- Installs Neo4j Enterprise (30-day eval) via apt
- Configures Neo4j memory for the tier
- Builds and deploys the Java plugin
- Downloads ogbn-papers100M
- Ingests full dataset (nodes + edges)
- Runs 5 repetitions of training + inference for both backends
- Writes results to `experiment_results/gcp_benchmark/results_summary.json`

### 4. Monitor Progress

```bash
tmux attach -t ingest      # Watch ingestion progress
tmux attach -t benchmark   # Watch benchmark progress
```

### 5. Collect Results

```bash
cat experiment_results/gcp_benchmark/results_summary.json
```

## Expected Results by Tier

### 128 GB
- **PyG in-memory**: Works but uses swap during training/inference. Slower than ideal.
- **Neo4j**: Good page cache (85g), reasonable heap. Should perform well.
- **Hypothesis**: Neo4j may outperform PyG due to swap overhead.

### 96 GB
- **PyG in-memory**: Heavy swap usage. Significantly slower.
- **Neo4j**: Reduced page cache (58g) but still viable.
- **Hypothesis**: Neo4j advantage increases.

### 72 GB
- **PyG in-memory**: Likely OOMs during training (edge_index + features + model). Inference may survive with swap.
- **Neo4j**: Page cache (40g) can't cover much of the 200+ GB store. More cache misses.
- **Hypothesis**: PyG training fails. Neo4j inference works but slower than higher tiers.

### 48 GB
- **PyG in-memory**: Almost certainly OOMs. May not even load the graph.
- **Neo4j**: Severely constrained. Page cache (24g) is minimal.
- **Hypothesis**: PyG completely fails. Neo4j is the only option but performance degrades.

### 32 GB
- **PyG in-memory**: Cannot run. OOM during graph loading.
- **Neo4j**: Page cache (12g) is negligible. Heavy reliance on SSD I/O.
- **Hypothesis**: Neo4j is the only viable backend, but performance is poor.

## What We're Measuring

For each tier, we record:
1. **Training time** (1 epoch): PyG vs Neo4j
2. **Inference time** (2048 nodes): PyG vs Neo4j
3. **OOM events**: Which variants fail at which tier?
4. **Memory pressure**: How much swap is used? (visible via `vmstat` or `htop`)

## Analysis

After running all 5 tiers, we can plot:
- **X-axis**: RAM (32, 48, 72, 96, 128 GB)
- **Y-axis**: Time (seconds)
- **Lines**: PyG in-memory vs Neo4j (training and inference separately)

Expected crossover: As RAM decreases, Neo4j becomes relatively faster (or PyG fails entirely).

## Re-running on Same VM

If you want to test a different tier on the same VM (e.g., after changing machine type):

```bash
# Clear old Neo4j data
sudo systemctl stop neo4j
sudo rm -rf /mnt/ssd/neo4j/data/*

# Re-run with new tier
sudo ./run_experiment.sh --ram-tier 96 --skip-to 1
```

This skips Phase 0 (disk already formatted) and re-applies Neo4j config from Phase 1.

## Cleanup

Delete the VM when done:

```bash
gcloud compute instances delete papers100m-bench-128
```
