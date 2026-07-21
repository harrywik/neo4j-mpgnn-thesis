#!/usr/bin/env bash
# run_experiment.sh — set up and run the papers100M benchmark on a fresh GCP Debian instance.
#
# Phases:
#   0  System packages, disk setup (SSD), swap, uv
#   1  Neo4j install + configuration (per RAM tier)
#   2  Python environment (uv sync)
#   3  Build & deploy Java plugin
#   4  Download ogbn-papers100M dataset
#   5  Ingest into Neo4j (nodes → edges → float features)
#   6  Run benchmark (training + inference × 5 repetitions)
#
# Usage:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh --ram-tier 128
#   ./run_experiment.sh --ram-tier 32
#   ./run_experiment.sh --ram-tier 64 --skip-to 6
#
# Each long-running phase runs inside a tmux session so SSH disconnects
# don't kill progress.  Attach:  tmux attach -t <phase_name>
set -euo pipefail

# ===========================================================================
# Configuration
# ===========================================================================
NEO4J_VERSION="2025.12.1"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data/ogbn-papers100M"
RESULTS_DIR="${PROJECT_DIR}/experiment_results/gcp_benchmark"
VENV_DIR="${PROJECT_DIR}/.venv"
PY="${VENV_DIR}/bin/python"

# Defaults — overridden by --ram-tier
RAM_TIER=""
SKIP_TO=0
SSD_DEV="/dev/sdb"
SSD_MOUNT="/mnt/ssd"
NEO4J_DATA_DIR=""   # set after SSD mount
NEO4J_RAW_DIR=""    # set after SSD mount
NEO4J_PAGECACHE=""
NEO4J_HEAP=""
SWAP_GB=""

# Neo4j paths (apt-installed)
NEO4J_HOME="/var/lib/neo4j"
NEO4J_CONF="/etc/neo4j/neo4j.conf"
NEO4J_BIN="/usr/bin/neo4j"
NEO4J_ADMIN="/usr/bin/neo4j-admin"
NEO4J_SHELL="/usr/bin/cypher-shell"

# ===========================================================================
# Per-tier memory configuration
#
# Principle: only Neo4j + Python scripts run on this machine.
#   - Reserve 2-3 GB for OS
#   - Maximize Neo4j page cache (reclaimable by OS when Python needs RAM)
#   - Give Neo4j heap enough for GNN stored procedures
#   - Remaining RAM for Python; overflow goes to swap on SSD
#
# Neo4j store after full ingestion: ~200 GB on disk.
# Page cache can never cover it at these tiers — it's a best-effort cache.
# ===========================================================================
declare -A TIER_PAGECACHE TIER_HEAP TIER_SWAP

TIER_PAGECACHE[128]="85g";  TIER_HEAP[128]="20g";  TIER_SWAP[128]=100
TIER_PAGECACHE[96]="58g";   TIER_HEAP[96]="16g";   TIER_SWAP[96]=150
TIER_PAGECACHE[72]="40g";   TIER_HEAP[72]="14g";   TIER_SWAP[72]=200
TIER_PAGECACHE[48]="24g";   TIER_HEAP[48]="10g";   TIER_SWAP[48]=250
TIER_PAGECACHE[32]="12g";   TIER_HEAP[32]="6g";    TIER_SWAP[32]=250

apply_tier() {
    local tier=$1
    if [[ -z "${TIER_PAGECACHE[$tier]+x}" ]]; then
        echo "ERROR: unknown RAM tier '${tier}'. Valid: 128 96 72 48 32" >&2
        exit 1
    fi
    RAM_TIER=$tier
    NEO4J_PAGECACHE="${TIER_PAGECACHE[$tier]}"
    NEO4J_HEAP="${TIER_HEAP[$tier]}"
    SWAP_GB="${TIER_SWAP[$tier]}"
    NEO4J_DATA_DIR="${SSD_MOUNT}/neo4j/data"
    NEO4J_RAW_DIR="${SSD_MOUNT}/neo4j/data/ogbn-papers100M"
}

# ===========================================================================
# Argument parsing
# ===========================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ram-tier)   apply_tier "$2"; shift 2 ;;
        --skip-to)    SKIP_TO="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --ram-tier <128|96|72|48|32> [--skip-to N]"
            echo ""
            echo "RAM tier determines Neo4j memory allocation and swap size:"
            echo "  128  pagecache=85g  heap=20g  swap=100g"
            echo "   96  pagecache=58g  heap=16g  swap=150g"
            echo "   72  pagecache=40g  heap=14g  swap=200g"
            echo "   48  pagecache=24g  heap=10g  swap=250g"
            echo "   32  pagecache=12g  heap= 6g  swap=250g"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$RAM_TIER" ]]; then
    echo "ERROR: --ram-tier is required.  Usage: $0 --ram-tier <128|96|72|48|32>" >&2
    exit 1
fi

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ===========================================================================
# Phase 0: System setup + disk + swap
# ===========================================================================
phase_0_system_setup() {
    log "Phase 0: System setup (RAM tier: ${RAM_TIER} GB)"

    # --- Base packages ---
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq tmux curl wget git openjdk-17-jdk maven procps htop parted

    # --- Format and mount SSD ---
    if mountpoint -q "$SSD_MOUNT" 2>/dev/null; then
        log "SSD already mounted at ${SSD_MOUNT}"
    else
        log "Formatting ${SSD_DEV} as ext4..."
        # Check if already formatted
        if ! blkid "$SSD_DEV" | grep -q TYPE; then
            parted -s "$SSD_DEV" mklabel gpt
            parted -s "$SSD_DEV" mkpart primary ext4 0% 100%
            mkfs.ext4 -F "${SSD_DEV}1"
        fi
        mkdir -p "$SSD_MOUNT"
        mount "${SSD_DEV}1" "$SSD_MOUNT"
        # Persist
        local uuid
        uuid=$(blkid -s UUID -o value "${SSD_DEV}1")
        if ! grep -q "$uuid" /etc/fstab; then
            echo "UUID=${uuid} ${SSD_MOUNT} ext4 defaults,noatime 0 2" >> /etc/fstab
        fi
        log "SSD mounted at ${SSD_MOUNT} ($(df -h "$SSD_MOUNT" | tail -1 | awk '{print $4}') available)"
    fi

    # --- Create directories on SSD ---
    mkdir -p "${SSD_MOUNT}/neo4j/data"
    mkdir -p "${SSD_MOUNT}/swap"

    # --- Create swap on SSD ---
    local swap_file="${SSD_MOUNT}/swap/swapfile"
    local current_swap
    current_swap=$(free -g | awk '/Swap:/{print $2}')
    if [[ "$current_swap" -ge "$SWAP_GB" ]]; then
        log "Swap already sufficient: ${current_swap}GB"
    else
        log "Creating ${SWAP_GB}GB swap file on SSD..."
        # Remove old swapfile if it exists but is wrong size
        [[ -f "$swap_file" ]] && swapoff "$swap_file" 2>/dev/null || true
        fallocate -l "${SWAP_GB}G" "$swap_file"
        chmod 600 "$swap_file"
        mkswap "$swap_file"
        swapon "$swap_file"
        if ! grep -q "$swap_file" /etc/fstab; then
            echo "${swap_file} none swap sw 0 0" >> /etc/fstab
        fi
        log "Swap active: ${SWAP_GB}GB on ${swap_file}"
    fi

    # --- Install uv ---
    if ! command -v uv &>/dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        log "uv installed"
    else
        log "uv already installed: $(uv --version)"
    fi

    # --- Kernel tuning ---
    # Low swappiness: prefer dropping page cache over swapping
    sysctl -w vm.swappiness=10 2>/dev/null || true
    sysctl -w vm.max_map_count=262144 2>/dev/null || true
    # Allow more dirty pages before blocking (helps SSD write throughput)
    sysctl -w vm.dirty_ratio=20 2>/dev/null || true
    sysctl -w vm.dirty_background_ratio=5 2>/dev/null || true

    log "Phase 0 complete — swap=${SWAP_GB}GB, SSD=$(df -h "$SSD_MOUNT" | tail -1 | awk '{print $4}') free"
}

# ===========================================================================
# Phase 1: Neo4j install + configuration
# ===========================================================================
phase_1_neo4j_setup() {
    log "Phase 1: Neo4j ${NEO4J_VERSION} setup (tier=${RAM_TIER}G, heap=${NEO4J_HEAP}, pagecache=${NEO4J_PAGECACHE})"

    # --- Install Neo4j via apt ---
    if command -v neo4j &>/dev/null; then
        log "Neo4j already installed: $(neo4j version 2>/dev/null || echo 'unknown version')"
    else
        log "Installing Neo4j Enterprise via apt..."

        # Add Neo4j GPG key
        wget -qO - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -

        # Add Neo4j repository (Enterprise requires license acceptance)
        echo "deb https://debian.neo4j.com stable ${NEO4J_VERSION%.*}" > /etc/apt/sources.list.d/neo4j.list

        # Accept license non-interactively
        export DEBIAN_FRONTEND=noninteractive
        echo "neo4j-enterprise neo4j/license note" | debconf-set-selections
        echo "neo4j-enterprise neo4j/license boolean true" | debconf-set-selections

        apt-get update -qq
        apt-get install -y -qq neo4j-enterprise

        log "Neo4j installed via apt"
    fi

    # --- Configure neo4j.conf ---
    log "Configuring neo4j.conf (heap=${NEO4J_HEAP}, pagecache=${NEO4J_PAGECACHE})..."

    # Backup original config
    [[ -f "${NEO4J_CONF}.bak" ]] || cp "${NEO4J_CONF}" "${NEO4J_CONF}.bak"

    # Memory settings
    sed -i "s|^#* *server.memory.heap.initial_size=.*|server.memory.heap.initial_size=${NEO4J_HEAP}|" "$NEO4J_CONF"
    sed -i "s|^#* *server.memory.heap.max_size=.*|server.memory.heap.max_size=${NEO4J_HEAP}|" "$NEO4J_CONF"
    sed -i "s|^#* *server.memory.pagecache.size=.*|server.memory.pagecache.size=${NEO4J_PAGECACHE}|" "$NEO4J_CONF"

    # Data directory on SSD
    sed -i "s|^#* *server.directories.data=.*|server.directories.data=${NEO4J_DATA_DIR}|" "$NEO4J_CONF"

    # Plugin security
    sed -i "s|^#* *dbms.security.procedures.unrestricted=.*|dbms.security.procedures.unrestricted=gnnProcedures.*|" "$NEO4J_CONF"
    sed -i "s|^#* *dbms.security.procedures.allowlist=.*|dbms.security.procedures.allowlist=gnnProcedures.*|" "$NEO4J_CONF"

    # Transaction timeout (large ingestion needs long transactions)
    sed -i "s|^#* *dbms.transaction.timeout=.*|dbms.transaction.timeout=3600s|" "$NEO4J_CONF"

    # --- Set initial password ---
    neo4j-admin dbms set-initial-password "benchmark2026" 2>/dev/null || true

    # --- Create directories on SSD ---
    mkdir -p "${NEO4J_DATA_DIR}"
    mkdir -p "${NEO4J_HOME}/plugins"
    mkdir -p "${NEO4J_HOME}/gnn_models"
    chown -R neo4j:neo4j "${NEO4J_DATA_DIR}" "${NEO4J_HOME}/plugins" "${NEO4J_HOME}/gnn_models"

    # --- Start Neo4j ---
    systemctl enable neo4j
    systemctl start neo4j

    log "Waiting for Neo4j to be ready..."
    local max_wait=180
    local waited=0
    while ! cypher-shell -u neo4j -p benchmark2026 "RETURN 1" &>/dev/null; do
        sleep 3
        waited=$((waited + 3))
        if [[ $waited -ge $max_wait ]]; then
            log "ERROR: Neo4j did not start within ${max_wait}s"
            log "Check logs: journalctl -u neo4j -n 50"
            exit 1
        fi
    done
    log "Neo4j is ready"

    # Create database
    cypher-shell -u neo4j -p benchmark2026 \
        "CREATE DATABASE papers100m IF NOT EXISTS" 2>/dev/null || true
    log "Database 'papers100m' created"

    # Write .env
    cat > "${PROJECT_DIR}/.env" <<EOF
URI=bolt://localhost:7687
USERNAME=neo4j
PASSWORD=benchmark2026
NEO4J_PLUGINS_DIR=${NEO4J_HOME}/plugins
NEO4J_GNN_MODEL_DIR=${NEO4J_HOME}/gnn_models
EOF
    log ".env written"
    log "Phase 1 complete"
}

# ===========================================================================
# Phase 2: Python environment
# ===========================================================================
phase_2_python_env() {
    log "Phase 2: Python environment"
    cd "$PROJECT_DIR"
    uv sync
    log "Phase 2 complete"
}

# ===========================================================================
# Phase 3: Build & deploy Java plugin
# ===========================================================================
phase_3_build_plugin() {
    log "Phase 3: Building Neo4j GNN plugin"
    cd "$PROJECT_DIR"

    (cd neo4j-gcn-plugin && mvn clean package -q -Dneo4j.version="${NEO4J_VERSION}")
    cp neo4j-gcn-plugin/target/neo4j-gcn-plugin-1.0.0.jar "${NEO4J_HOME}/plugins/"
    chown neo4j:neo4j "${NEO4J_HOME}/plugins/neo4j-gcn-plugin-1.0.0.jar"
    log "Plugin deployed"

    # Restart Neo4j
    log "Restarting Neo4j..."
    systemctl restart neo4j

    log "Waiting for Neo4j restart..."
    local waited=0
    while ! cypher-shell -u neo4j -p benchmark2026 "RETURN 1" &>/dev/null; do
        sleep 3
        waited=$((waited + 3))
        if [[ $waited -ge 180 ]]; then
            log "ERROR: Neo4j did not restart"; exit 1
        fi
    done
    log "Neo4j restarted with plugin"

    local proc_count
    proc_count=$(cypher-shell -u neo4j -p benchmark2026 -d papers100m \
        "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'gnnProcedures.' RETURN count(*) AS c" 2>/dev/null || echo "0")
    log "Plugin procedures registered: ${proc_count}"
    log "Phase 3 complete"
}

# ===========================================================================
# Phase 4: Download dataset
# ===========================================================================
phase_4_download_dataset() {
    log "Phase 4: Downloading ogbn-papers100M"
    mkdir -p "$DATA_DIR"

    if [[ -f "${DATA_DIR}/ogbn_papers100M/raw/data.npz" ]]; then
        log "Dataset already downloaded"
    else
        log "Downloading via OGB (this may take a while)..."
        cd "$PROJECT_DIR"
        PYTHONPATH=src "${PY}" -c "
from ogb.nodeproppred import NodePropPredDataset
import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, 'weights_only': False})
dataset = NodePropPredDataset(name='ogbn-papers100M', root='data/ogbn-papers100M')
print(f'Downloaded: {dataset[0][0][\"num_nodes\"]} nodes')
"
    fi

    # Symlink to Neo4j data dir on SSD for the ingest scripts
    if [[ ! -d "$NEO4J_RAW_DIR" ]]; then
        mkdir -p "$(dirname "$NEO4J_RAW_DIR")"
        ln -sf "${DATA_DIR}/ogbn_papers100M" "$NEO4J_RAW_DIR"
        log "Symlinked dataset to ${NEO4J_RAW_DIR}"
    fi

    log "Phase 4 complete"
}

# ===========================================================================
# Phase 5: Ingest into Neo4j
# ===========================================================================
phase_5_ingest() {
    log "Phase 5: Ingesting ogbn-papers100M into Neo4j"
    cd "$PROJECT_DIR"
    source .env
    export URI USERNAME PASSWORD

    log "Full ingestion: 111M nodes + 1.6B edges (this will take hours)"
    tmux new-session -d -s ingest \
        "cd ${PROJECT_DIR} && PYTHONPATH=src ${PY} data/ogbn-papers100M/ingest.py 2>&1 | tee ${RESULTS_DIR}/ingest.log"

    log "Ingestion started in tmux session 'ingest' — monitor: tmux attach -t ingest"
    while tmux has-session -t ingest 2>/dev/null; do sleep 30; done
    log "Node + edge ingestion complete"
    log "Phase 5 complete"
}

# ===========================================================================
# Phase 6: Run benchmark
# ===========================================================================
phase_6_benchmark() {
    log "Phase 6: Running benchmark"
    cd "$PROJECT_DIR"
    source .env
    export URI USERNAME PASSWORD NEO4J_PLUGINS_DIR NEO4J_GNN_MODEL_DIR
    mkdir -p "$RESULTS_DIR"

    local bench_cmd="PYTHONPATH=src ${PY} run_benchmark.py --results_dir ${RESULTS_DIR} --n_runs 5 --n_nodes 2048"

    tmux new-session -d -s benchmark "cd ${PROJECT_DIR} && ${bench_cmd} 2>&1 | tee ${RESULTS_DIR}/benchmark.log"
    log "Benchmark started in tmux session 'benchmark' — monitor: tmux attach -t benchmark"
    while tmux has-session -t benchmark 2>/dev/null; do sleep 30; done

    log "Phase 6 complete — Results: ${RESULTS_DIR}/results_summary.json"

    if [[ -f "${RESULTS_DIR}/results_summary.json" ]]; then
        echo ""
        echo "============================================"
        echo "  FINAL RESULTS  (tier=${RAM_TIER}G)"
        echo "============================================"
        "${PY}" -c "
import json
with open('${RESULTS_DIR}/results_summary.json') as f:
    data = json.load(f)
for section in ('training', 'inference'):
    print(f'\n  {section.upper()}:')
    for variant, vdata in data[section].items():
        s = vdata.get('summary', {})
        status = s.get('status', '')
        if status in ('OOM', 'ERROR', 'SKIPPED'):
            print(f'    {variant}: {status}')
        else:
            m = s.get('mean', 0)
            sd = s.get('std', 0)
            n = s.get('n_ok', 0)
            oom = s.get('n_oom', 0)
            line = f'    {variant}: {m:.4f}s +/- {sd:.4f}s (n={n}'
            if oom: line += f', {oom} OOM'
            line += ')'
            print(line)
"
    fi
}

# ===========================================================================
# Main
# ===========================================================================
log "=============================================="
log "  papers100M Benchmark on GCP"
log "  RAM tier:    ${RAM_TIER} GB"
log "  Neo4j:       ${NEO4J_VERSION}"
log "  pagecache:   ${NEO4J_PAGECACHE}"
log "  heap:        ${NEO4J_HEAP}"
log "  swap:        ${SWAP_GB} GB (on SSD)"
log "  SSD:         ${SSD_DEV} → ${SSD_MOUNT}"
log "=============================================="

phases=(phase_0_system_setup phase_1_neo4j_setup phase_2_python_env phase_3_build_plugin phase_4_download_dataset phase_5_ingest phase_6_benchmark)

for i in "${!phases[@]}"; do
    if [[ $i -ge $SKIP_TO ]]; then
        ${phases[$i]}
    fi
done

log "All phases complete!"
