#!/usr/bin/env bash
set -euo pipefail

RAM_TIER="${1:-128}"
SSD_DEV="/dev/sdb1"
SSD_MOUNT="/mnt/ssd"

# --- Mount SSD (already formatted, data intact) ---
if mountpoint -q "$SSD_MOUNT" 2>/dev/null; then
    echo "SSD already mounted at $SSD_MOUNT"
else
    sudo mkdir -p "$SSD_MOUNT"
    sudo mount "$SSD_DEV" "$SSD_MOUNT"
    UUID=$(sudo blkid -s UUID -o value "$SSD_DEV")
    if ! grep -q "$UUID" /etc/fstab; then
        echo "UUID=${UUID} ${SSD_MOUNT} ext4 defaults,noatime 0 2" | sudo tee -a /etc/fstab
    fi
    echo "SSD mounted at $SSD_MOUNT"
fi

df -h "$SSD_MOUNT"

# --- Configure neo4j.conf if needed ---
if ! grep -q 'NEO4J_GNN_MODEL_DIR' /etc/neo4j/neo4j.conf 2>/dev/null; then
    echo 'server.jvm.additional=-DNEO4J_GNN_MODEL_DIR=/var/lib/neo4j/gnn_models' | sudo tee -a /etc/neo4j/neo4j.conf
    echo "Added NEO4J_GNN_MODEL_DIR to neo4j.conf"
fi

# --- Run benchmark ---
cd "${SSD_MOUNT}/neo4j-mpgnn-thesis"
sudo ./run_experiment.sh --ram-tier "$RAM_TIER" --skip-to 6 --skip_pyg_training
