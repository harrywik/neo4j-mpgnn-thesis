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

# --- Run setup + benchmark ---
# SSD has data+repo already, so skip download (4) and ingest (5).
# Phase 0 installs base packages (gnupg, tmux, etc.) and mounts SSD.
# Phases 1-3 install Neo4j, Python env, and build the plugin.
cd "${SSD_MOUNT}/neo4j-mpgnn-thesis"

# Fix terminal compatibility for tmux (ghostty not recognized)
if [[ "$TERM" == "xterm-ghostty" ]]; then
    export TERM=xterm-256color
fi

# If already in tmux, run directly; otherwise create a new session
if [[ -n "${TMUX:-}" ]]; then
    echo "Already in tmux, running directly..."
    export DEBIAN_FRONTEND=interactive
    sudo -E ./run_experiment.sh --ram-tier "$RAM_TIER" --skip-phases 4,5 --skip_pyg_training
else
    echo "Starting tmux session for interactive installation..."
    tmux new-session -d -s setup "cd ${SSD_MOUNT}/neo4j-mpgnn-thesis && sudo -E ./run_experiment.sh --ram-tier $RAM_TIER --skip-phases 4,5 --skip_pyg_training"
    echo "Attaching to tmux session 'setup'..."
    tmux attach -t setup
fi
