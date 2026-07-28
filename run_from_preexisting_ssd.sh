#!/usr/bin/env bash
set -euo pipefail

RAM_TIER="${1:-128}"
INFERENCE_ONLY="${2:-}"
SSD_MOUNT="/mnt/ssd"

# --- Auto-detect 500G SSD partition ---
echo "Auto-detecting 500G SSD..."
BOOT_DISK=$(lsblk -no PKNAME "$(findmnt -n -o SOURCE /)" 2>/dev/null || echo "")
SSD_DEV=""

for disk in /dev/sda1 /dev/sdb1 /dev/nvme0n1p1 /dev/nvme1n1p1; do
    if [[ -b "$disk" ]]; then
        parent=$(basename "$(lsblk -no PKNAME "$disk" 2>/dev/null)")
        # Skip boot disk
        if [[ "$parent" == "$BOOT_DISK" ]]; then
            continue
        fi
        # Check size (accept 400G-600G range)
        size_gb=$(lsblk -bno SIZE "$disk" 2>/dev/null | head -1)
        if [[ -n "$size_gb" ]]; then
            size_gb=$((size_gb / 1024 / 1024 / 1024))
            if [[ $size_gb -ge 400 && $size_gb -le 600 ]]; then
                SSD_DEV="$disk"
                echo "Detected ${size_gb}G SSD partition at ${SSD_DEV}"
                break
            fi
        fi
    fi
done

if [[ -z "$SSD_DEV" ]]; then
    echo "ERROR: No 500G SSD partition found" >&2
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT >&2
    exit 1
fi

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

# --- Update repo ---
echo "Updating repo..."
sudo git pull || echo "WARNING: git pull failed, continuing with existing code"

# Fix terminal compatibility (ghostty not recognized by some tools)
if [[ "$TERM" == "xterm-ghostty" ]]; then
    export TERM=xterm-256color
fi

# Run directly - user should already be in tmux or similar
export DEBIAN_FRONTEND=interactive

EXTRA_ARGS="--skip-phases 4,5 --skip_pyg_training"
if [[ "$INFERENCE_ONLY" == "inference-only" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --inference-only"
fi

exec sudo -E ./run_experiment.sh --ram-tier "$RAM_TIER" $EXTRA_ARGS
