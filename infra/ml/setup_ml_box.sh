#!/bin/bash
set -e

# Verify GVNIC for TIER_1 Networking
# Required for high-speed GDS Arrow data transfer
if lsmod | grep -q gvnic; then
    echo "gVNIC driver detected. TIER_1 networking active."
else
    echo "Warning: gVNIC driver not found. Check machine image."
fi

apt-get update
apt-get install -y git tmux htop build-essential

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/harrywik/neo4j-mpgnn-thesis.git
cd neo4j-mpgnn-thesis

nvidia-smi

echo "ML Box Setup Complete. Ready to connect to Neo4j via GDS Arrow."

