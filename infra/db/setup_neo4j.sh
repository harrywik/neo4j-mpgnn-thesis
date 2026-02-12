#!/bin/bash
set -e

# Add Neo4j 2026 Stable Repository
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add -
# Using 'stable latest' ensures you get the 2026.x calendar version
echo 'deb https://debian.neo4j.com stable latest' | tee /etc/apt/sources.list.d/neo4j.list
apt-get update

# Pre-accept Enterprise License & Install
echo "neo4j-enterprise neo4j/accept-license select Accept commercial license" | debconf-set-selections
apt-get install -y neo4j-enterprise openjdk-21-jdk libcap2-bin mdadm lvm2 git tmux

devices=$(ls /dev/nvme0n*)
count=$(echo $devices | wc -w)

if [ "$count" -gt 0 ]; then
    # Create the RAID array
    mdadm --create /dev/md0 --level=0 --raid-devices=$count $devices
    
    # Format the resulting array
    mkfs.ext4 -F /dev/md0
    
    # Mount it to the Neo4j data directory
    mkdir -p /var/lib/neo4j/data
    mount /dev/md0 /var/lib/neo4j/data
    
    # Ensure it persists across reboots
    echo "/dev/md0 /var/lib/neo4j/data ext4 defaults,nofail 0 2" >> /etc/fstab
fi

# Download Latest Plugins (2026 Versions)
# In 2026, APOC Core is often bundled, but we grab the specific versions for GDS 2.26
cd /var/lib/neo4j/plugins
wget https://github.com/neo4j/graph-data-science/releases/download/2.26.0/neo4j-graph-data-science-2.26.0.jar
wget https://github.com/neo4j/apoc/releases/download/2026.01.3/apoc-2026.01.3-core.jar

# This logic assumes you are scp-ing the file later, but we ensure the directory exists 
# and has the right permissions for the future move.
chown -R neo4j:neo4j /var/lib/neo4j
chmod -R 777 /var/lib/neo4j/plugins

# Metadata-Flavor: Google is a required header for GCE security
CONF_CONTENT=$(curl -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/neo4j_config_content")

# Write the content to the correct file path
echo "$CONF_CONTENT" > /etc/neo4j/neo4j.conf

# Get the internal IP from the metadata server
INTERNAL_IP=$(curl -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)

# Replace the placeholder in the config file
sed -i "s/INTERNAL_IP_PLACEHOLDER/$INTERNAL_IP/g" /etc/neo4j/neo4j.conf

# Secure permissions
chown neo4j:neo4j /etc/neo4j/neo4j.conf
chmod 644 /etc/neo4j/neo4j.conf

# Install uv for python imports
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Initialize a new directory
mkdir papers100m-import && cd papers100m-import
git init

# Add the remote repository
git remote add origin https://github.com/harrywik/neo4j-mpgnn-thesis.git

# Enable sparse checkout
git config core.sparseCheckout true

# Define the subdirectory you want to pull
echo "data/ogbn-papers100M/*" >> .git/info/sparse-checkout
echo "pyproject.toml" >> .git/info/sparse-checkout
echo "uv.lock" >> .git/info/sparse-checkout
echo ".python-version" >> .git/info/sparse-checkout

# Pull the specific branch
git pull origin arrow

# Ensure the script is executable
chmod +x data/ogbn-papers100M/ingest.sh

# Use tmux to manage the ingestion
TERM=xterm tmux new-session -d -s ogb_ingest 'uv run python data/ogbn-papers100M/prepare_import.py && sudo -u neo4j ./data/ogbn-papers100M/ingest.sh; read'

# Enable and Start
systemctl enable neo4j
systemctl start neo4j
