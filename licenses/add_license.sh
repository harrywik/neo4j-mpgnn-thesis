#!/bin/bash

VM_IP=$(terraform output -raw external_ip)

if [ -z "$VM_IP" ]; then
    echo "Error: Could not fetch VM IP. Did you run terraform apply?"
    exit 1
fi

echo "Connecting to Neo4j at $VM_IP..."

echo "Copying GDS license to $VM_IP..."

scp ./gds.license ${USER}@${VM_IP}:/tmp/gds.license

echo "Moving license and restarting Neo4j..."
ssh -i ~/.ssh/id_rsa ${USER}@${VM_IP} << 'EOF'
  sudo mv /tmp/gds.license /etc/neo4j/gds.license
  sudo chown neo4j:neo4j /etc/neo4j/gds.license
  sudo chmod 400 /etc/neo4j/gds.license
  sudo systemctl restart neo4j
EOF

echo "Neo4j is restarting with GDS Enterprise enabled."
