#!/bin/bash
scp ./gds.license thesis-vm:/tmp/gds.license

echo "Moving license and restarting Neo4j..."
ssh thesis-vm << 'EOF'
  sudo mv /tmp/gds.license /etc/neo4j/gds.license
  sudo chown neo4j:neo4j /etc/neo4j/gds.license
  sudo chmod 400 /etc/neo4j/gds.license
  sudo systemctl restart neo4j
EOF

echo "Neo4j is restarting with GDS Enterprise enabled."
