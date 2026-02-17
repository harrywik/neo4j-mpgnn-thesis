#!/bin/sh

chgrp -R neo4j .
chmod -R g+rwX .

# JAVA_OPTS provides redundancy for the JVM
export JAVA_OPTS="-Xmx64G -Xms64G"

# Create a dedicated config directory to force the Page Cache
IMPORT_CONF_DIR="/tmp/neo4j-import-config"
mkdir -p $IMPORT_CONF_DIR
echo "server.memory.pagecache.size=80G" > "$IMPORT_CONF_DIR/neo4j.conf"
# Add the legacy namespace just in case
echo "dbms.memory.pagecache.size=80G" >> "$IMPORT_CONF_DIR/neo4j.conf"
# Set store format
echo "db.format=block" >> "$IMPORT_CONF_DIR/neo4j.conf"

neo4j-admin database import full \
        --input-type=parquet \
        --threads=8 \
        --additional-config="$IMPORT_CONF_DIR/neo4j.conf" \
        --overwrite-destination \
        --nodes=Paper="/var/lib/neo4j/data/import-workspace/papers_nodes.parquet" \
        --relationships=CITES="/var/lib/neo4j/data/import-workspace/citations_data.parquet" \
        --normalize-types=false \
        --id-type=actual \
        --verbose \
        papers100m

echo "IMPORT DONE"
