#!/bin/sh
chgrp -R neo4j .
chmod -R g+rwX .
neo4j-admin database import full \
  --input-type=parquet \
  --overwrite-destination \
  --nodes=Paper="/var/lib/neo4j/data/import-workspace/papers_nodes.parquet" \
  --relationships=CITES="/var/lib/neo4j/data/import-workspace/citations_data.parquet" \
  --id-type=INTEGER \
  --verbose \
  papers100m
echo "IMPORT DONE"
