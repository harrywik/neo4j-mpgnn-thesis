#!/bin/sh
sudo chgrp -R neo4j .
chmod -R g+rwX .
neo4j-admin database import full \
  --input-type=parquet \
  --overwrite-destination \
  --nodes=Paper="data/ogbn-papers100M/papers_data.parquet" \
  --relationships=CITES="data/ogbn-papers100M/citations_data.parquet" \
  --id-type=INTEGER \
  --verbose \
  papers100m
echo "IMPORT DONE"
