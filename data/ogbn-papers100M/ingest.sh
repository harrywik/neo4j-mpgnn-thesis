#!/bin/sh
neo4j-admin database import full \
  --input-type=parquet \
  --overwrite-destination \
  --nodes=Paper="import/papers_header.csv,import/papers_data.parquet" \
  --relationships=CITES="import/citations_header.csv,import/citations_data.parquet" \
  --id-type=INTEGER \
  --verbose \
  papers100m
