#!/bin/sh
echo "DOWNLOAD BEGIN"
uv run python data/ogbn-papers100M/prepare_import.py
echo "PARQUET WRITTEN"

neo4j-admin database import full \
  --input-type=parquet \
  --overwrite-destination \
  --nodes=Paper="data/ogbn-papers100M/papers_header.csv,data/ogbn-papers100M/papers_data.parquet" \
  --relationships=CITES="data/ogbn-papers100M/citations_header.csv,data/ogbn-papers100M/citations_data.parquet" \
  --id-type=INTEGER \
  --verbose \
  papers100m
echo "IMPORT DONE"
