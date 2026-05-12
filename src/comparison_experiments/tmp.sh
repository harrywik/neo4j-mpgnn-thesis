#!/bin/bash

./src/comparison_experiments/preagg_experiment.sh java_neo4j products 128 

IMPL_CONFIG="src/configs/training/implementations/java_neo4j.json"
NEIGHBORS="[10, 5]"

if jq --argjson n "$NEIGHBORS" '.sampler.num_neighbors = $n' "$IMPL_CONFIG" > "$IMPL_CONFIG.tmp"; then
    mv "$IMPL_CONFIG.tmp" "$IMPL_CONFIG"
else
    echo "Error: Failed to update $IMPL_CONFIG with num_neighbors $NEIGHBORS"
    exit 1
fi

./src/comparison_experiments/preagg_experiment.sh java_neo4j products 32 64 256 512
