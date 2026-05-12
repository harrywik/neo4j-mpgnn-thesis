#!/bin/bash


IMPL1=preagg_neo4j
IMPL2=java_neo4j
DATASET=citeseer

IMPL1_CONFIG="src/configs/training/implementations/$IMPL1.json"
IMPL2_CONFIG="src/configs/training/implementations/$IMPL2.json"
NEIGHBORS="[10, 5]"

if jq --argjson n "$NEIGHBORS" '.sampler.num_neighbors = $n' "$IMPL1_CONFIG" > "$IMPL1_CONFIG.tmp"; then
    mv "$IMPL1_CONFIG.tmp" "$IMPL1_CONFIG"
else
    echo "Error: Failed to update $IMPL1_CONFIG with num_neighbors $NEIGHBORS"
    exit 1
fi

if jq --argjson n "$NEIGHBORS" '.sampler.num_neighbors = $n' "$IMPL2_CONFIG" > "$IMPL2_CONFIG.tmp"; then
    mv "$IMPL2_CONFIG.tmp" "$IMPL2_CONFIG"
else
    echo "Error: Failed to update $IMPL2_CONFIG with num_neighbors $NEIGHBORS"
    exit 1
fi

./src/comparison_experiments/preagg_experiment.sh $IMPL1 $DATASET 32 64 128 256 512 
./src/comparison_experiments/preagg_experiment.sh $IMPL2 $DATASET 32 64 128 256 512

NEIGHBORS="[20, 10]"

if jq --argjson n "$NEIGHBORS" '.sampler.num_neighbors = $n' "$IMPL1_CONFIG" > "$IMPL1_CONFIG.tmp"; then
    mv "$IMPL1_CONFIG.tmp" "$IMPL1_CONFIG"
else
    echo "Error: Failed to update $IMPL1_CONFIG with num_neighbors $NEIGHBORS"
    exit 1
fi

if jq --argjson n "$NEIGHBORS" '.sampler.num_neighbors = $n' "$IMPL2_CONFIG" > "$IMPL2_CONFIG.tmp"; then
    mv "$IMPL2_CONFIG.tmp" "$IMPL2_CONFIG"
else
    echo "Error: Failed to update $IMPL2_CONFIG with num_neighbors $NEIGHBORS"
    exit 1
fi

./src/comparison_experiments/preagg_experiment.sh $IMPL1 $DATASET 32 64 128 256 512 
./src/comparison_experiments/preagg_experiment.sh $IMPL2 $DATASET 32 64 128 256 512
