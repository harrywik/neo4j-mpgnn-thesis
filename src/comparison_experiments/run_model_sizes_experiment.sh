#!/bin/bash
set -e

# This script implements the following plan:
# 1. Train models on byte[] features (fast) for 1 epoch.
# 2. Add float64[] feature representations to Neo4j.
# 3. Update configs to use float64[] and longer training for future runs.
# 4. Run full inference benchmarking suite.

DATASETS=("arxiv" "papers100M" "products")

# Ensure we are in the project root
cd "$(git rev-parse --show-toplevel)"

for DS in "${DATASETS[@]}"; do
    echo "========================================================================"
    echo "Processing dataset: $DS"
    echo "========================================================================"

    # make baseline_neo4j DATASET=$DS
    make baseline_neo4j DATASET=$DS
    make java_neo4j DATASET=$DS
    make preagg_neo4j DATASET=$DS


    # --- Add Float Features ---
    echo "Adding float features to Neo4j..."
    make add_float_features_$DS


    # --- Update Configs ---
    echo "Updating configurations..."
    
    # Update training config: use floats, longer training, batch size 128
    jq '.feature_property = "feature_vector_floats" | 
        .feature_property_type = "float64[]" | 
        .max_epochs = 16 | 
        .patience = 4 | 
        .batch_size = 128 |
        .max_validation_size = 0 |
        .max_test_size = 0' \
       src/configs/training/datasets/$DS.json > src/configs/training/datasets/$DS.tmp.json
    mv src/configs/training/datasets/$DS.tmp.json src/configs/training/datasets/$DS.json

    # Ensure inference config points to the new float features
    jq '.feature_property = "feature_vector_floats" | .feature_property_type = "float64[]"' \
       src/configs/inference/datasets/$DS.json > src/configs/inference/datasets/$DS.tmp.json
    mv src/configs/inference/datasets/$DS.tmp.json src/configs/inference/datasets/$DS.json


    # --- Run Inference Experiment ---
    echo "Running full inference benchmarking..."
    
    make inference_experiment INFERENCE_DATASET=$DS INFERENCE_MODEL=gcn_$DS
done

echo "All datasets processed successfully."
