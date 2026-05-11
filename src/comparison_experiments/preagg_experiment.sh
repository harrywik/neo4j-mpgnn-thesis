#!/bin/bash

# Check if at least 3 arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <command> <dataset> <batch_size1> <batch_size2> ..."
    exit 1
fi

COMMAND=$1
shift
DATASET=$2
shift
BATCH_SIZES=("$@")

CONFIG_FILE="src/configs/training/datasets/$DATASET.json"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found."
    exit 1
fi

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "----------------------------------------------------"
    echo "Setting batch_size to $BATCH_SIZE in $CONFIG_FILE"
    echo "----------------------------------------------------"
    
    # Update batch_size in JSON using jq
    # We use a temporary file to safely overwrite the original
    if jq --argjson bs "$BATCH_SIZE" '.batch_size = $bs' "$CONFIG_FILE" > "$CONFIG_FILE.tmp"; then
        mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    else
        echo "Error: Failed to update $CONFIG_FILE with batch_size $BATCH_SIZE"
        exit 1
    fi
    
    echo "Running: make $COMMAND DATASET=$DATASET"
    make "$COMMAND" DATASET=$DATASET
done

echo "Done."
