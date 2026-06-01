#!/bin/bash

# Check if .env file exists
if [ -f .env ]; then
    # export all variables defined in the .env file
    set -a
    source .env
    set +a
else
    echo ".env file not found!"
    exit 1
fi

DATASETS=("arxiv2" "products" "papers100m")

# Ensure the temp directory exists and is writable by neo4j
mkdir -p /var/lib/neo4j/data/tmp
sudo chown -R neo4j:neo4j /var/lib/neo4j/data/tmp
cd /var/lib/neo4j/data/tmp

for DS in "${DATASETS[@]}"; do
    echo "--------------------------------------------"
    echo "Starting compaction process for database: $DS"
    echo "--------------------------------------------"

    # Stop the database synchronously
    echo "Stopping database $DS and waiting for locks to clear..."
    cypher-shell -u "$USERNAME" -p "$PASSWORD" -d system "STOP DATABASE $DS WAIT 600 SECONDS;"

    # Run the compaction as the neo4j user
    echo "Running store compaction..."
    sudo -u neo4j neo4j-admin database copy "$DS" "$DS" --compact-node-store --temp-path=/var/lib/neo4j/data/tmp

    # Start the database back up synchronously
    echo "Starting database $DS..."
    cypher-shell -u "$USERNAME" -p "$PASSWORD" -d system "START DATABASE $DS WAIT 600 SECONDS;"

    # Apply the schema file
    # We check for both exact case and lowercase filenames just in case Neo4j normalized it
    DS_LOWER=$(echo "$DS" | tr '[:upper:]' '[:lower:]')
    SCHEMA_FILE=""

    if [ -f "${DS}-schema.cypher" ]; then
        SCHEMA_FILE="${DS}-schema.cypher"
    elif [ -f "${DS_LOWER}-schema.cypher" ]; then
        SCHEMA_FILE="${DS_LOWER}-schema.cypher"
    fi

    if [ -n "$SCHEMA_FILE" ]; then
        echo "Applying schema and recreating indexes using $SCHEMA_FILE..."
        cypher-shell -u "$USERNAME" -p "$PASSWORD" -d "$DS" -f "$SCHEMA_FILE"
        rm "$SCHEMA_FILE"
    else
        echo "Warning: Schema file for $DS not found. Skipping index recreation."
    fi

done

echo "All database compactions completed successfully!"
