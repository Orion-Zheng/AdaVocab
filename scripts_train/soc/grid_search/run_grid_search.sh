#!/bin/bash
# Define the directory containing YAML files
YAML_DIR="scripts_train/soc/new_grid_search/hp_grid/"
TRAIN_SCRIPT="scripts_train/soc/new_grid_search/run_dist_train.sh"
# Iterate over all YAML files in the directory
for yaml_file in "$YAML_DIR"*.yaml; do
    # Check if the current item is a file
    if [ -f "$yaml_file" ]; then
        echo "Running script for YAML file: $yaml_file"
        # Run the script and pass the YAML file path as a parameter
        bash "$TRAIN_SCRIPT" "$yaml_file"
    fi
done
