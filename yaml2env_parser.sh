#!/bin/bash
# Override parameters with command line arguments if provided
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
# echo "Config file: $CONFIG_FILE"

# Function to read parameters from config file
read_config_parameters() {
  local section=$1
  local config_file=$2
  local section_parameters=$(cat "$config_file" | yq ".${section}")
  echo "$section_parameters"  # return value
}

# Iterate over categories and export parameters as environment variables
categories=("dist_arguments" "wandb_arguments" "model_arguments" "data_arguments" "output_arguments" "train_arguments"  "ada_arguments" "train_script") 

for category in "${categories[@]}"; do
    # Read parameters from config file
    parameters=$(read_config_parameters "$category" "$CONFIG_FILE")
    # echo "$parameters"
    # Check if parameters are `null`(string), if so, skip
    if [ "$parameters" = "null" ]; then
        # echo "Argument for category '$category' are null. Skipping..."
        continue
    fi

    # Export parameters as environment variables
    echo "$parameters" | jq -r 'to_entries[] | "export \(.key)=\(.value)"' | while read -r line; do
        echo "$line"
        # eval "$line"
    done
done