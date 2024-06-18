#!/bin/bash
#SBATCH -J SFT_full       # Job name
#SBATCH -p gpu-long          # Queue (partition) name
#SBATCH -N 1              # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 120:00:00        # Run time (hh:mm:ss)
#SBATCH --mem=784GB     # min mem per node
#SBATCH --gpus-per-node=h100-96:2
#SBATCH --mail-user=zheng_zian@u.nus.edu
#SBATCH --mail-type=all    # Send email at begin and end of job
source ~/.bashrc
TRAIN_SCRIPT="scripts_train/soc/run_train.sh"  
BASIC_CONFIG="scripts_train/soc/experiments_config/ada_qwen2-1.5b/ada_qwen2_ft_dist_conifg_2_h100_96.yaml"
# TRAIN_SCRIPT="scripts_train/soc/run_dist_train.sh"  # for MIG training
# BASIC_CONFIG="scripts_train/soc/experiments_config/ada_qwen2-1.5b/ada_qwen2_ft_dist_conifg_4_h100_47.yaml"

# ========== Single Test ==========
# TEST_CONFIG="scripts_train/soc/experiments_config/ada_qwen2-1.5b/hp_grid/lora_dim_192.yaml"
# bash "$TRAIN_SCRIPT" "$BASIC_CONFIG" "$TEST_CONFIG"

# ========== HyperParmeter Grid Search ==========
# Define the directory containing YAML files
YAML_DIR="scripts_train/soc/experiments_config/ada_qwen2-1.5b/hp_grid/"  # DIR must end with "/"
# Iterate over all YAML files in the directory
for TEST_CONFIG in "$YAML_DIR"*.yaml; do
    # Check if the current item is a file
    if [ -f "$TEST_CONFIG" ]; then
        echo "Running script for YAML file: $TEST_CONFIG"
        # Run the script and pass the YAML file path as a parameter
        bash "$TRAIN_SCRIPT" "$BASIC_CONFIG" "$TEST_CONFIG"
    fi
done