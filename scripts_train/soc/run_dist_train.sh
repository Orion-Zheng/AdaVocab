#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../.."

# Obtain the default NIC Name
DEFAULT_NIC_NAME=$(ip route show default | grep -Po '(?<=dev )(\S+)')
export GLOO_SOCKET_IFNAME=$DEFAULT_NIC_NAME
export GLOO_LOG_LEVEL=DEBUG
export MASTER_PORT=9999
export MASTER_ADDR=$(ip addr show $DEFAULT_NIC_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
echo "MASTER_ADDR="$MASTER_ADDR

export N_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export NODE_LIST_COMMA=$(echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")" | tr '\n' ',' | sed 's/,$//')

# Create dir to store dist config files of each (MIG) devices
export SAVE_CONFIG_DIR="multi_gpu_output_log"  
if [ ! -d "$SAVE_CONFIG_DIR" ]; then
    mkdir "$SAVE_CONFIG_DIR"
fi

export GPU_PER_NODE=$(nvidia-smi -L | grep MIG | wc -l)  # MIG Devices per node

# Set Default Config
# Check if at least one argument is provided
if [ "$#" -ge 1 ]; then
    # Use the first argument as DEFAULT_CONFIG_FILE
    DEFAULT_CONFIG_FILE="$1"
    source <(bash yaml2env_parser.sh --config $DEFAULT_CONFIG_FILE)
    # Shift to process the remaining arguments
    shift
    # Update Customized Configs
    for CUSTOM_CONFIG_FILE in "$@"; do
        source <(bash yaml2env_parser.sh --config $CUSTOM_CONFIG_FILE)
    done
else
    echo "Usage: $0 DEFAULT_CONFIG_FILE [CUSTOM_CONFIG_FILE...]"
    exit 1
fi

if [ "$GPU_PER_NODE" -eq 0 ]; then
    export GPU_PER_NODE=$(nvidia-smi -L | wc -l)  # GPUs per node
    export WORLD_SIZE=$(($N_NODE * $GPU_PER_NODE))
    DIST_LOG_DIR="${SAVE_CONFIG_DIR}/dist_output_log"
    if [ ! -d "$DIST_LOG_DIR" ]; then
        mkdir "$DIST_LOG_DIR"
    fi
else
    export WORLD_SIZE=$(($N_NODE * $GPU_PER_NODE))
    MIG_LOG_DIR="${SAVE_CONFIG_DIR}/mig_output_log"
    if [ ! -d "$MIG_LOG_DIR" ]; then
        mkdir "$MIG_LOG_DIR"
    fi
fi
echo "GPU_PER_NODE="$GPU_PER_NODE
echo "WORLD_SIZE="$WORLD_SIZE
# Add TimeStamp to Run Name and Output Dir for better tracking
export TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
export WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
export OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"

# Start Multi-Node Training
echo "Number of Nodes: " $N_NODE "; World Size: " $WORLD_SIZE
echo "Main Host Address: "$MAIN_HOST "; NIC Name: " $DEFAULT_NIC_NAME "; Master Port: " $MASTER_PORT
echo "Dispatching Accelerate Configs to All Nodes..."
# =================== Set Dist Config ===================
if [ "$N_NODE" -eq 1 ]; then
  echo "Running on Single Node..."
  bash scripts_train/soc/setup_local_dist_config.sh
else
  echo "Running on Multiple Nodes..."
  srun -N $N_NODE --ntasks-per-node=1 -w $NODE_LIST_COMMA bash scripts_train/soc/setup_local_dist_config.sh
fi

echo "Dispatching Training to All Nodes..."
# =================== Run Dist Training ===================
if [ "$N_NODE" -eq 1 ]; then
  echo "Running on Single Node..."
  bash scripts_train/soc/setup_local_train.sh
else
  echo "Running on Multiple Nodes..."
  srun -N $N_NODE --ntasks-per-node=1 -w $NODE_LIST_COMMA bash scripts_train/soc/setup_local_train.sh
fi
wait