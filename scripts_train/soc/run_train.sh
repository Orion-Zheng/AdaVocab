#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../.."

export N_NODE=1

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

export GPU_PER_NODE=$(nvidia-smi -L | wc -l)  # GPUs per node
export WORLD_SIZE=$(($N_NODE * $GPU_PER_NODE))
DIST_LOG_DIR="${SAVE_CONFIG_DIR}/dist_output_log"
if [ ! -d "$DIST_LOG_DIR" ]; then
    mkdir "$DIST_LOG_DIR"
fi

echo "GPU_PER_NODE="$GPU_PER_NODE
echo "WORLD_SIZE="$WORLD_SIZE
# Add TimeStamp to Run Name and Output Dir for better tracking
export TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
export WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
export OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"

# Start Single-Node Training
echo "Number of Nodes: " $N_NODE "; World Size: " $WORLD_SIZE
# =================== Set Dist Config ===================

echo "Running on Single Node..."
python codebase/tools/dist_env/set_dist_config_new.py \
                        --config_template ${ACCELERATE_CONFIG} \
                        --global_rank 0 \
                        --n_node ${N_NODE} \
                        --world_size ${WORLD_SIZE} \
                        --local_save_path ${SAVE_CONFIG_DIR}"/"$(hostname)".yaml"

# =================== Run Dist Training ===================
echo "Running on Single Node..."
source $LAUNCH_TEMPLATE
LOCAL_CONFIG_PATH="$SAVE_CONFIG_DIR/$(hostname).yaml"
LOG_FILE="${SAVE_CONFIG_DIR}/dist_output_log/log_$(hostname).txt"  # Update the LOG file path
LAUNCH_SCRIPT="accelerate launch --config_file ${LOCAL_CONFIG_PATH} --gradient_accumulation_steps ${GRAD_ACC_STEP} --gradient_clipping ${GRAD_CLIP} --mixed_precision bf16"
CMD="${LAUNCH_SCRIPT} ${TRAIN_SCRIPT}"
echo $LOG_FILE
# echo "$CMD" #> "$LOG_FILE" 
eval "$CMD" > "$LOG_FILE" 2>&1 &
wait