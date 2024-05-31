# Set local accelerate config(s) to SAVE_CONFIG_DIR
# Require input environment variables:
# N_NODE, GPU_PER_NODE, MASTER_ADDR, ACCELERATE_CONFIG, SAVE_CONFIG_DIR
# SLURM_NODEID (from Slurm)
MIG_PER_NODE=$(nvidia-smi -L | grep MIG | wc -l)
if [ "$MIG_PER_NODE" -eq 0 ]; then
    # 1) if MIG_DEVICES is not available --> save config to dist_config_dir
    echo "No MIG devices are available on $(hostname)"
    GLOBAL_RANK=$((SLURM_NODEID))
    WORLD_SIZE=$(($N_NODE * $GPU_PER_NODE))
    python codebase/tools/dist_env/set_dist_config_new.py \
                        --config_template ${ACCELERATE_CONFIG} \
                        --global_rank ${GLOBAL_RANK} \
                        --main_proc_ip ${MASTER_ADDR} \
                        --main_proc_port ${MASTER_PORT} \
                        --n_node ${N_NODE} \
                        --world_size ${WORLD_SIZE} \
                        --local_save_path ${SAVE_CONFIG_DIR}"/"$(hostname)".yaml"
else
    # 2) if MIG_DEVICES is available --> save configs of each devices to dist_config_dir
    MIG_DEVICES_LIST=$(nvidia-smi -L | grep -Eo 'MIG-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+' | sort -u)
    echo "MIG devices are available on $(hostname)"
    index=0
    for UUID in $MIG_DEVICES_LIST
    do
        GLOBAL_RANK=$(($SLURM_NODEID * $MIG_PER_NODE + index))
        WORLD_SIZE=$(($N_NODE * $MIG_PER_NODE))
        python codebase/tools/dist_env/set_dist_config_new.py \
                        --config_template ${ACCELERATE_CONFIG} \
                        --global_rank ${GLOBAL_RANK} \
                        --main_proc_ip ${MASTER_ADDR} \
                        --main_proc_port ${MASTER_PORT} \
                        --n_node ${WORLD_SIZE} \
                        --world_size ${WORLD_SIZE} \
                        --local_save_path ${SAVE_CONFIG_DIR}"/"$UUID".yaml"
        index=$((index+1))
    done
fi






