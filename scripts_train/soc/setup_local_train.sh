source $LAUNCH_TEMPLATE

# Deepspeed Usage: https://github.com/huggingface/accelerate/blob/b8c85839531ded28efb77c32e0ad85af2062b27a/docs/source/usage_guides/deepspeed.md?plain=1#L582
MIG_PER_NODE=$(nvidia-smi -L | grep MIG | wc -l)
if [ "$MIG_PER_NODE" -eq 0 ]; then
    # 1) if MIG_DEVICES is not available --> launch local training process with accelerate config `LOCAL_CONFIG_PATH`
    LOCAL_CONFIG_PATH="$SAVE_CONFIG_DIR/$(hostname).yaml"
    LOG_FILE="${SAVE_CONFIG_DIR}/dist_output_log/log_$(hostname).txt"  # Update the LOG file path
    LAUNCH_SCRIPT="accelerate launch --config_file ${LOCAL_CONFIG_PATH} --gradient_accumulation_steps ${GRAD_ACC_STEP} --gradient_clipping ${GRAD_CLIP} --mixed_precision bf16"
    CMD="${LAUNCH_SCRIPT} ${TRAIN_SCRIPT}"
    echo $LOG_FILE
    # echo "$CMD" #> "$LOG_FILE" 
    eval "$CMD" > "$LOG_FILE" 2>&1 &
else
    # 2) if MIG_DEVICES is available --> launch training processes on each MIG device with accelerate config `LOCAL_CONFIG_PATH`
    MIG_DEVICES_LIST=$(nvidia-smi -L | grep -Eo 'MIG-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+-[a-f0-9]+' | sort -u)
    echo "MIG devices are available on $(hostname)"
    index=0
    for UUID in $MIG_DEVICES_LIST
    do
        LOCAL_CONFIG_PATH="$SAVE_CONFIG_DIR/$UUID.yaml"
        LOG_FILE="${SAVE_CONFIG_DIR}/mig_output_log/log_${UUID}.txt"  # Update the LOG file path
        LAUNCH_SCRIPT="accelerate launch --config_file ${LOCAL_CONFIG_PATH} --gradient_accumulation_steps ${GRAD_ACC_STEP} --gradient_clipping ${GRAD_CLIP} --mixed_precision bf16"
        CMD="CUDA_VISIBLE_DEVICES=${UUID} ${LAUNCH_SCRIPT} ${TRAIN_SCRIPT}"
        # echo "$CMD" # > "$LOG_FILE" 
        echo "$LOG_FILE" 
        eval "$CMD" > "$LOG_FILE" 2>&1 &
        index=$((index+1))
    done
fi

wait