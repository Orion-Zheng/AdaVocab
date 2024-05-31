TRAIN_SCRIPT="train.py \
              --run_name ${WANDB_RUN_NAME} \
              --model_dir ${MODEL_DIR} \
              --tokenizer_dir ${TOKENIZER_DIR} \
              --train_data_dir ${TRAIN_DATA_DIR} \
              --eval_data_dir ${EVAL_DATA_DIR} \
              --output_dir ${OUTPUT_DIR} \
              --gradient_accumulation_steps ${GRAD_ACC_STEP} \
              --per_device_eval_batch_size ${PER_DEVICE_EVAL_BATCH_SIZE} \
              --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
              --max_token_per_seq ${MAX_TOKEN_PER_SEQ} \
              --eval_steps ${EVAL_STEPS} \
              --save_steps ${SAVE_STEPS} \
              --learning_rate ${LEARNING_RATE} \
              --optim ${OPTIMIZER} --adam_beta1 ${ADAM_BETA1} --adam_beta2 ${ADAM_BETA2} \
              --weight_decay ${WEIGHT_DECAY} \
              --lr_scheduler_type ${LR_SCHEDULER_TYPE} \
              --num_train_epochs ${NUM_TRAIN_EPOCHS} \
              --warmup_steps ${WARMUP_STEPS} \
              --seed ${SEED} \
              --load_dtype ${MODEL_DTYPE} \
              --dataloader_num_workers ${DATALOADER_NUM_WORKERS} \
              --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
              --max_grad_norm ${GRAD_CLIP} \
              --use_flash ${USE_FLASH} \
              --do_train ${DO_TRAIN} \
              --bf16 ${BF16_TRAINING} \
              --freeze_non_embed ${FREEZE_NON_EMBED} \
              --ddp_backend ${DDP_BACKEND} \
              --ADA_RATIO ${ADA_RATIO} \
              --ADA_TOPK ${ADA_TOPK} \
              --ADA_LOSS_WEIGHT ${ADA_LOSS_WEIGHT} \
              --ADA_MASK_WEIGHT ${ADA_MASK_WEIGHT} \
              --ADA_TOPK_WEIGHT ${ADA_TOPK_WEIGHT} \
              --ADA_ACT ${ADA_ACT} \
              "
              #   --max_steps ${MAX_STEPS} \

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