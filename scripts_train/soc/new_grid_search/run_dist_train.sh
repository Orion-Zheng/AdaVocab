#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../.."
N_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)  # get N_NODE from slurm cluster
MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)  # get MAIN_HOST from slurm cluster
LOCAL_CONFIG="/tmp/my_soc_dist_config.yaml"  # Multi-Node, Multi-GPU
# Note: Above Environment Variables can be overwritten by the following commands

# Set Default Config
DEFAULT_CONFIG_FILE="scripts_train/soc/new_grid_search/default_dist_config.yaml"
source <(bash yaml2env_parser.sh --config $DEFAULT_CONFIG_FILE)
# Update Customized Config
if [ "$#" -eq 1 ]; then
    CUSTOM_CONFIG_FILE="$1"
    source <(bash yaml2env_parser.sh --config $CUSTOM_CONFIG_FILE)
fi

# Add TimeStamp to Run Name and Output Dir for better tracking
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"

# Start Multi-Node Training
echo "Number of Nodes: " $N_NODE "; GPU per Node: " $GPU_PER_NODE
echo "Main Host Address: "$MAIN_HOST
echo "Dispatching Accelerate Configs to All Nodes..."
# 1. Dispatch accelerate config to all nodes with correct `machine_rank` on each node by mpirun -np xxx (PBS Pro) or srun -N xxx (Slurm)
NODE_LIST=$(echo $(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_LIST_COMMA=$(echo "$(scontrol show hostnames "$SLURM_JOB_NODELIST")" | tr '\n' ',' | sed 's/,$//')
srun -N $N_NODE -w $NODE_LIST_COMMA --ntasks-per-node=1 python codebase/tools/dist_env/set_dist_config.py \
                        --node_list ${NODE_LIST} \
                        --gpu_per_node ${GPU_PER_NODE} \
                        --main_ip ${MAIN_HOST} \
                        --default_config ${ACCELERATE_CONFIG} \
                        --local_save_path ${LOCAL_CONFIG}

# # 2. Run Multi-node Training
# # Deepspeed Usage: https://github.com/huggingface/accelerate/blob/b8c85839531ded28efb77c32e0ad85af2062b27a/docs/source/usage_guides/deepspeed.md?plain=1#L582
LAUNCH_SCRIPT="accelerate launch --config_file ${LOCAL_CONFIG} --gradient_accumulation_steps ${GRAD_ACC_STEP} --gradient_clipping ${GRAD_CLIP} --mixed_precision bf16"
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
echo "$LAUNCH_SCRIPT $TRAIN_SCRIPT"
# # Run Multi-Node Training
srun -N $N_NODE --ntasks-per-node=$GPU_PER_NODE -w $NODE_LIST_COMMA $LAUNCH_SCRIPT $TRAIN_SCRIPT
