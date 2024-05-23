#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/../.."
export GLOO_SOCKET_IFNAME="ens6f0np0" 
# Distributed Arguments
GPU_PER_NODE=2
N_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)  # GET N_NODE FROM SLURM CLUSTER
MAIN_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
WORK_ABS_DIR="/home/z/zheng22/AdaVocab"
ACCELERATE_CONFIG="config/accelerate/nscc/multi_node_template.yaml"  # Set up the accelerate config template
# ACCELERATE_CONFIG="config/accelerate/nscc/multi_node_zero2_offload_template.yaml"  # Deepspeed not Supported yet
# Training Arguments
export WANDB_PROJECT="AdaVocab_0522_wildchat_1M_data"
WANDB_RUN_NAME="AdaVocab_2_epoch_topk_800_lr_2e-6"
MODEL_DIR="original_models/tinyllama-chat"
TOKENIZER_DIR="original_models/tinyllama-chat"
TRAIN_DATA_DIR="tokenized_datasets/wildchat_1M_tinyllama-chat_2048_ft_split/train"  
EVAL_DATA_DIR="tokenized_datasets/wildchat_1M_tinyllama-chat_2048_ft_split/eval"
OUTPUT_DIR="experiment_ckpts/AdaVocab_0522"

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"
# 8 for 4 GPU --> 4(gpu) * 8(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
# 4 for 8 GPU --> 8(gpu) * 4(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
GRAD_ACC_STEP=4  
GRAD_CLIP=1.0

# Start Multi-Node Training
LOCAL_CONFIG="/tmp/my_soc_dist_config.yaml"  # Multi-Node, Multi-GPU
echo "Number of Nodes: " $N_NODE "; GPU per Node: " $GPU_PER_NODE
echo "Main Host Address: "$MAIN_HOST
echo "Dispatching Accelerate Configs to All Nodes..."
# 1. Dispatch accelerate config to all nodes with correct `machine_rank` on each node (mpirun -np xxx or srun -N xxx)
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
              --per_device_eval_batch_size 1 \
              --per_device_train_batch_size 32 \
              --max_token_per_seq 2048 \
              --eval_steps 100 \
              --d 1000 \
              --learning_rate 2e-6 \
              --optim paged_adamw_32bit --adam_beta1 0.9 --adam_beta2 0.95 \
              --weight_decay 0.01 \
              --lr_scheduler_type cosine \
              --num_train_epochs 1 \
              --warmup_ratio 0.03 \
              --seed 42 \
              --load_dtype bfloat16 \
              --dataloader_num_workers 0 \
              --gradient_checkpointing True \
              --max_grad_norm ${GRAD_CLIP} \
              --use_flash True \
              --do_train True \
              --bf16 True \
              --freeze_non_embed False \
              --ddp_backend gloo
              "
echo "$LAUNCH_SCRIPT $TRAIN_SCRIPT"
# # Run Multi-Node Training
srun -N $N_NODE --ntasks-per-node=$GPU_PER_NODE -w $NODE_LIST_COMMA $LAUNCH_SCRIPT $TRAIN_SCRIPT
