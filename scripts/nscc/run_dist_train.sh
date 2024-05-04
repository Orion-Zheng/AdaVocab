#!/bin/bash
export NCCL_DEBUG=INFO
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
# Distributed Arguments
GPU_PER_NODE=2
N_NODE=$(wc -l < $PBS_NODEFILE)  # GET N_NODE FROM PBS PRO CLUSTER
MAIN_HOST=$(hostname)
WORK_ABS_DIR="/home/users/nus/e0792473/scratch/EfficientVocabExtend"
ACCELERATE_CONFIG="config/accelerate_config/nscc/multi_node_zero2_offload_template.yaml"  # Set up the accelerate config template
# Training Arguments
export WANDB_PROJECT="NSCC_PROJ_Baseline"
WANDB_RUN_NAME="nscc-tinyllama_training"
MODEL_DIR="experiment_models/tinyllama_expanded_empty"
TOKENIZER_DIR="experiment_models/tinyllama_expanded_empty"
TRAIN_DATA_DIR="tokenized_datasets/skypile_2022_sampled_50M_2048_colossal_ft"  # for debug
# TRAIN_DATA_DIR="tokenized_datasets/skypile_2022_sampled_5000M_2048_colossal_ft"  # for train
EVAL_DATA_DIR="tokenized_datasets/skypile_2023_sampled_100_eval_colossal_ft_"
OUTPUT_DIR="experiment_ckpts/tinyllama_expanded_frez_embed"

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"
# 8 for 4 GPU --> 4(gpu) * 8(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
# 4 for 8 GPU --> 8(gpu) * 4(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
GRAD_ACC_STEP=8  
GRAD_CLIP=1.0

# Start Multi-Node Training
LOCAL_CONFIG="/tmp/my_nscc_dist_config.yaml"  # Multi-Node, Multi-GPU + Deepspeed
echo "Number of Nodes: " $N_NODE "; GPU per Node: " $GPU_PER_NODE
echo "Main Host Address: "$MAIN_HOST
echo "Dispatching Accelerate Configs to All Nodes..."
# 1. Dispatch accelerate config to all nodes with correct `machine_rank` on each node (mpirun -np xxx or srun -N xxx)
mpirun -np $N_NODE python dist_env_tools/set_dist_config.py --n_node ${N_NODE} --gpu_per_node ${GPU_PER_NODE} \
                        --main_ip ${MAIN_HOST} \
                        --default_config ${ACCELERATE_CONFIG} \
                        --local_save_path ${LOCAL_CONFIG}
# 2. Run Multi-node Training
# Deepspeed Usage: https://github.com/huggingface/accelerate/blob/b8c85839531ded28efb77c32e0ad85af2062b27a/docs/source/usage_guides/deepspeed.md?plain=1#L582
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
              --per_device_train_batch_size 16 \
              --max_token_per_seq 2048 \
              --eval_steps 100 \
              --save_steps 200 \
              --learning_rate 2e-5 \
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
              --freeze_embed True \
              "

if [ -z "$PBS_NODEFILE" ]; then
  echo "PBS_NODEFILE is not set. Are you running this script in a PBS environment?"
  exit 1
fi
HOSTLIST=$(paste -sd, $PBS_NODEFILE)
export PDSH_RCMD_TYPE=ssh
trap 'echo "Terminating accelerate processes on all nodes..."; pdsh -w $HOSTLIST "pkill -u $USER -f accelerate"' SIGINT
pdsh -w $HOSTLIST "bash --login -c 'cd $WORK_ABS_DIR && $LAUNCH_SCRIPT $TRAIN_SCRIPT'"

