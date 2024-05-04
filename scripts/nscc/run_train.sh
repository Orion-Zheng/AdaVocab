#!/bin/bash
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

MODEL_DIR="experiment_models/tinyllama_expanded_empty"
TOKENIZER_DIR="experiment_models/tinyllama_expanded_empty"
# TRAIN_DATA_DIR="tokenized_datasets/skypile_2022_sampled_50M_2048_colossal_ft"  # for debug
TRAIN_DATA_DIR="tokenized_datasets/skypile_2022_sampled_5000M_2048_colossal_ft"  # for train
EVAL_DATA_DIR="tokenized_datasets/skypile_2023_sampled_100_eval_colossal_ft_"
OUTPUT_DIR="experiment_ckpts/tinyllama_expanded_frez_embed"

# DIST_CONFIG="config/accelerate_config/nscc/one_node_one_gpu.yaml"  # Single-Node, Single-GPU (for debug)
DIST_CONFIG="config/accelerate_config/nscc/one_node_4_gpu_zero2_offload.yaml"  # Single-Node, Multi-GPU + Deepspeed

export WANDB_PROJECT="baseline"
WANDB_RUN_NAME="nscc-tinyllama_training"

TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
WANDB_RUN_NAME="${WANDB_RUN_NAME}-${TIMESTAMP}"
OUTPUT_DIR="${OUTPUT_DIR}-${TIMESTAMP}"
# Situation 1: provide `gradient_accumulation_steps` directly
# 8 for 4 GPU --> 4(gpu) * 8(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
# 4 for 8 GPU --> 8(gpu) * 4(grad_acc) * 16(seq/gpu) * 2(K token/seq) = 1M tokens/batch
GRAD_ACC_STEP=8  
GRAD_CLIP=1.0
# Situation 2: provide `world_gpu_size` and `target_token_per_batch` and set `gradient_accumulation_steps` to `auto`
# TODO ...
# Deepspeed Usage: https://github.com/huggingface/accelerate/blob/b8c85839531ded28efb77c32e0ad85af2062b27a/docs/source/usage_guides/deepspeed.md?plain=1#L582
accelerate launch --config_file ${DIST_CONFIG} --gradient_accumulation_steps ${GRAD_ACC_STEP} --gradient_clipping ${GRAD_CLIP} --mixed_precision bf16 \
                  train.py \
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
                #   --resume_from_checkpoint experiment_ckpts/tinyllama_expanded_frez_embed-2024-04-12-221505/checkpoint-132/ \
                #   --save_total_limit 3 \
                #   --load_best_model_at_end True
                #   --warmup_steps 300 \  
                #   --quant_config_path config/quant_config/4bit_quant.json \
                #   --lora_config_path config/peft_config/lora.json \
                #   --max_steps 100 \
                  
                           
