# learning_rate is from Fu Yao's Data Engineering paper
# weight_decay is removed
SCRIPT_DIR=$(dirname "$(realpath "$0")")
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}/.."
python codebase/args_parser.py --run_name test_run \
                      --model_dir ./PATH_TO_MODEL \
                      --tokenizer_dir ./PATH_TO_TOKENIZER \
                      --train_data_dir ./PATH_TO_TRAIN_DATA \
                      --eval_data_dir ./PATH_TO_EVAL_DATA \
                      --output_dir ./PATH_TO_SAVE_CKPT \
                      --per_device_eval_batch_size 1 \
                      --per_device_train_batch_size 1 \
                      --max_token_per_seq 2048 \
                      --gradient_accumulation_steps 1 \
                      --eval_steps 3 \
                      --save_steps 3 \
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
                      --max_grad_norm 1.0 \
                      --use_flash True \
                      --do_train True \
                      --bf16 True \
                      --freeze_non_embed True

python codebase/args_parser.py --run_name test_run \
                      --model_dir ./PATH_TO_MODEL \
                      --tokenizer_dir ./PATH_TO_TOKENIZER \
                      --train_data_dir ./PATH_TO_TRAIN_DATA \
                      --eval_data_dir ./PATH_TO_EVAL_DATA \
                      --output_dir ./PATH_TO_SAVE_CKPT \
                      --per_device_eval_batch_size 1 \
                      --per_device_train_batch_size 10 \
                      --gradient_accumulation_steps auto \
                      --max_token_per_seq 2048 \
                      --world_gpu_size 4 \
                      --target_token_per_batch 1e6 \
                      --eval_steps 3 \
                      --save_steps 3 \
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
                      --max_grad_norm 1.0 \
                      --use_flash True \
                      --do_train True \
                      --bf16 True \
                      --freeze_non_embed False