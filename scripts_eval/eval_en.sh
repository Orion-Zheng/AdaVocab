# 1. Evaluate Original Model
ORIGIN_CKPT="experiment_models/tinyllama_expanded_empty"
lm_eval --model hf \
    --model_args pretrained=${ORIGIN_CKPT} \
    --tasks hellaswag,xnli_en,xwinograd_en \
    --device cuda:0 \
    --batch_size 64 \
    --output_path eval/eval_result/en/before_train.json

# 2. Evaluate Freezed except Embedding
STEPS=(200 400 800 1600)
FREZ_CKPT_DIR="experiment_ckpts/tinyllama_expanded_frez_embed-2024-04-16-223017"
# Iterate over the steps in the STEPS array
for STEP in "${STEPS[@]}"; do
  MODEL_CKPT="$FREZ_CKPT_DIR/checkpoint-$STEP"
  # Execute the lm_eval command
  lm_eval --model hf \
    --model_args pretrained=${MODEL_CKPT} \
    --tasks hellaswag,xnli_en,xwinograd_en \
    --device cuda:0 \
    --batch_size 64 \
    --output_path eval/eval_result/en/freeze/step_$STEP.json
done

# 3. Evaluate Full-Finetuned Model
STEPS=(400 800 1600 6000)
FULL_CKPT_DIR="experiment_ckpts/tinyllama_expanded-2024-03-03-10-26-24"
for STEP in "${STEPS[@]}"; do
  MODEL_CKPT="$FULL_CKPT_DIR/checkpoint-$STEP"
  # Execute the lm_eval command
  lm_eval --model hf \
    --model_args pretrained=${MODEL_CKPT} \
    --tasks hellaswag,xnli_en,xwinograd_en \
    --device cuda:0 \
    --batch_size 64 \
    --output_path eval/eval_result/en/full/step_$STEP.json
done