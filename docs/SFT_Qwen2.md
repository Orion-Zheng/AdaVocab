1. Download Model and Checkpoint
```
python codebase/tools/huggingface/download_hf_repo.py \
       --repo_type model \
       --repo_id Qwen/Qwen2-1.5B \
       --local_dir original_models/Qwen2-1.5B
```

2. Prepare SFT Data
```
PYTHONPATH=. python codebase/tools/data_prep/tokenize_dataset.py \
       --tokenizer Qwen/Qwen2-1.5B-Instruct  \
       --source_dir original_datasets/wildchat-1M \
       --split train \
       --output_dir tokenized_datasets \
       --sft

PYTHONPATH=. python codebase/tools/data_prep/pack_const_token_dataset.py \
       --tokenizer Qwen/Qwen2-1.5B-Instruct \
       --token_field input_ids \
       --seq_num_token 2048 \
       --buffer_size 100000 \
       --source_dir tokenized_datasets/wildchat-1M_Qwen2-1.5B-Instruct_sft \
       --output_dir tokenized_datasets/wildchat-1M_Qwen2-Instruct_2048_sft

PYTHONPATH=. python codebase/tools/data_prep/train_test_split.py \
       --dataset_path tokenized_datasets/wildchat-1M_Qwen2-Instruct_2048_sft \
       --output_dir tokenized_datasets/wildchat-1M_Qwen2_2048_sft_split \
       --test_size 1024 
```
Total Tokens in Dataset: 1602758656 --> 1.6B

```
PYTHONPATH=. python codebase/tools/data_prep/tokenize_dataset.py \
       --tokenizer Qwen/Qwen2-1.5B-Instruct  \
       --source_dir original_datasets/wildchat-1M \
       --split train \
       --output_dir tokenized_datasets \
       --sft --sft_no_mask

PYTHONPATH=. python codebase/tools/data_prep/pack_const_token_dataset.py \
       --tokenizer Qwen/Qwen2-1.5B-Instruct \
       --token_field input_ids \
       --seq_num_token 2048 \
       --buffer_size 100000 \
       --source_dir tokenized_datasets/wildchat-1M_Qwen2-1.5B-Instruct_sft_no_mask \
       --output_dir tokenized_datasets/wildchat-1M_Qwen2-Instruct_2048_sft_no_mask

PYTHONPATH=. python codebase/tools/data_prep/train_test_split.py \
       --dataset_path tokenized_datasets/wildchat-1M_Qwen2-Instruct_2048_sft_no_mask \
       --output_dir tokenized_datasets/wildchat-1M_Qwen2_2048_sft_no_mask_split \
       --test_size 1024 
```
3. Start Training
```
bash scripts_train/soc/experiments/launch_Qwen2-1.5B_sft.sh
```