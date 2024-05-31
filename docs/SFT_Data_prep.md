```
python codebase/tools/data_prep/tokenize_dataset.py \
       --tokenizer ./original_models/tinyllama-chat \
       --source_dir ./original_datasets/wildchat-1M \
       --split train \
       --output_dir ./tokenized_datasets \
       --sft

PYTHONPATH=. python codebase/tools/data_prep/pack_const_token_dataset.py \
       --tokenizer original_models/tinyllama-chat \
       --token_field input_ids \
       --seq_num_token 2048 \
       --buffer_size 100000 \
       --source_dir tokenized_datasets/wildchat-1M_tinyllama-chat_sft \
       --output_dir tokenized_datasets/wildchat-1M_tinyllama-chat_2048_sft
       
python codebase/tools/data_prep/train_test_split.py \
       --dataset_path tokenized_datasets/wildchat-1M_tinyllama-chat_2048_sft \
       --output_dir tokenized_datasets/wildchat-1M_tinyllama-chat_2048_sft_split \
       --test_size 1024 
```