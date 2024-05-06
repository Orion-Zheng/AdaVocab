# 1. Dowload Dataset and Model
## 1.1 Download Dataset
The argument `--token` is needed if you are acessing a private repo or repo with signed license argeement.
```
python data_preparation/download_hf_repo.py \
       --repo_type dataset \
       --repo_id allenai/WildChat \
       --local_dir ./original_datasets/wildchat
       [--token YOUR_HF_TOKEN ]
```
## 1.2 Download Model with Tokenizer
```
python data_preparation/download_hf_repo.py \
       --repo_type model \
       --repo_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --local_dir ./original_models/tinyllama-chat \
       [--token YOUR_HF_TOKEN ]
```

# 2. Tokenize Dataset
## 2.1 Tokenize the Whole Dataset
```
python data_preparation/tokenize_dataset.py \
       --tokenizer ./original_models/tinyllama-chat \
       --source_dir ./original_datasets/wildchat \
       --split train \
       --output_dir ./tokenized_datasets 
```
## 2.2 Tokenize Given Tokens from a Huge Dataset
```

```

# 3. Packing Dataset
```
python data_preparation/constant_tokens_batch_dataset.py \
       --tokenizer original_models/tinyllama-chat \
       --token_field input_ids \
       --seq_num_token 2048 \
       --buffer_size 100000 \
       --source_dir tokenized_datasets/wildchat_tinyllama-chat_ft \
       --output_dir tokenized_datasets/wildchat_tinyllama-chat_2048_ft
```

# 4. Sample Evaluation Dataset
```
python data_preparation/select_given_samples.py \
       --dataset_path tokenized_datasets/wildchat_tinyllama-chat_2048_ft \
       --output_path tokenized_datasets/wildchat_tinyllama-chat_1M_eval_fake \
       --num_samples 500 \
       --num_proc 100 \
       --seed 42 \
       --columns input_ids labels
```

# 5. Run Training
```
bash scripts/soc/run_train.sh
```